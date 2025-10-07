"""
Nuclear Diffusion Simulator
Multi-compartment nuclear diffusion simulation based on DNA and splicing factor intensities.
Implements physics from github.com/mhendzel2/nuclear-diffusion-si

Compartment Classification:
- Nucleolus: Low DNA, Low SF
- Heterochromatin: High DNA, Low SF  
- Speckles: High SF, Low DNA
- Euchromatin: Moderate DNA, Low SF
- Nucleoplasm: Everything else
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import warnings

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Visualization features will be limited.")


# ==================== CONSTANTS ====================

BOLTZMANN_CONSTANT = 1.38e-23  # J/K
PIXEL_TO_NM = 65  # nm per pixel (from nuclear-diffusion-si)
WATER_VISCOSITY = 0.001  # Pa·s
ELEMENTARY_CHARGE = 1.602e-19  # Coulombs


# ==================== DATA CLASSES ====================

class CompartmentType(Enum):
    """Nuclear compartment types"""
    NUCLEOPLASM = "nucleoplasm"
    NUCLEOLUS = "nucleolus"
    HETEROCHROMATIN = "heterochromatin"
    SPECKLES = "speckles"
    EUCHROMATIN = "euchromatin"


class DiffusionModel(Enum):
    """Diffusion model types"""
    NORMAL = "normal"
    ANOMALOUS_SUBDIFFUSIVE = "anomalous-subdiffusive"
    ANOMALOUS_SUPERDIFFUSIVE = "anomalous-superdiffusive"
    FRACTIONAL_BROWNIAN = "fractional-brownian"


@dataclass
class ParticleProperties:
    """Properties of a simulated particle"""
    radius: float  # nm
    charge: float = 0.0  # elementary charges
    shape: str = 'spherical'
    surface_chemistry: str = 'neutral'  # neutral, positive, negative, pegylated
    binding_affinity: float = 0.0  # 0-1, probability of binding


@dataclass
class MediumProperties:
    """Properties of a compartment medium"""
    viscosity: float = 1.0  # relative to water
    pore_size: Optional[float] = None  # nm, for gel-like media
    crowding_factor: float = 0.0  # 0-1
    electrostatic_potential: float = 0.0  # mV
    diffusion_model: DiffusionModel = DiffusionModel.NORMAL
    alpha: float = 1.0  # anomalous diffusion exponent
    tau: float = 0.01  # characteristic time for CTRW


@dataclass
class Compartment:
    """Nuclear compartment definition"""
    type: CompartmentType
    center: Tuple[float, float]  # pixels
    radius: Optional[float] = None  # pixels, for circular compartments
    width: Optional[float] = None  # pixels, for elliptical/speckles
    height: Optional[float] = None  # pixels
    medium: MediumProperties = None
    permeability: float = 1.0  # boundary permeability 0-1
    
    def __post_init__(self):
        if self.medium is None:
            self.medium = self._default_medium()
    
    def _default_medium(self) -> MediumProperties:
        """Set default medium properties based on compartment type"""
        if self.type == CompartmentType.NUCLEOLUS:
            return MediumProperties(
                viscosity=2.0,
                diffusion_model=DiffusionModel.ANOMALOUS_SUBDIFFUSIVE,
                alpha=0.7,
                crowding_factor=0.5
            )
        elif self.type == CompartmentType.HETEROCHROMATIN:
            return MediumProperties(
                viscosity=3.0,
                pore_size=50.0,
                diffusion_model=DiffusionModel.ANOMALOUS_SUBDIFFUSIVE,
                alpha=0.6,
                crowding_factor=0.7
            )
        elif self.type == CompartmentType.SPECKLES:
            return MediumProperties(
                viscosity=1.5,
                diffusion_model=DiffusionModel.ANOMALOUS_SUBDIFFUSIVE,
                alpha=0.75,
                crowding_factor=0.4
            )
        elif self.type == CompartmentType.EUCHROMATIN:
            return MediumProperties(
                viscosity=1.2,
                diffusion_model=DiffusionModel.NORMAL,
                alpha=1.0,
                crowding_factor=0.2
            )
        else:  # NUCLEOPLASM
            return MediumProperties(
                viscosity=1.0,
                diffusion_model=DiffusionModel.NORMAL,
                alpha=1.0,
                crowding_factor=0.1
            )
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is inside this compartment"""
        dx = x - self.center[0]
        dy = y - self.center[1]
        
        if self.radius is not None:
            # Circular compartment
            return (dx**2 + dy**2) <= self.radius**2
        elif self.width is not None and self.height is not None:
            # Elliptical/rectangular compartment
            return (dx / self.width)**2 + (dy / self.height)**2 <= 1.0
        else:
            return False


# ==================== NUCLEAR GEOMETRY ====================

class NuclearGeometry:
    """Defines nuclear architecture with compartments"""
    
    def __init__(self, width: float = 450, height: float = 250):
        """
        Create nuclear geometry.
        
        Parameters
        ----------
        width : float
            Nucleus width in pixels
        height : float
            Nucleus height in pixels
        """
        self.width = width
        self.height = height
        self.center = (width / 2, height / 2)
        self.compartments: List[Compartment] = []
        
        # Create default architecture
        self._create_default_geometry()
    
    def _create_default_geometry(self):
        """Create default nuclear architecture (from nuclear-diffusion-si)"""
        cx, cy = self.center
        
        # Nucleoplasm (base ellipse)
        nucleoplasm = Compartment(
            type=CompartmentType.NUCLEOPLASM,
            center=self.center,
            width=self.width / 2,
            height=self.height / 2
        )
        self.compartments.append(nucleoplasm)
        
        # Nucleolus (center, 125nm radius = ~1.9 pixels at 65nm/pixel)
        nucleolus = Compartment(
            type=CompartmentType.NUCLEOLUS,
            center=self.center,
            radius=125 / PIXEL_TO_NM,
            permeability=0.3
        )
        self.compartments.append(nucleolus)
        
        # Heterochromatin (3 spheres, 30px radius)
        hetero_positions = [
            (cx - 80, cy - 40),
            (cx + 70, cy + 30),
            (cx - 40, cy + 60)
        ]
        for pos in hetero_positions:
            hetero = Compartment(
                type=CompartmentType.HETEROCHROMATIN,
                center=pos,
                radius=30,
                permeability=0.4
            )
            self.compartments.append(hetero)
        
        # Speckles (10 elongated structures, 30x10 pixels)
        n_speckles = 10
        placement_radius = (self.width / 4 + nucleolus.radius + 20)
        for i in range(n_speckles):
            angle = (i / n_speckles) * 2 * np.pi
            x = cx + placement_radius * np.cos(angle)
            y = cy + placement_radius * np.sin(angle)
            
            speckle = Compartment(
                type=CompartmentType.SPECKLES,
                center=(x, y),
                width=15,  # half-width
                height=5,   # half-height
                permeability=0.6
            )
            self.compartments.append(speckle)
        
        # Euchromatin (fills spaces between structures)
        euchromatin = Compartment(
            type=CompartmentType.EUCHROMATIN,
            center=self.center,
            width=self.width / 2.5,
            height=self.height / 2.5,
            permeability=0.8
        )
        self.compartments.insert(1, euchromatin)  # Insert after nucleoplasm
    
    def classify_point(self, x: float, y: float) -> CompartmentType:
        """Determine which compartment a point belongs to (priority order)"""
        # Check specific compartments first (higher priority)
        for comp in reversed(self.compartments):
            if comp.type != CompartmentType.NUCLEOPLASM and comp.contains_point(x, y):
                return comp.type
        
        # Default to nucleoplasm
        return CompartmentType.NUCLEOPLASM
    
    def get_compartment(self, comp_type: CompartmentType) -> Optional[Compartment]:
        """Get the first compartment of a given type"""
        for comp in self.compartments:
            if comp.type == comp_type:
                return comp
        return None
    
    @classmethod
    def from_images(cls, dna_image: np.ndarray, sf_image: np.ndarray,
                   dna_thresholds: Tuple[float, float, float] = (0.2, 0.5, 0.8),
                   sf_thresholds: Tuple[float, float, float] = (0.2, 0.5, 0.8)) -> 'NuclearGeometry':
        """
        Create geometry from DNA and splicing factor images.
        
        Parameters
        ----------
        dna_image : np.ndarray
            DNA channel (normalized 0-1)
        sf_image : np.ndarray
            Splicing factor channel (normalized 0-1)
        dna_thresholds : tuple
            (low, mid, high) thresholds for DNA
        sf_thresholds : tuple
            (low, mid, high) thresholds for SF
            
        Returns
        -------
        NuclearGeometry
            Geometry with compartments derived from images
        """
        height, width = dna_image.shape
        geometry = cls(width, height)
        geometry.compartments.clear()
        
        # Classify each pixel
        dna_low, dna_mid, dna_high = dna_thresholds
        sf_low, sf_mid, sf_high = sf_thresholds
        
        classification = np.zeros(dna_image.shape, dtype=int)
        
        for i in range(height):
            for j in range(width):
                dna_val = dna_image[i, j]
                sf_val = sf_image[i, j]
                
                # Classification logic (from nuclear-diffusion-si)
                if dna_val <= dna_low and sf_val <= sf_low:
                    classification[i, j] = 1  # Nucleolus
                elif dna_val >= dna_high and sf_val <= sf_low:
                    classification[i, j] = 2  # Heterochromatin
                elif sf_val >= sf_high and dna_val <= dna_low:
                    classification[i, j] = 3  # Speckles
                elif dna_val > dna_low and dna_val <= dna_mid and sf_val <= sf_low:
                    classification[i, j] = 4  # Euchromatin
                else:
                    classification[i, j] = 0  # Nucleoplasm
        
        # Create compartments from classification
        # (simplified - full implementation would extract connected components)
        geometry.compartments.append(Compartment(
            type=CompartmentType.NUCLEOPLASM,
            center=(width/2, height/2),
            width=width/2,
            height=height/2
        ))
        
        return geometry


# ==================== PHYSICS ENGINE ====================

class PhysicsEngine:
    """Physics simulation for particle diffusion"""
    
    def __init__(self, temperature: float = 310, time_step: float = 0.001):
        """
        Parameters
        ----------
        temperature : float
            Temperature in Kelvin (default 310K = 37°C)
        time_step : float
            Time step in seconds
        """
        self.temperature = temperature
        self.time_step = time_step
        self.rng = np.random.default_rng(42)
    
    def calculate_diffusion_coefficient(self, particle: ParticleProperties,
                                       medium: MediumProperties) -> float:
        """
        Calculate diffusion coefficient using Stokes-Einstein relation.
        
        D = kT / (6πηr)
        
        Modified for:
        - Surface chemistry (PEGylation reduces viscosity interaction)
        - Pore size (hindrance in gel media)
        - Crowding (reduces effective D)
        """
        # Base Stokes-Einstein
        radius_m = particle.radius * 1e-9  # nm to m
        eta = WATER_VISCOSITY * medium.viscosity
        
        D = (BOLTZMANN_CONSTANT * self.temperature) / (6 * np.pi * eta * radius_m)
        
        # Surface chemistry modifier
        if particle.surface_chemistry == 'pegylated':
            D *= 1.2  # Reduced friction
        
        # Pore size hindrance (for gel media)
        if medium.pore_size is not None:
            hindrance = np.exp(-particle.radius / medium.pore_size)
            D *= hindrance
        
        # Crowding effect
        D *= (1.0 - medium.crowding_factor)
        
        return D  # m²/s
    
    def generate_displacement(self, particle: ParticleProperties, 
                             medium: MediumProperties,
                             diffusion_coeff: float) -> Tuple[float, float]:
        """
        Generate random displacement based on diffusion model.
        
        Returns
        -------
        dx, dy : float
            Displacement in meters
        """
        if medium.diffusion_model == DiffusionModel.NORMAL:
            return self._normal_displacement(diffusion_coeff)
        
        elif medium.diffusion_model == DiffusionModel.ANOMALOUS_SUBDIFFUSIVE:
            return self._ctrw_displacement(diffusion_coeff, medium.alpha, medium.tau)
        
        elif medium.diffusion_model == DiffusionModel.FRACTIONAL_BROWNIAN:
            return self._fbm_displacement(diffusion_coeff, medium.alpha)
        
        else:
            return self._normal_displacement(diffusion_coeff)
    
    def _normal_displacement(self, D: float) -> Tuple[float, float]:
        """Normal Brownian motion: <r²> = 4Dt"""
        sigma = np.sqrt(2 * D * self.time_step)
        dx = self.rng.normal(0, sigma)
        dy = self.rng.normal(0, sigma)
        return dx, dy
    
    def _ctrw_displacement(self, D: float, alpha: float, tau: float) -> Tuple[float, float]:
        """
        Continuous-Time Random Walk for subdiffusion.
        Uses power-law waiting time distribution.
        """
        # Generate waiting time from power-law distribution
        u = self.rng.uniform(0, 1)
        waiting_time = tau * (u ** (-1.0 / alpha))
        
        # Displacement magnitude scales with waiting time
        effective_time = min(waiting_time, self.time_step)
        sigma = np.sqrt(2 * D * effective_time**alpha)
        
        dx = self.rng.normal(0, sigma)
        dy = self.rng.normal(0, sigma)
        return dx, dy
    
    def _fbm_displacement(self, D: float, H: float) -> Tuple[float, float]:
        """
        Fractional Brownian motion.
        H < 0.5: subdiffusion (anti-persistent)
        H = 0.5: normal diffusion
        H > 0.5: superdiffusion (persistent)
        """
        # Simplified fBm (proper implementation would use Cholesky decomposition)
        sigma = np.sqrt(2 * D * self.time_step**(2*H))
        dx = self.rng.normal(0, sigma)
        dy = self.rng.normal(0, sigma)
        return dx, dy
    
    def calculate_electrostatic_force(self, particle: ParticleProperties,
                                      medium: MediumProperties,
                                      position: Tuple[float, float]) -> Tuple[float, float]:
        """
        Calculate electrostatic force on particle.
        F = qE where E is from medium potential
        """
        if particle.charge == 0 or medium.electrostatic_potential == 0:
            return 0.0, 0.0
        
        # Simplified: assume radial field from compartment center
        # Real implementation would use actual potential gradient
        E_field = medium.electrostatic_potential * 1e-3  # mV to V
        force_magnitude = particle.charge * ELEMENTARY_CHARGE * E_field
        
        # Force direction (toward/away from center depending on charge)
        # This is a simplified model
        fx = force_magnitude * 1e-12  # Scale to reasonable force
        fy = force_magnitude * 1e-12
        
        return fx, fy
    
    def check_binding(self, particle: ParticleProperties, 
                     compartment: Compartment) -> bool:
        """Check if particle binds to compartment structure"""
        if particle.binding_affinity > 0:
            bind_prob = particle.binding_affinity * self.time_step * 1000  # Scale to reasonable probability
            return self.rng.random() < bind_prob
        return False


# ==================== PARTICLE SIMULATOR ====================

@dataclass
class Particle:
    """Simulated particle with trajectory"""
    id: int
    properties: ParticleProperties
    position: np.ndarray  # [x, y] in pixels
    velocity: np.ndarray = None  # [vx, vy] in pixels/s
    trajectory: List[np.ndarray] = None
    current_compartment: CompartmentType = CompartmentType.NUCLEOPLASM
    compartment_history: List[Tuple[CompartmentType, float]] = None
    bound_until: Optional[float] = None
    
    def __post_init__(self):
        if self.velocity is None:
            self.velocity = np.zeros(2)
        if self.trajectory is None:
            self.trajectory = [self.position.copy()]
        if self.compartment_history is None:
            self.compartment_history = [(self.current_compartment, 0.0)]


class NuclearDiffusionSimulator:
    """Main simulator class"""
    
    def __init__(self, geometry: NuclearGeometry, physics: PhysicsEngine):
        self.geometry = geometry
        self.physics = physics
        self.particles: List[Particle] = []
        self.current_time = 0.0
    
    def add_particles(self, n_particles: int, particle_props: ParticleProperties):
        """Add particles at random positions in nucleoplasm"""
        for i in range(n_particles):
            # Find valid position in nucleoplasm
            max_attempts = 100
            for _ in range(max_attempts):
                x = self.physics.rng.uniform(0, self.geometry.width)
                y = self.physics.rng.uniform(0, self.geometry.height)
                
                # Check if inside nucleus
                comp_type = self.geometry.classify_point(x, y)
                if comp_type is not None:
                    particle = Particle(
                        id=len(self.particles),
                        properties=particle_props,
                        position=np.array([x, y])
                    )
                    particle.current_compartment = comp_type
                    self.particles.append(particle)
                    break
    
    def step(self):
        """Advance simulation by one time step"""
        for particle in self.particles:
            # Check if particle is bound
            if particle.bound_until is not None and self.current_time < particle.bound_until:
                continue
            
            # Get current compartment
            comp_type = self.geometry.classify_point(particle.position[0], particle.position[1])
            
            # Get compartment properties (or use nucleoplasm default)
            compartment = self.geometry.get_compartment(comp_type)
            if compartment is None:
                compartment = self.geometry.get_compartment(CompartmentType.NUCLEOPLASM)
            
            medium = compartment.medium
            
            # Calculate diffusion coefficient
            D = self.physics.calculate_diffusion_coefficient(particle.properties, medium)
            
            # Generate displacement
            dx_m, dy_m = self.physics.generate_displacement(particle.properties, medium, D)
            
            # Convert to pixels
            dx_px = dx_m / (PIXEL_TO_NM * 1e-9)
            dy_px = dy_m / (PIXEL_TO_NM * 1e-9)
            
            # Apply electrostatic force (if any)
            fx, fy = self.physics.calculate_electrostatic_force(
                particle.properties, medium, particle.position
            )
            # Force contribution to displacement (simplified)
            dx_px += fx * 1e6  # Scale factor
            dy_px += fy * 1e6
            
            # Update position
            new_position = particle.position + np.array([dx_px, dy_px])
            
            # Boundary handling (reflective at nucleus edge)
            new_position = self._enforce_boundaries(new_position)
            
            # Check for compartment crossing
            new_comp_type = self.geometry.classify_point(new_position[0], new_position[1])
            if new_comp_type != particle.current_compartment:
                # Apply permeability check
                new_compartment = self.geometry.get_compartment(new_comp_type)
                if new_compartment is not None:
                    if self.physics.rng.random() < new_compartment.permeability:
                        particle.current_compartment = new_comp_type
                        particle.compartment_history.append((new_comp_type, self.current_time))
                    else:
                        # Reflect at boundary
                        new_position = particle.position
            
            # Check for binding
            if self.physics.check_binding(particle.properties, compartment):
                particle.bound_until = self.current_time + 0.1  # Bind for 0.1s
            
            # Update particle
            particle.position = new_position
            particle.trajectory.append(new_position.copy())
        
        self.current_time += self.physics.time_step
    
    def _enforce_boundaries(self, position: np.ndarray) -> np.ndarray:
        """Keep particle inside nucleus (reflective boundaries)"""
        x, y = position
        cx, cy = self.geometry.center
        a = self.geometry.width / 2
        b = self.geometry.height / 2
        
        # Ellipse equation: (x-cx)²/a² + (y-cy)²/b² <= 1
        dx = x - cx
        dy = y - cy
        
        # Check if outside
        if (dx/a)**2 + (dy/b)**2 > 1.0:
            # Reflect back inside
            angle = np.arctan2(dy, dx)
            x = cx + a * 0.95 * np.cos(angle)
            y = cy + b * 0.95 * np.sin(angle)
        
        return np.array([x, y])
    
    def run(self, n_steps: int, progress_callback: Optional[callable] = None):
        """Run simulation for n steps"""
        for step in range(n_steps):
            self.step()
            if progress_callback is not None:
                progress_callback(step, n_steps)
    
    def get_tracks_dataframe(self) -> pd.DataFrame:
        """
        Convert particle trajectories to standard tracks DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Columns: [track_id, frame, x, y, compartment]
        """
        records = []
        for particle in self.particles:
            for frame, pos in enumerate(particle.trajectory):
                # Determine compartment at this position
                comp_type = self.geometry.classify_point(pos[0], pos[1])
                
                records.append({
                    'track_id': particle.id,
                    'frame': frame,
                    'x': pos[0],
                    'y': pos[1],
                    'compartment': comp_type.value if comp_type else 'nucleoplasm'
                })
        
        return pd.DataFrame(records)
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Calculate summary statistics for all particles"""
        stats = {
            'n_particles': len(self.particles),
            'total_steps': len(self.particles[0].trajectory) if self.particles else 0,
            'simulation_time': self.current_time,
            'by_compartment': {}
        }
        
        # Count particles by compartment
        for comp_type in CompartmentType:
            count = sum(1 for p in self.particles if p.current_compartment == comp_type)
            stats['by_compartment'][comp_type.value] = count
        
        return stats


# ==================== HIGH-LEVEL API ====================

def simulate_nuclear_diffusion(n_particles: int = 100,
                               particle_radius: float = 40,  # nm
                               n_steps: int = 1000,
                               time_step: float = 0.001,  # seconds
                               temperature: float = 310,  # Kelvin
                               geometry: Optional[NuclearGeometry] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    High-level function to run nuclear diffusion simulation.
    
    Parameters
    ----------
    n_particles : int
        Number of particles to simulate
    particle_radius : float
        Particle radius in nanometers (20, 40, or 100 typical)
    n_steps : int
        Number of simulation steps
    time_step : float
        Time step in seconds
    temperature : float
        Temperature in Kelvin
    geometry : NuclearGeometry, optional
        Custom geometry, or None for default
        
    Returns
    -------
    tracks_df : pd.DataFrame
        Trajectory data
    summary : dict
        Simulation summary statistics
    """
    # Create geometry if not provided
    if geometry is None:
        geometry = NuclearGeometry()
    
    # Create physics engine
    physics = PhysicsEngine(temperature=temperature, time_step=time_step)
    
    # Create simulator
    simulator = NuclearDiffusionSimulator(geometry, physics)
    
    # Create particle properties
    particle_props = ParticleProperties(
        radius=particle_radius,
        charge=0.0,
        shape='spherical',
        surface_chemistry='neutral'
    )
    
    # Add particles
    simulator.add_particles(n_particles, particle_props)
    
    # Run simulation
    simulator.run(n_steps)
    
    # Get results
    tracks_df = simulator.get_tracks_dataframe()
    summary = simulator.get_summary_statistics()
    
    # Add simulation parameters to summary
    summary['parameters'] = {
        'n_particles': n_particles,
        'particle_radius_nm': particle_radius,
        'n_steps': n_steps,
        'time_step_s': time_step,
        'temperature_K': temperature
    }
    
    return tracks_df, summary
