import numpy as np
from md_integration import MDSimulation


def test_md_run_local_diffusion():
    sim = MDSimulation()
    diameters = [5, 10]
    results = sim.run_local_diffusion(diameters, mobility=0.5, num_steps=50)
    assert not results.empty
    assert set(results['particle_diameter']) == set(diameters)
    print("✓ MDSimulation local diffusion works")
