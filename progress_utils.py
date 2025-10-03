"""
Progress Utilities

Enhanced progress feedback system for long-running operations.
Provides real-time progress updates, ETA calculation, and cancellation support.
"""

import streamlit as st
import time
from typing import Callable, Optional, Any, List
from dataclasses import dataclass
from datetime import timedelta
import threading


@dataclass
class ProgressStep:
    """Single step in a progress sequence."""
    name: str
    weight: float = 1.0  # Relative weight for progress calculation
    description: str = ""


class AnalysisProgress:
    """
    Rich progress feedback for long operations with ETA and cancellation.
    
    Features:
    - Real-time progress bar
    - ETA calculation
    - Step-by-step status
    - Cancellation support
    - Memory usage tracking (optional)
    - Custom messages
    
    Example:
        progress = AnalysisProgress("Diffusion Analysis", total_steps=100)
        
        for i in range(100):
            if not progress.update(i + 1, f"Processing track {i}"):
                break  # Cancelled by user
            
            # Do work
            process_track(i)
        
        progress.complete("Analysis finished!")
    """
    
    def __init__(self, 
                 title: str,
                 total_steps: int,
                 show_eta: bool = True,
                 show_cancel: bool = True,
                 show_memory: bool = False):
        """
        Initialize progress tracker.
        
        Parameters
        ----------
        title : str
            Title displayed above progress bar
        total_steps : int
            Total number of steps for 100% completion
        show_eta : bool
            Display estimated time remaining
        show_cancel : bool
            Show cancellation button
        show_memory : bool
            Track and display memory usage
        """
        self.title = title
        self.total_steps = total_steps
        self.show_eta = show_eta
        self.show_cancel = show_cancel
        self.show_memory = show_memory
        
        # Create UI elements
        self.container = st.container()
        with self.container:
            st.subheader(f"⚙️ {title}")
            
            # Progress bar
            self.progress_bar = st.progress(0)
            
            # Status text
            self.status_text = st.empty()
            
            # ETA and memory in columns
            if self.show_eta or self.show_memory:
                col1, col2 = st.columns(2)
                with col1:
                    self.eta_text = st.empty() if self.show_eta else None
                with col2:
                    self.memory_text = st.empty() if self.show_memory else None
            else:
                self.eta_text = None
                self.memory_text = None
            
            # Cancel button
            if self.show_cancel:
                self.cancel_button = st.button(
                    "🛑 Cancel Operation", 
                    key=f"cancel_{title}_{time.time()}",
                    help="Stop the current operation"
                )
            else:
                self.cancel_button = False
        
        self.start_time = time.time()
        self.current_step = 0
        self.cancelled = False
        self.completed = False
    
    def update(self, step: int, message: str = "") -> bool:
        """
        Update progress.
        
        Parameters
        ----------
        step : int
            Current step number (1 to total_steps)
        message : str
            Status message to display
        
        Returns
        -------
        bool
            False if cancelled, True otherwise
        """
        if self.cancelled or self.completed:
            return not self.cancelled
        
        # Check for cancellation
        if self.show_cancel and self.cancel_button:
            self.cancelled = True
            self.status_text.markdown("⚠️ **Operation cancelled by user**")
            return False
        
        self.current_step = step
        
        # Update progress bar
        progress = min(step / self.total_steps, 1.0)
        self.progress_bar.progress(progress)
        
        # Update status
        status_msg = f"**Step {step}/{self.total_steps}**"
        if message:
            status_msg += f": {message}"
        self.status_text.markdown(status_msg)
        
        # Calculate and display ETA
        if self.eta_text and step > 0:
            elapsed = time.time() - self.start_time
            steps_remaining = self.total_steps - step
            
            if step >= 3:  # Need at least 3 steps for reliable estimate
                eta_seconds = (elapsed / step) * steps_remaining
                eta_delta = timedelta(seconds=int(eta_seconds))
                
                # Format ETA nicely
                if eta_seconds < 60:
                    eta_str = f"{int(eta_seconds)}s"
                elif eta_seconds < 3600:
                    minutes = int(eta_seconds // 60)
                    seconds = int(eta_seconds % 60)
                    eta_str = f"{minutes}m {seconds}s"
                else:
                    hours = int(eta_seconds // 3600)
                    minutes = int((eta_seconds % 3600) // 60)
                    eta_str = f"{hours}h {minutes}m"
                
                self.eta_text.markdown(f"⏱️ ETA: **{eta_str}**")
            else:
                self.eta_text.markdown("⏱️ ETA: *Calculating...*")
        
        # Display memory usage
        if self.memory_text:
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024 ** 2)
                self.memory_text.markdown(f"💾 Memory: **{memory_mb:.1f} MB**")
            except:
                pass
        
        return True
    
    def complete(self, message: str = "✓ Complete!"):
        """Mark progress as complete."""
        if not self.cancelled:
            self.completed = True
            self.progress_bar.progress(1.0)
            
            elapsed = time.time() - self.start_time
            elapsed_str = f"{elapsed:.1f}s"
            if elapsed >= 60:
                minutes = int(elapsed // 60)
                seconds = elapsed % 60
                elapsed_str = f"{minutes}m {seconds:.1f}s"
            
            self.status_text.markdown(
                f"✅ **{message}** (completed in {elapsed_str})"
            )
            
            if self.eta_text:
                self.eta_text.markdown("")  # Clear ETA


class MultiStepProgress:
    """
    Progress tracker for multi-stage operations.
    
    Example:
        steps = [
            ProgressStep("Load Data", weight=1.0, description="Loading tracks"),
            ProgressStep("Calculate MSD", weight=3.0, description="Computing MSD"),
            ProgressStep("Fit Diffusion", weight=2.0, description="Fitting curves"),
        ]
        
        progress = MultiStepProgress("Full Analysis", steps)
        
        progress.start_step(0)
        load_data()
        progress.complete_step()
        
        progress.start_step(1)
        for i in range(100):
            progress.update_substep(i, 100, f"Track {i}")
            calculate_msd(i)
        progress.complete_step()
    """
    
    def __init__(self, title: str, steps: List[ProgressStep]):
        """
        Initialize multi-step progress.
        
        Parameters
        ----------
        title : str
            Overall operation title
        steps : List[ProgressStep]
            List of steps with weights
        """
        self.title = title
        self.steps = steps
        self.total_weight = sum(s.weight for s in steps)
        self.current_step_index = -1
        self.current_substep = 0
        self.current_substep_total = 1
        
        # Create UI
        self.container = st.container()
        with self.container:
            st.subheader(f"⚙️ {title}")
            
            # Overall progress
            st.markdown("**Overall Progress:**")
            self.overall_progress = st.progress(0)
            
            # Current step
            st.markdown("**Current Step:**")
            self.step_progress = st.progress(0)
            self.step_text = st.empty()
            self.status_text = st.empty()
            
            # Step list
            with st.expander("📋 Steps", expanded=False):
                self.step_list = st.empty()
                self._update_step_list()
        
        self.start_time = time.time()
    
    def _update_step_list(self):
        """Update the step list display."""
        step_lines = []
        for i, step in enumerate(self.steps):
            if i < self.current_step_index:
                icon = "✅"
            elif i == self.current_step_index:
                icon = "▶️"
            else:
                icon = "⏸️"
            
            step_lines.append(f"{icon} **{step.name}**: {step.description}")
        
        self.step_list.markdown("\n".join(step_lines))
    
    def start_step(self, step_index: int, total_substeps: int = 1):
        """Start a specific step."""
        self.current_step_index = step_index
        self.current_substep = 0
        self.current_substep_total = total_substeps
        
        step = self.steps[step_index]
        self.step_text.markdown(f"**{step.name}**")
        self.status_text.markdown(f"*{step.description}*")
        
        self._update_step_list()
        self._update_overall_progress()
    
    def update_substep(self, substep: int, message: str = ""):
        """Update progress within current step."""
        self.current_substep = substep
        
        # Update step progress bar
        step_progress = substep / self.current_substep_total
        self.step_progress.progress(step_progress)
        
        if message:
            self.status_text.markdown(f"*{message}*")
        
        self._update_overall_progress()
    
    def complete_step(self):
        """Mark current step as complete."""
        self.step_progress.progress(1.0)
        self._update_overall_progress()
    
    def _update_overall_progress(self):
        """Calculate and update overall progress."""
        completed_weight = sum(
            s.weight for i, s in enumerate(self.steps)
            if i < self.current_step_index
        )
        
        if self.current_step_index >= 0:
            current_step = self.steps[self.current_step_index]
            step_fraction = self.current_substep / self.current_substep_total
            completed_weight += current_step.weight * step_fraction
        
        overall_progress = completed_weight / self.total_weight
        self.overall_progress.progress(overall_progress)
    
    def complete(self, message: str = "All steps complete!"):
        """Mark entire operation as complete."""
        self.overall_progress.progress(1.0)
        self.step_progress.progress(1.0)
        
        elapsed = time.time() - self.start_time
        elapsed_str = f"{elapsed:.1f}s"
        if elapsed >= 60:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            elapsed_str = f"{minutes}m {seconds:.1f}s"
        
        self.status_text.markdown(
            f"✅ **{message}** (completed in {elapsed_str})"
        )
        self._update_step_list()


def with_progress(func: Callable, 
                  title: str,
                  total_items: int,
                  message_template: str = "Processing item {current}/{total}") -> Any:
    """
    Decorator/wrapper to add automatic progress tracking to any function.
    
    Parameters
    ----------
    func : Callable
        Function to wrap. Must accept `progress_callback` parameter.
    title : str
        Progress bar title
    total_items : int
        Total number of items to process
    message_template : str
        Template for status messages. Use {current} and {total} placeholders.
    
    Returns
    -------
    Any
        Result from wrapped function
    
    Example:
        def process_tracks(tracks, progress_callback=None):
            for i, track in enumerate(tracks):
                if progress_callback:
                    progress_callback(i + 1, f"Track {track.id}")
                # Process track
            return results
        
        result = with_progress(
            process_tracks,
            "Processing Tracks",
            len(tracks)
        )(tracks)
    """
    progress = AnalysisProgress(title, total_items)
    
    def progress_callback(current: int, message: str = ""):
        """Callback for progress updates."""
        if not message:
            message = message_template.format(current=current, total=total_items)
        return progress.update(current, message)
    
    try:
        # Call function with progress callback
        result = func(progress_callback=progress_callback)
        progress.complete()
        return result
    except Exception as e:
        progress.status_text.markdown(f"❌ **Error**: {str(e)}")
        raise


# Simpler progress context manager
class SimpleProgress:
    """
    Simplified progress context manager for quick use.
    
    Example:
        with SimpleProgress("Loading data", 3) as progress:
            progress.step("Loading file...")
            data = load_file()
            
            progress.step("Parsing content...")
            parsed = parse(data)
            
            progress.step("Validating...")
            validate(parsed)
    """
    
    def __init__(self, title: str, total_steps: int):
        """Initialize simple progress."""
        self.title = title
        self.total_steps = total_steps
        self.current_step = 0
        self.progress_bar = None
        self.status_text = None
    
    def __enter__(self):
        """Enter context."""
        st.markdown(f"### {self.title}")
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        return self
    
    def step(self, message: str):
        """Complete one step."""
        self.current_step += 1
        progress = self.current_step / self.total_steps
        self.progress_bar.progress(progress)
        self.status_text.markdown(f"**{message}** ({self.current_step}/{self.total_steps})")
        
        # Small delay for visual feedback
        time.sleep(0.1)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        if exc_type is None:
            self.progress_bar.progress(1.0)
            self.status_text.markdown(f"✅ **Complete!** ({self.total_steps}/{self.total_steps})")
        else:
            self.status_text.markdown(f"❌ **Error**: {exc_val}")
        
        return False  # Don't suppress exceptions


if __name__ == "__main__":
    # Demo/test the progress utilities
    import streamlit as st
    import time
    
    st.set_page_config(page_title="Progress Demo", layout="wide")
    st.title("Progress Utilities Demo")
    
    # Demo 1: Simple progress
    if st.button("Demo 1: Simple Progress"):
        progress = AnalysisProgress("Simple Analysis", total_steps=50)
        
        for i in range(50):
            if not progress.update(i + 1, f"Processing item {i + 1}"):
                st.warning("Cancelled!")
                break
            time.sleep(0.1)
        else:
            progress.complete("All items processed!")
    
    # Demo 2: Multi-step progress
    if st.button("Demo 2: Multi-Step Progress"):
        steps = [
            ProgressStep("Load Data", weight=1.0, description="Loading tracks from file"),
            ProgressStep("Calculate MSD", weight=3.0, description="Computing mean squared displacement"),
            ProgressStep("Fit Diffusion", weight=2.0, description="Fitting diffusion model"),
            ProgressStep("Generate Report", weight=1.0, description="Creating visualizations"),
        ]
        
        progress = MultiStepProgress("Complete Analysis", steps)
        
        # Step 1
        progress.start_step(0)
        time.sleep(1)
        progress.complete_step()
        
        # Step 2
        progress.start_step(1, total_substeps=100)
        for i in range(100):
            progress.update_substep(i + 1, f"Track {i + 1}/100")
            time.sleep(0.02)
        progress.complete_step()
        
        # Step 3
        progress.start_step(2, total_substeps=50)
        for i in range(50):
            progress.update_substep(i + 1, f"Fitting {i + 1}/50")
            time.sleep(0.05)
        progress.complete_step()
        
        # Step 4
        progress.start_step(3)
        time.sleep(1)
        progress.complete_step()
        
        progress.complete()
    
    # Demo 3: Simple context manager
    if st.button("Demo 3: Context Manager"):
        with SimpleProgress("Quick Analysis", 4) as progress:
            progress.step("Loading...")
            time.sleep(0.5)
            
            progress.step("Processing...")
            time.sleep(0.5)
            
            progress.step("Analyzing...")
            time.sleep(0.5)
            
            progress.step("Finishing...")
            time.sleep(0.5)
