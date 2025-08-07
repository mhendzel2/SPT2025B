"""
Sample Data Manager for SPT Analysis Application
Provides functionality to load and select from multiple sample datasets for testing.
"""

import os
import pandas as pd
import streamlit as st
from typing import Dict, List, Optional
from pathlib import Path

class SampleDataManager:
    """Manages sample datasets for the SPT Analysis application."""
    
    def __init__(self, sample_data_dir: str = "sample_data"):
        """Initialize the sample data manager.
        
        Parameters
        ----------
        sample_data_dir : str
            Directory containing sample data files
        """
        self.sample_data_dir = Path(sample_data_dir)
        self._available_datasets = {}
        self._load_available_datasets()
    
    def _load_available_datasets(self):
        """Load information about available sample datasets."""
        if not self.sample_data_dir.exists():
            return
            
        # Define dataset descriptions
        dataset_info = {
            "Cell1_spots.csv": {
                "name": "Cell1 Spots",
                "description": "Single-particle tracking data from Cell1 with multi-channel intensity measurements",
                "tracks": None,  # Will be loaded dynamically
                "particles": None,
                "frames": None,
                "channels": ["CH1", "CH2", "CH3"]
            },
            "Cell2_spots.csv": {
                "name": "Cell2 Spots", 
                "description": "Single-particle tracking data from Cell2 experiment",
                "tracks": None,
                "particles": None,
                "frames": None,
                "channels": ["CH1", "CH2", "CH3"]
            },
            "Cropped_spots.csv": {
                "name": "Cropped Spots",
                "description": "Cropped region single-particle tracking data with focused analysis area",
                "tracks": None,
                "particles": None,
                "frames": None,
                "channels": []
            },
            "Frame8_spots.csv": {
                "name": "Frame8 Spots",
                "description": "Single frame analysis data for detailed particle characterization",
                "tracks": None,
                "particles": None,
                "frames": None,
                "channels": []
            },
            "MS2_spots_F1C1.csv": {
                "name": "MS2 F1C1 Spots",
                "description": "MS2 mRNA tracking data from field 1, cell 1 with temporal dynamics",
                "tracks": None,
                "particles": None,
                "frames": None,
                "channels": []
            }
        }
        
        # Check which files exist and load their metadata
        for filename, info in dataset_info.items():
            file_path = self.sample_data_dir / filename
            if file_path.exists():
                try:
                    # Load basic metadata without loading full dataset
                    df = pd.read_csv(file_path, nrows=5)  # Just peek at structure
                    info["available"] = True
                    info["file_path"] = str(file_path)
                    info["file_size"] = file_path.stat().st_size
                    
                    # Detect columns and structure
                    info["columns"] = list(df.columns)
                    if "TRACK_ID" in df.columns:
                        info["has_tracks"] = True
                    if "FRAME" in df.columns:
                        info["has_temporal"] = True
                    if any("CH" in col for col in df.columns):
                        info["has_multichannel"] = True
                        
                    self._available_datasets[filename] = info
                except Exception as e:
                    st.warning(f"Could not load metadata for {filename}: {str(e)}")
    
    def get_available_datasets(self) -> Dict:
        """Get dictionary of available sample datasets."""
        return self._available_datasets
    
    def load_dataset(self, filename: str) -> Optional[pd.DataFrame]:
        """Load a specific sample dataset.
        
        Parameters
        ----------
        filename : str
            Name of the dataset file to load
            
        Returns
        -------
        pd.DataFrame or None
            Loaded dataset or None if not found
        """
        if filename not in self._available_datasets:
            return None
            
        try:
            file_path = self._available_datasets[filename]["file_path"]
            df = pd.read_csv(file_path)
            
            # Update metadata with actual counts
            if "TRACK_ID" in df.columns:
                self._available_datasets[filename]["tracks"] = df["TRACK_ID"].nunique()
            self._available_datasets[filename]["particles"] = len(df)
            if "FRAME" in df.columns:
                self._available_datasets[filename]["frames"] = df["FRAME"].nunique()
                
            return df
            
        except Exception as e:
            st.error(f"Error loading dataset {filename}: {str(e)}")
            return None
    
    def create_dataset_selector(self) -> Optional[str]:
        """Create Streamlit interface for dataset selection.
        
        Returns
        -------
        str or None
            Selected dataset filename or None
        """
        if not self._available_datasets:
            st.warning("No sample datasets found in sample_data directory.")
            return None
            
        st.subheader("📊 Sample Dataset Selection")
        
        # Create selection options
        dataset_options = {}
        for filename, info in self._available_datasets.items():
            display_name = f"{info['name']} ({len(info.get('columns', []))} columns)"
            dataset_options[display_name] = filename
        
        # Add selection interface
        selected_display = st.selectbox(
            "Choose a sample dataset:",
            list(dataset_options.keys()),
            help="Select a pre-loaded sample dataset for testing and analysis"
        )
        
        if selected_display:
            selected_filename = dataset_options[selected_display]
            info = self._available_datasets[selected_filename]
            
            # Display dataset information
            with st.expander("Dataset Information", expanded=True):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Description:** {info['description']}")
                    st.write(f"**File:** {selected_filename}")
                    st.write(f"**Size:** {info['file_size'] / 1024:.1f} KB")
                    
                with col2:
                    if info.get('has_tracks'):
                        st.success("✓ Track data")
                    if info.get('has_temporal'):
                        st.success("✓ Temporal data")
                    if info.get('has_multichannel'):
                        st.success("✓ Multi-channel")
                
                # Show column information
                if info.get('columns'):
                    st.write("**Available Columns:**")
                    cols_display = ", ".join(info['columns'][:10])
                    if len(info['columns']) > 10:
                        cols_display += f" ... (+{len(info['columns']) - 10} more)"
                    st.write(cols_display)
            
            return selected_filename
            
        return None
    
    def create_sample_data_interface(self):
        """Create complete interface for sample data management."""
        st.markdown("### 🎯 Sample Data for Testing")
        
        # Dataset selector
        selected_dataset = self.create_dataset_selector()
        
        if selected_dataset:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if st.button("Load Selected Dataset", type="primary"):
                    df = self.load_dataset(selected_dataset)
                    if df is not None:
                        # Store in session state for use by the application
                        st.session_state.sample_data = df
                        st.session_state.sample_data_name = selected_dataset
                        st.success(f"Loaded {selected_dataset} successfully!")
                        
                        # Show preview
                        st.write("**Data Preview:**")
                        st.dataframe(df.head(), use_container_width=True)
                        
                        # Basic statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Particles", len(df))
                        with col2:
                            if "TRACK_ID" in df.columns:
                                st.metric("Unique Tracks", df["TRACK_ID"].nunique())
                        with col3:
                            if "FRAME" in df.columns:
                                st.metric("Time Points", df["FRAME"].nunique())
                        
                        return df
            
            with col2:
                st.write("**Quick Actions:**")
                if st.button("Clear Data"):
                    if 'sample_data' in st.session_state:
                        del st.session_state.sample_data
                    if 'sample_data_name' in st.session_state:
                        del st.session_state.sample_data_name
                    st.rerun()
        
        # Show currently loaded data
        if 'sample_data' in st.session_state:
            st.info(f"Currently loaded: {st.session_state.get('sample_data_name', 'Unknown dataset')}")
        
        return None

def initialize_sample_data_manager():
    """Initialize sample data manager in session state."""
    if 'sample_data_manager' not in st.session_state:
        st.session_state.sample_data_manager = SampleDataManager()
    return st.session_state.sample_data_manager

def get_sample_data_manager() -> SampleDataManager:
    """Get the sample data manager from session state."""
    return initialize_sample_data_manager()