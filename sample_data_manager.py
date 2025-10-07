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
        """Load information about available sample datasets by scanning directories."""
        if not self.sample_data_dir.exists():
            return
        
        # Recursively find all CSV files in sample_data and subdirectories
        csv_files = list(self.sample_data_dir.glob('**/*.csv'))
        
        for csv_file in csv_files:
            try:
                # Create relative path for cleaner display
                rel_path = csv_file.relative_to(self.sample_data_dir)
                
                # Use relative path as key for uniqueness
                file_key = str(rel_path).replace('\\', '/')
                
                # Extract dataset category from subdirectory
                parts = rel_path.parts
                if len(parts) > 1:
                    category = parts[0]  # e.g., 'U2OS_MS2', 'C2C12_40nm_SC35'
                    filename = parts[-1]
                else:
                    category = "General"
                    filename = parts[0]
                
                # Create a nice display name
                display_name = filename.replace('_spots.csv', '').replace('_', ' ').title()
                if category != "General":
                    display_name = f"{category} - {display_name}"
                
                # Load basic metadata without loading full dataset
                df = pd.read_csv(csv_file, nrows=5)  # Just peek at structure
                
                info = {
                    "name": display_name,
                    "description": f"Single-particle tracking data from {category}",
                    "category": category,
                    "filename": filename,
                    "file_path": str(csv_file),
                    "relative_path": file_key,
                    "file_size": csv_file.stat().st_size,
                    "available": True,
                    "tracks": None,  # Will be loaded dynamically
                    "particles": None,
                    "frames": None,
                    "columns": list(df.columns)
                }
                
                # Detect columns and structure
                if "TRACK_ID" in df.columns or "track_id" in df.columns:
                    info["has_tracks"] = True
                if "FRAME" in df.columns or "frame" in df.columns:
                    info["has_temporal"] = True
                if any("CH" in col.upper() for col in df.columns):
                    info["has_multichannel"] = True
                    info["channels"] = [col for col in df.columns if "CH" in col.upper()]
                
                self._available_datasets[file_key] = info
                
            except Exception as e:
                # Silently skip files that can't be read
                continue
    
    def get_available_datasets(self) -> Dict:
        """Get dictionary of available sample datasets."""
        return self._available_datasets
    
    def load_dataset(self, file_key: str) -> Optional[pd.DataFrame]:
        """Load a specific sample dataset.
        
        Parameters
        ----------
        file_key : str
            Relative path key of the dataset file to load
            
        Returns
        -------
        pd.DataFrame or None
            Loaded dataset or None if not found
        """
        if file_key not in self._available_datasets:
            return None
            
        try:
            file_path = self._available_datasets[file_key]["file_path"]
            df = pd.read_csv(file_path)
            
            # Update metadata with actual counts
            if "TRACK_ID" in df.columns:
                self._available_datasets[file_key]["tracks"] = df["TRACK_ID"].nunique()
            elif "track_id" in df.columns:
                self._available_datasets[file_key]["tracks"] = df["track_id"].nunique()
                
            self._available_datasets[file_key]["particles"] = len(df)
            
            if "FRAME" in df.columns:
                self._available_datasets[file_key]["frames"] = df["FRAME"].nunique()
            elif "frame" in df.columns:
                self._available_datasets[file_key]["frames"] = df["frame"].nunique()
                
            return df
            
        except Exception as e:
            st.error(f"Error loading dataset {file_key}: {str(e)}")
            return None
    
    def create_dataset_selector(self) -> Optional[str]:
        """Create Streamlit interface for dataset selection.
        
        Returns
        -------
        str or None
            Selected dataset relative path or None
        """
        if not self._available_datasets:
            st.warning("No sample datasets found in sample_data directory.")
            return None
            
        st.subheader("📊 Sample Dataset Selection")
        
        # Group datasets by category
        categories = {}
        for file_key, info in self._available_datasets.items():
            category = info.get('category', 'General')
            if category not in categories:
                categories[category] = []
            categories[category].append((file_key, info))
        
        # Create selection options with categories
        dataset_options = {}
        for category in sorted(categories.keys()):
            for file_key, info in sorted(categories[category], key=lambda x: x[1]['name']):
                display_name = f"{info['name']} ({len(info.get('columns', []))} cols)"
                dataset_options[display_name] = file_key
        
        # Add selection interface
        selected_display = st.selectbox(
            "Choose a sample dataset:",
            list(dataset_options.keys()),
            help="Select a pre-loaded sample dataset for testing and analysis"
        )
        
        if selected_display:
            selected_file_key = dataset_options[selected_display]
            info = self._available_datasets[selected_file_key]
            
            # Display dataset information
            with st.expander("Dataset Information", expanded=True):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Category:** {info['category']}")
                    st.write(f"**Description:** {info['description']}")
                    st.write(f"**File:** {info['filename']}")
                    st.write(f"**Path:** {info['relative_path']}")
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
            
            return selected_file_key
            
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