"""
Anomaly Detection Visualization Module

Creates interactive overlays and visualizations for detected anomalies in particle tracking data.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns

class AnomalyVisualizer:
    """
    Creates comprehensive visualizations for anomaly detection results.
    """
    
    def __init__(self):
        self.color_map = {
            'velocity': '#FF6B6B',      # Red
            'confinement': '#4ECDC4',   # Teal
            'directional': '#45B7D1',   # Blue
            'ml_detected': '#FFA07A',   # Light orange
            'spatial_outlier': '#98D8C8', # Mint
            'normal': '#95A5A6'         # Gray
        }
        
    def create_anomaly_overlay(self, tracks_df: pd.DataFrame, anomaly_results: Dict[str, Any], 
                              width: int = 800, height: int = 600) -> go.Figure:
        """
        Create an interactive overlay visualization showing all detected anomalies.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame
            Track data
        anomaly_results : Dict[str, Any]
            Results from comprehensive anomaly detection
        width, height : int
            Figure dimensions
            
        Returns
        -------
        go.Figure
            Interactive plotly figure with anomaly overlays
        """
        fig = go.Figure()
        
        # Get anomaly categories
        anomaly_types = self._categorize_tracks(anomaly_results)
        
        # Plot normal tracks first (background)
        normal_tracks = []
        for track_id in tracks_df['track_id'].unique():
            if track_id not in anomaly_types:
                normal_tracks.append(track_id)
        
        if normal_tracks:
            normal_data = tracks_df[tracks_df['track_id'].isin(normal_tracks)]
            fig.add_trace(go.Scatter(
                x=normal_data['x'],
                y=normal_data['y'],
                mode='markers',
                marker=dict(size=3, color=self.color_map['normal'], opacity=0.6),
                name='Normal',
                showlegend=True,
                hovertemplate='Track: %{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
                text=normal_data['track_id']
            ))
        
        # Plot anomalous tracks by type
        for anomaly_type, color in self.color_map.items():
            if anomaly_type == 'normal':
                continue
                
            anomalous_tracks = [track_id for track_id, types in anomaly_types.items() 
                              if anomaly_type in types]
            
            if anomalous_tracks:
                anomaly_data = tracks_df[tracks_df['track_id'].isin(anomalous_tracks)]
                
                fig.add_trace(go.Scatter(
                    x=anomaly_data['x'],
                    y=anomaly_data['y'],
                    mode='markers',
                    marker=dict(size=6, color=color, opacity=0.8, 
                              line=dict(width=1, color='black')),
                    name=f'{anomaly_type.replace("_", " ").title()}',
                    showlegend=True,
                    hovertemplate=f'<b>{anomaly_type.replace("_", " ").title()}</b><br>' +
                                'Track: %{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
                    text=anomaly_data['track_id']
                ))
        
        # Add specific anomaly markers for velocity and directional anomalies
        self._add_specific_anomaly_markers(fig, tracks_df, anomaly_results)
        
        # Update layout
        fig.update_layout(
            title='AI-Powered Anomaly Detection Overlay',
            xaxis_title='X Position',
            yaxis_title='Y Position',
            width=width,
            height=height,
            hovermode='closest',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Set equal aspect ratio
        fig.update_yaxis(scaleanchor="x", scaleratio=1)
        
        return fig
    
    def _add_specific_anomaly_markers(self, fig: go.Figure, tracks_df: pd.DataFrame, 
                                     anomaly_results: Dict[str, Any]):
        """Add specific markers for frame-level anomalies."""
        
        # Add velocity anomaly markers
        if 'velocity_anomalies' in anomaly_results:
            for track_id, frames in anomaly_results['velocity_anomalies'].items():
                if not frames:  # Skip empty lists
                    continue
                    
                anomaly_points = tracks_df[
                    (tracks_df['track_id'] == track_id) & 
                    (tracks_df['frame'].isin(frames))
                ]
                
                if not anomaly_points.empty:
                    fig.add_trace(go.Scatter(
                        x=anomaly_points['x'],
                        y=anomaly_points['y'],
                        mode='markers',
                        marker=dict(size=10, color='red', symbol='x', 
                                  line=dict(width=2, color='darkred')),
                        name='Velocity Anomaly',
                        showlegend=False,
                        hovertemplate='<b>Velocity Anomaly</b><br>' +
                                    'Track: %{text}<br>Frame: %{customdata}<br>' +
                                    'X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
                        text=[track_id] * len(anomaly_points),
                        customdata=anomaly_points['frame']
                    ))
        
        # Add directional anomaly markers
        if 'directional_anomalies' in anomaly_results:
            for track_id, frames in anomaly_results['directional_anomalies'].items():
                if not frames:  # Skip empty lists
                    continue
                    
                anomaly_points = tracks_df[
                    (tracks_df['track_id'] == track_id) & 
                    (tracks_df['frame'].isin(frames))
                ]
                
                if not anomaly_points.empty:
                    fig.add_trace(go.Scatter(
                        x=anomaly_points['x'],
                        y=anomaly_points['y'],
                        mode='markers',
                        marker=dict(size=8, color='blue', symbol='triangle-up', 
                                  line=dict(width=2, color='darkblue')),
                        name='Direction Change',
                        showlegend=False,
                        hovertemplate='<b>Directional Anomaly</b><br>' +
                                    'Track: %{text}<br>Frame: %{customdata}<br>' +
                                    'X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
                        text=[track_id] * len(anomaly_points),
                        customdata=anomaly_points['frame']
                    ))
    
    def create_anomaly_timeline(self, tracks_df: pd.DataFrame, anomaly_results: Dict[str, Any]) -> go.Figure:
        """
        Create a timeline visualization showing when anomalies occur.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame
            Track data
        anomaly_results : Dict[str, Any]
            Anomaly detection results
            
        Returns
        -------
        go.Figure
            Timeline visualization
        """
        fig = go.Figure()
        
        # Collect all anomaly events
        anomaly_events = []
        
        # Velocity anomalies
        if 'velocity_anomalies' in anomaly_results:
            for track_id, frames in anomaly_results['velocity_anomalies'].items():
                for frame in frames:
                    anomaly_events.append({
                        'track_id': track_id,
                        'frame': frame,
                        'type': 'Velocity',
                        'color': self.color_map['velocity']
                    })
        
        # Directional anomalies
        if 'directional_anomalies' in anomaly_results:
            for track_id, frames in anomaly_results['directional_anomalies'].items():
                for frame in frames:
                    anomaly_events.append({
                        'track_id': track_id,
                        'frame': frame,
                        'type': 'Directional',
                        'color': self.color_map['directional']
                    })
        
        # Confinement violations (corrected key name)
        if 'confinement_violations' in anomaly_results:
            for track_id, frames in anomaly_results['confinement_violations'].items():
                for frame in frames:
                    anomaly_events.append({
                        'track_id': track_id,
                        'frame': frame,
                        'type': 'Confinement',
                        'color': self.color_map['confinement']
                    })
        
        if not anomaly_events:
            fig.add_annotation(
                text="No frame-specific anomalies detected",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
        else:
            events_df = pd.DataFrame(anomaly_events)
            
            # Create scatter plot
            for anomaly_type in events_df['type'].unique():
                type_data = events_df[events_df['type'] == anomaly_type]
                
                fig.add_trace(go.Scatter(
                    x=type_data['frame'],
                    y=type_data['track_id'],
                    mode='markers',
                    marker=dict(size=8, color=type_data['color'].iloc[0]),
                    name=anomaly_type,
                    hovertemplate=f'<b>{anomaly_type} Anomaly</b><br>' +
                                'Frame: %{x}<br>Track: %{y}<extra></extra>'
                ))
        
        fig.update_layout(
            title='Anomaly Timeline',
            xaxis_title='Frame',
            yaxis_title='Track ID',
            height=400,
            hovermode='closest'
        )
        
        return fig
    
    def create_anomaly_heatmap(self, tracks_df: pd.DataFrame, anomaly_results: Dict[str, Any]) -> go.Figure:
        """
        Create a spatial heatmap of anomaly density.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame
            Track data
        anomaly_results : Dict[str, Any]
            Anomaly detection results
            
        Returns
        -------
        go.Figure
            Heatmap visualization
        """
        # Get all anomalous points
        anomaly_types = self._categorize_tracks(anomaly_results)
        anomalous_track_ids = list(anomaly_types.keys())
        
        if not anomalous_track_ids:
            fig = go.Figure()
            fig.add_annotation(
                text="No anomalies detected for heatmap",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        anomaly_data = tracks_df[tracks_df['track_id'].isin(anomalous_track_ids)]
        
        # Create 2D histogram
        fig = go.Figure(data=go.Histogram2d(
            x=anomaly_data['x'],
            y=anomaly_data['y'],
            colorscale='Reds',
            nbinsx=30,
            nbinsy=30,
            hovertemplate='X: %{x}<br>Y: %{y}<br>Count: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Anomaly Density Heatmap',
            xaxis_title='X Position',
            yaxis_title='Y Position',
            height=500
        )
        
        return fig
    
    def create_anomaly_dashboard(self, tracks_df: pd.DataFrame, anomaly_results: Dict[str, Any]) -> None:
        """
        Create a comprehensive dashboard in Streamlit.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame
            Track data
        anomaly_results : Dict[str, Any]
            Anomaly detection results
        """
        st.subheader("🤖 AI-Powered Anomaly Detection Results")
        
        # Summary statistics
        anomaly_types = self._categorize_tracks(anomaly_results)
        total_tracks = len(tracks_df['track_id'].unique())
        anomalous_tracks = len(anomaly_types)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Tracks", total_tracks)
        with col2:
            st.metric("Anomalous Tracks", anomalous_tracks)
        with col3:
            st.metric("Anomaly Rate", f"{(anomalous_tracks/total_tracks*100):.1f}%" if total_tracks > 0 else "0%")
        with col4:
            st.metric("ML Confidence", f"{len(anomaly_results.get('ml_anomaly_scores', {}))}")
        
        # Main overlay visualization
        st.subheader("Anomaly Overlay")
        overlay_fig = self.create_anomaly_overlay(tracks_df, anomaly_results)
        st.plotly_chart(overlay_fig, use_container_width=True)
        
        # Additional visualizations in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Timeline", "Heatmap", "Statistics", "Details"])
        
        with tab1:
            timeline_fig = self.create_anomaly_timeline(tracks_df, anomaly_results)
            st.plotly_chart(timeline_fig, use_container_width=True)
        
        with tab2:
            heatmap_fig = self.create_anomaly_heatmap(tracks_df, anomaly_results)
            st.plotly_chart(heatmap_fig, use_container_width=True)
        
        with tab3:
            self._show_anomaly_statistics(anomaly_results, anomaly_types)
        
        with tab4:
            self._show_anomaly_details(tracks_df, anomaly_results, anomaly_types)
    
    def _show_anomaly_statistics(self, anomaly_results: Dict[str, Any], anomaly_types: Dict[int, List[str]]):
        """Show detailed anomaly statistics."""
        
        # Count anomalies by type
        type_counts = {}
        for track_id, types in anomaly_types.items():
            for anomaly_type in types:
                type_counts[anomaly_type] = type_counts.get(anomaly_type, 0) + 1
        
        if type_counts:
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(x=list(type_counts.keys()), y=list(type_counts.values()),
                      marker_color=[self.color_map.get(t, '#95A5A6') for t in type_counts.keys()])
            ])
            fig.update_layout(
                title='Anomaly Types Distribution',
                xaxis_title='Anomaly Type',
                yaxis_title='Count',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # ML anomaly scores distribution
        if 'ml_anomaly_scores' in anomaly_results and anomaly_results['ml_anomaly_scores']:
            scores = list(anomaly_results['ml_anomaly_scores'].values())
            fig = go.Figure(data=[go.Histogram(x=scores, nbinsx=20)])
            fig.update_layout(
                title='ML Anomaly Scores Distribution',
                xaxis_title='Anomaly Score',
                yaxis_title='Count',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"ML Detection Threshold: 0.0 (negative scores indicate anomalies)")
    
    def _show_anomaly_details(self, tracks_df: pd.DataFrame, anomaly_results: Dict[str, Any], 
                             anomaly_types: Dict[int, List[str]]):
        """Show detailed information about detected anomalies."""
        
        if not anomaly_types:
            st.info("No anomalies detected.")
            return
        
        # Create a detailed table
        details = []
        for track_id, types in anomaly_types.items():
            track_data = tracks_df[tracks_df['track_id'] == track_id]
            
            detail = {
                'Track ID': track_id,
                'Anomaly Types': ', '.join(types),
                'Track Length': len(track_data),
                'X Range': f"{track_data['x'].min():.2f} - {track_data['x'].max():.2f}",
                'Y Range': f"{track_data['y'].min():.2f} - {track_data['y'].max():.2f}",
            }
            
            # Add ML score if available
            if track_id in anomaly_results.get('ml_anomaly_scores', {}):
                detail['ML Score'] = f"{anomaly_results['ml_anomaly_scores'][track_id]:.3f}"
            
            details.append(detail)
        
        details_df = pd.DataFrame(details)
        st.dataframe(details_df, use_container_width=True)
        
        # Allow user to select a specific track for detailed view
        if anomaly_types:
            selected_track = st.selectbox("Select track for detailed view:", 
                                        options=list(anomaly_types.keys()),
                                        format_func=lambda x: f"Track {x}")
            
            if selected_track:
                self._show_track_details(tracks_df, anomaly_results, selected_track)
    
    def _show_track_details(self, tracks_df: pd.DataFrame, anomaly_results: Dict[str, Any], track_id: int):
        """Show detailed information for a specific track."""
        
        track_data = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
        
        if track_data.empty:
            st.warning(f"No data found for track {track_id}")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Track trajectory
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=track_data['x'], y=track_data['y'],
                mode='lines+markers',
                name=f'Track {track_id}',
                line=dict(width=2),
                marker=dict(size=6)
            ))
            
            # Mark anomalous frames
            for anomaly_type, frames_dict in anomaly_results.items():
                if isinstance(frames_dict, dict) and track_id in frames_dict:
                    frames_list = frames_dict[track_id]
                    if isinstance(frames_list, list) and frames_list:
                        anomaly_points = track_data[track_data['frame'].isin(frames_list)]
                        if not anomaly_points.empty:
                            fig.add_trace(go.Scatter(
                                x=anomaly_points['x'], y=anomaly_points['y'],
                                mode='markers',
                                name=f'{anomaly_type.replace("_", " ").title()}',
                                marker=dict(size=10, symbol='x')
                            ))
            
            fig.update_layout(title=f'Track {track_id} Trajectory', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Velocity profile
            if len(track_data) > 1:
                dx = np.diff(track_data['x'])
                dy = np.diff(track_data['y'])
                velocities = np.sqrt(dx**2 + dy**2)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=track_data['frame'].iloc[1:], y=velocities,
                    mode='lines+markers',
                    name='Velocity'
                ))
                
                fig.update_layout(
                    title=f'Track {track_id} Velocity Profile',
                    xaxis_title='Frame',
                    yaxis_title='Velocity',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Track too short for velocity analysis")
    
    def _categorize_tracks(self, anomaly_results: Dict[str, Any]) -> Dict[int, List[str]]:
        """Categorize tracks by anomaly types."""
        track_anomalies = {}
        
        # Get all track IDs with any anomaly
        all_track_ids = set()
        for category_results in anomaly_results.values():
            if isinstance(category_results, dict):
                all_track_ids.update(category_results.keys())
        
        # Categorize each track
        for track_id in all_track_ids:
            anomaly_types = []
            
            # Check each anomaly type
            if track_id in anomaly_results.get('velocity_anomalies', {}):
                anomaly_types.append('velocity')
            
            if track_id in anomaly_results.get('confinement_violations', {}):
                anomaly_types.append('confinement')
            
            if track_id in anomaly_results.get('directional_anomalies', {}):
                anomaly_types.append('directional')
            
            if (track_id in anomaly_results.get('ml_anomaly_scores', {}) and 
                anomaly_results['ml_anomaly_scores'][track_id] < 0):
                anomaly_types.append('ml_detected')
            
            if (track_id in anomaly_results.get('spatial_clustering', {}) and 
                anomaly_results['spatial_clustering'][track_id] == 'outlier'):
                anomaly_types.append('spatial_outlier')
            
            if anomaly_types:
                track_anomalies[track_id] = anomaly_types
        
        return track_anomalies