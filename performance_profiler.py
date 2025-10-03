"""
Performance Profiler Dashboard

Real-time monitoring and profiling system for SPT analysis operations.
Tracks CPU, memory, analysis timing, bottlenecks, and generates interactive visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import psutil
import time
import tracemalloc
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from functools import wraps
from dataclasses import dataclass, field
from collections import deque
import threading
import json

from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetric:
    """Container for a single performance measurement."""
    timestamp: datetime
    operation: str
    duration: float  # seconds
    memory_used: float  # MB
    cpu_percent: float
    track_count: int = 0
    frame_count: int = 0
    success: bool = True
    error_message: str = ""


@dataclass
class SystemSnapshot:
    """System resource snapshot."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float


class PerformanceProfiler:
    """
    Centralized performance profiling system.
    
    Features:
    - Real-time CPU/memory monitoring
    - Operation timing and profiling
    - Bottleneck detection
    - Historical performance tracking
    - Interactive dashboard visualization
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize profiler.
        
        Parameters
        ----------
        max_history : int
            Maximum number of metrics to store in history
        """
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.system_history: deque = deque(maxlen=max_history)
        self.active_operations: Dict[str, Dict[str, Any]] = {}
        self._monitoring = False
        self._monitor_thread = None
        
        # Initialize tracemalloc for memory profiling
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        
        logger.info("PerformanceProfiler initialized")
    
    def start_monitoring(self, interval: float = 1.0):
        """
        Start background system monitoring.
        
        Parameters
        ----------
        interval : float
            Monitoring interval in seconds
        """
        if self._monitoring:
            logger.warning("Monitoring already active")
            return
        
        self._monitoring = True
        
        def monitor_loop():
            while self._monitoring:
                snapshot = self._capture_system_snapshot()
                self.system_history.append(snapshot)
                time.sleep(interval)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info(f"Started system monitoring (interval={interval}s)")
    
    def stop_monitoring(self):
        """Stop background system monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.info("Stopped system monitoring")
    
    def _capture_system_snapshot(self) -> SystemSnapshot:
        """Capture current system resource state."""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return SystemSnapshot(
            timestamp=datetime.now(),
            cpu_percent=psutil.cpu_percent(interval=0.1),
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 ** 2),
            memory_available_mb=memory.available / (1024 ** 2),
            disk_usage_percent=disk.percent
        )
    
    def profile_operation(self, operation_name: str):
        """
        Decorator to profile function execution.
        
        Usage:
            @profiler.profile_operation("MSD Calculation")
            def calculate_msd(tracks_df):
                # ... analysis code ...
                return results
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Start profiling
                start_time = time.time()
                start_memory = tracemalloc.get_traced_memory()[0] / (1024 ** 2)
                start_cpu = psutil.cpu_percent(interval=0.1)
                
                # Track active operation
                op_id = f"{operation_name}_{id(func)}"
                self.active_operations[op_id] = {
                    'name': operation_name,
                    'start_time': start_time,
                    'start_memory': start_memory
                }
                
                success = True
                error_msg = ""
                result = None
                
                try:
                    # Execute function
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    success = False
                    error_msg = str(e)
                    logger.error(f"Operation '{operation_name}' failed: {e}")
                    raise
                finally:
                    # End profiling
                    end_time = time.time()
                    end_memory = tracemalloc.get_traced_memory()[0] / (1024 ** 2)
                    end_cpu = psutil.cpu_percent(interval=0.1)
                    
                    duration = end_time - start_time
                    memory_delta = end_memory - start_memory
                    avg_cpu = (start_cpu + end_cpu) / 2
                    
                    # Extract track/frame counts if available
                    track_count = 0
                    frame_count = 0
                    if args and hasattr(args[0], '__len__'):
                        try:
                            if hasattr(args[0], 'track_id'):
                                track_count = args[0]['track_id'].nunique()
                                frame_count = len(args[0])
                        except:
                            pass
                    
                    # Record metric
                    metric = PerformanceMetric(
                        timestamp=datetime.now(),
                        operation=operation_name,
                        duration=duration,
                        memory_used=memory_delta,
                        cpu_percent=avg_cpu,
                        track_count=track_count,
                        frame_count=frame_count,
                        success=success,
                        error_message=error_msg
                    )
                    
                    self.metrics_history.append(metric)
                    
                    # Remove from active operations
                    if op_id in self.active_operations:
                        del self.active_operations[op_id]
                    
                    # Log performance
                    logger.info(
                        f"Performance [{operation_name}]: "
                        f"duration={duration:.3f}s, memory={memory_delta:.2f}MB, "
                        f"cpu={avg_cpu:.1f}%, success={success}"
                    )
            
            return wrapper
        return decorator
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get summary statistics for recent metrics.
        
        Parameters
        ----------
        hours : int
            Number of hours to include in summary
        
        Returns
        -------
        dict
            Summary statistics including totals, averages, bottlenecks
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {
                'total_operations': 0,
                'success_rate': 0.0,
                'avg_duration': 0.0,
                'avg_memory': 0.0,
                'avg_cpu': 0.0,
                'bottlenecks': []
            }
        
        df = pd.DataFrame([vars(m) for m in recent_metrics])
        
        # Calculate bottlenecks (operations taking > 2x average time)
        operation_stats = df.groupby('operation').agg({
            'duration': ['mean', 'max', 'count'],
            'memory_used': 'mean',
            'cpu_percent': 'mean'
        }).round(3)
        
        bottlenecks = []
        for op in operation_stats.index:
            mean_time = operation_stats.loc[op, ('duration', 'mean')]
            max_time = operation_stats.loc[op, ('duration', 'max')]
            if max_time > 2 * mean_time and mean_time > 0.1:  # Only significant operations
                bottlenecks.append({
                    'operation': op,
                    'avg_time': mean_time,
                    'max_time': max_time,
                    'slowdown_factor': max_time / mean_time if mean_time > 0 else 0
                })
        
        bottlenecks.sort(key=lambda x: x['slowdown_factor'], reverse=True)
        
        return {
            'total_operations': len(recent_metrics),
            'success_rate': (df['success'].sum() / len(df)) * 100,
            'avg_duration': df['duration'].mean(),
            'avg_memory': df['memory_used'].mean(),
            'avg_cpu': df['cpu_percent'].mean(),
            'total_tracks_processed': df['track_count'].sum(),
            'total_frames_processed': df['frame_count'].sum(),
            'bottlenecks': bottlenecks[:5]  # Top 5 bottlenecks
        }
    
    def get_metrics_dataframe(self, hours: int = 24) -> pd.DataFrame:
        """Get metrics as DataFrame for analysis."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return pd.DataFrame()
        
        return pd.DataFrame([vars(m) for m in recent_metrics])
    
    def get_system_dataframe(self, hours: int = 24) -> pd.DataFrame:
        """Get system snapshots as DataFrame."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_snapshots = [s for s in self.system_history if s.timestamp >= cutoff_time]
        
        if not recent_snapshots:
            return pd.DataFrame()
        
        return pd.DataFrame([vars(s) for s in recent_snapshots])
    
    def clear_history(self):
        """Clear all performance history."""
        self.metrics_history.clear()
        self.system_history.clear()
        logger.info("Performance history cleared")
    
    def export_metrics(self, filepath: str):
        """Export metrics history to JSON file."""
        data = {
            'exported_at': datetime.now().isoformat(),
            'metrics': [
                {
                    'timestamp': m.timestamp.isoformat(),
                    'operation': m.operation,
                    'duration': m.duration,
                    'memory_used': m.memory_used,
                    'cpu_percent': m.cpu_percent,
                    'track_count': m.track_count,
                    'frame_count': m.frame_count,
                    'success': m.success,
                    'error_message': m.error_message
                }
                for m in self.metrics_history
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported {len(self.metrics_history)} metrics to {filepath}")


# Global profiler instance
_profiler_instance = None


def get_profiler() -> PerformanceProfiler:
    """Get or create global profiler instance."""
    global _profiler_instance
    if _profiler_instance is None:
        _profiler_instance = PerformanceProfiler()
    return _profiler_instance


def show_performance_dashboard():
    """
    Display interactive performance profiler dashboard in Streamlit.
    
    Shows:
    - Real-time system metrics (CPU, memory, disk)
    - Operation performance history
    - Bottleneck analysis
    - Interactive charts and statistics
    """
    st.title("⚡ Performance Profiler Dashboard")
    st.markdown("Real-time monitoring and profiling of SPT analysis operations")
    
    profiler = get_profiler()
    
    # Control panel
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Start Monitoring"):
            profiler.start_monitoring(interval=1.0)
            st.success("Monitoring started")
    
    with col2:
        if st.button("⏸️ Stop Monitoring"):
            profiler.stop_monitoring()
            st.info("Monitoring stopped")
    
    with col3:
        if st.button("🗑️ Clear History"):
            profiler.clear_history()
            st.success("History cleared")
    
    with col4:
        time_range = st.selectbox("Time Range", [1, 6, 12, 24], index=3, key="perf_time_range")
    
    # Get data
    metrics_df = profiler.get_metrics_dataframe(hours=time_range)
    system_df = profiler.get_system_dataframe(hours=time_range)
    summary = profiler.get_metrics_summary(hours=time_range)
    
    # Summary statistics
    st.markdown("---")
    st.markdown("### 📊 Summary Statistics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Operations", summary['total_operations'])
    with col2:
        st.metric("Success Rate", f"{summary['success_rate']:.1f}%")
    with col3:
        st.metric("Avg Duration", f"{summary['avg_duration']:.3f}s")
    with col4:
        st.metric("Avg Memory", f"{summary['avg_memory']:.2f}MB")
    with col5:
        st.metric("Avg CPU", f"{summary['avg_cpu']:.1f}%")
    
    # System resource monitoring
    if not system_df.empty:
        st.markdown("---")
        st.markdown("### 💻 System Resources")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('CPU Usage (%)', 'Memory Usage (%)', 
                          'Memory Used (MB)', 'Disk Usage (%)'),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # CPU
        fig.add_trace(
            go.Scatter(x=system_df['timestamp'], y=system_df['cpu_percent'],
                      mode='lines', name='CPU %', line=dict(color='#FF6B6B')),
            row=1, col=1
        )
        
        # Memory %
        fig.add_trace(
            go.Scatter(x=system_df['timestamp'], y=system_df['memory_percent'],
                      mode='lines', name='Memory %', line=dict(color='#4ECDC4')),
            row=1, col=2
        )
        
        # Memory MB
        fig.add_trace(
            go.Scatter(x=system_df['timestamp'], y=system_df['memory_used_mb'],
                      mode='lines', name='Memory MB', line=dict(color='#95E1D3')),
            row=2, col=1
        )
        
        # Disk
        fig.add_trace(
            go.Scatter(x=system_df['timestamp'], y=system_df['disk_usage_percent'],
                      mode='lines', name='Disk %', line=dict(color='#F38181')),
            row=2, col=2
        )
        
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Operation performance
    if not metrics_df.empty:
        st.markdown("---")
        st.markdown("### 🎯 Operation Performance")
        
        tab1, tab2, tab3 = st.tabs(["Timeline", "By Operation", "Bottlenecks"])
        
        with tab1:
            # Timeline of operations
            fig = px.scatter(
                metrics_df,
                x='timestamp',
                y='duration',
                color='operation',
                size='memory_used',
                hover_data=['cpu_percent', 'track_count', 'success'],
                title='Operation Timeline (size = memory usage)'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Box plot by operation
            fig = go.Figure()
            
            for operation in metrics_df['operation'].unique():
                op_data = metrics_df[metrics_df['operation'] == operation]
                fig.add_trace(go.Box(
                    y=op_data['duration'],
                    name=operation,
                    boxmean='sd'
                ))
            
            fig.update_layout(
                title='Duration Distribution by Operation',
                yaxis_title='Duration (s)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics table
            stats = metrics_df.groupby('operation').agg({
                'duration': ['mean', 'std', 'min', 'max', 'count'],
                'memory_used': 'mean',
                'cpu_percent': 'mean'
            }).round(3)
            st.dataframe(stats, use_container_width=True)
        
        with tab3:
            # Bottleneck analysis
            if summary['bottlenecks']:
                st.markdown("**Detected Performance Bottlenecks:**")
                st.markdown("Operations with significant slowdowns (max > 2x average)")
                
                bottleneck_df = pd.DataFrame(summary['bottlenecks'])
                bottleneck_df['slowdown_factor'] = bottleneck_df['slowdown_factor'].round(2)
                bottleneck_df['avg_time'] = bottleneck_df['avg_time'].round(3)
                bottleneck_df['max_time'] = bottleneck_df['max_time'].round(3)
                
                st.dataframe(bottleneck_df, use_container_width=True)
                
                # Bottleneck chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=bottleneck_df['operation'],
                        y=bottleneck_df['avg_time'],
                        name='Average',
                        marker_color='lightblue'
                    ),
                    go.Bar(
                        x=bottleneck_df['operation'],
                        y=bottleneck_df['max_time'],
                        name='Maximum',
                        marker_color='red'
                    )
                ])
                fig.update_layout(
                    title='Bottleneck Comparison (Average vs Maximum Duration)',
                    yaxis_title='Duration (s)',
                    barmode='group',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No significant bottlenecks detected")
    
    # Export functionality
    st.markdown("---")
    st.markdown("### 💾 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Metrics (JSON)"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"performance_metrics_{timestamp}.json"
            profiler.export_metrics(filepath)
            st.success(f"Exported to {filepath}")
    
    with col2:
        if not metrics_df.empty:
            csv = metrics_df.to_csv(index=False)
            st.download_button(
                label="Download Metrics (CSV)",
                data=csv,
                file_name=f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Raw data view
    with st.expander("📋 View Raw Metrics Data"):
        if not metrics_df.empty:
            st.dataframe(metrics_df, use_container_width=True)
        else:
            st.info("No metrics data available")


if __name__ == "__main__":
    # Example usage
    profiler = get_profiler()
    profiler.start_monitoring()
    
    # Example profiled function
    @profiler.profile_operation("Test Operation")
    def test_function(n):
        time.sleep(0.5)
        data = np.random.randn(n, n)
        result = np.dot(data, data.T)
        return result
    
    # Run test
    test_function(100)
    test_function(200)
    
    # Get summary
    summary = profiler.get_metrics_summary()
    print("Performance Summary:", summary)
    
    profiler.stop_monitoring()
