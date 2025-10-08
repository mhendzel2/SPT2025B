"""
Data Quality Checker

Automated validation and quality assessment system for particle tracking data.
Performs comprehensive checks on track data quality, completeness, and reliability.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats

from logging_config import get_logger
from data_access_utils import get_track_data

logger = get_logger(__name__)


@dataclass
class QualityCheck:
    """Container for a single quality check result."""
    check_name: str
    category: str  # 'critical', 'warning', 'info'
    passed: bool
    score: float  # 0-100
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityReport:
    """Complete quality assessment report."""
    timestamp: datetime
    overall_score: float  # 0-100
    total_checks: int
    passed_checks: int
    failed_checks: int
    warnings: int
    checks: List[QualityCheck]
    track_statistics: Dict[str, Any]
    recommendations: List[str]


class DataQualityChecker:
    """
    Comprehensive data quality validation system.
    
    Performs automated checks on:
    - Data completeness and integrity
    - Track continuity and gaps
    - Spatial and temporal outliers
    - Statistical properties
    - Physical plausibility
    """
    
    def __init__(self):
        """Initialize quality checker."""
        self.checks: List[QualityCheck] = []
        logger.info("DataQualityChecker initialized")
    
    def run_all_checks(self, 
                       tracks_df: pd.DataFrame,
                       pixel_size: float = 1.0,
                       frame_interval: float = 1.0) -> QualityReport:
        """
        Run all quality checks on track data.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame
            Track data with columns 'track_id', 'frame', 'x', 'y'
        pixel_size : float
            Pixel size in micrometers
        frame_interval : float
            Frame interval in seconds
        
        Returns
        -------
        QualityReport
            Comprehensive quality assessment report
        """
        logger.info("Running comprehensive data quality checks")
        self.checks = []
        
        # Run all checks
        self._check_required_columns(tracks_df)
        self._check_data_completeness(tracks_df)
        self._check_track_continuity(tracks_df)
        self._check_spatial_outliers(tracks_df, pixel_size)
        self._check_temporal_consistency(tracks_df, frame_interval)
        self._check_track_length_distribution(tracks_df)
        self._check_displacement_plausibility(tracks_df, pixel_size, frame_interval)
        self._check_duplicate_detections(tracks_df)
        self._check_statistical_properties(tracks_df)
        self._check_coordinate_ranges(tracks_df)
        
        # Calculate statistics
        track_stats = self._calculate_track_statistics(tracks_df, pixel_size, frame_interval)
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        # Calculate overall score
        total_checks = len(self.checks)
        passed_checks = sum(1 for c in self.checks if c.passed)
        failed_checks = sum(1 for c in self.checks if not c.passed and c.category == 'critical')
        warnings = sum(1 for c in self.checks if not c.passed and c.category == 'warning')
        
        # Weighted scoring: critical = 30%, warning = 10%, info = 5%
        critical_score = sum(c.score for c in self.checks if c.category == 'critical') / max(1, sum(1 for c in self.checks if c.category == 'critical'))
        warning_score = sum(c.score for c in self.checks if c.category == 'warning') / max(1, sum(1 for c in self.checks if c.category == 'warning'))
        info_score = sum(c.score for c in self.checks if c.category == 'info') / max(1, sum(1 for c in self.checks if c.category == 'info'))
        
        overall_score = (critical_score * 0.6 + warning_score * 0.3 + info_score * 0.1)
        
        report = QualityReport(
            timestamp=datetime.now(),
            overall_score=overall_score,
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            warnings=warnings,
            checks=self.checks,
            track_statistics=track_stats,
            recommendations=recommendations
        )
        
        logger.info(f"Quality check complete: score={overall_score:.1f}, "
                   f"passed={passed_checks}/{total_checks}")
        
        return report
    
    def _check_required_columns(self, tracks_df: pd.DataFrame):
        """Check if required columns are present."""
        required = ['track_id', 'frame', 'x', 'y']
        missing = [col for col in required if col not in tracks_df.columns]
        
        if missing:
            self.checks.append(QualityCheck(
                check_name="Required Columns",
                category='critical',
                passed=False,
                score=0.0,
                message=f"Missing required columns: {', '.join(missing)}",
                details={'missing_columns': missing}
            ))
        else:
            self.checks.append(QualityCheck(
                check_name="Required Columns",
                category='critical',
                passed=True,
                score=100.0,
                message="All required columns present",
                details={'columns': list(tracks_df.columns)}
            ))
    
    def _check_data_completeness(self, tracks_df: pd.DataFrame):
        """Check for missing or null values."""
        total_cells = tracks_df.size
        null_count = tracks_df.isnull().sum().sum()
        completeness = ((total_cells - null_count) / total_cells) * 100
        
        passed = completeness >= 99.0
        category = 'critical' if completeness < 95.0 else 'warning'
        
        self.checks.append(QualityCheck(
            check_name="Data Completeness",
            category=category,
            passed=passed,
            score=completeness,
            message=f"Data completeness: {completeness:.2f}% ({null_count} null values)",
            details={'null_count': int(null_count), 'completeness': completeness}
        ))
    
    def _check_track_continuity(self, tracks_df: pd.DataFrame):
        """Check for gaps in track frame sequences."""
        gaps_detected = []
        max_gap_size = 0
        
        for track_id, track_data in tracks_df.groupby('track_id'):
            frames = sorted(track_data['frame'].unique())
            expected_frames = set(range(frames[0], frames[-1] + 1))
            actual_frames = set(frames)
            missing_frames = expected_frames - actual_frames
            
            if missing_frames:
                gap_size = len(missing_frames)
                max_gap_size = max(max_gap_size, gap_size)
                gaps_detected.append({
                    'track_id': track_id,
                    'gap_size': gap_size,
                    'missing_frames': len(missing_frames)
                })
        
        total_tracks = tracks_df['track_id'].nunique()
        tracks_with_gaps = len(gaps_detected)
        continuity_rate = ((total_tracks - tracks_with_gaps) / total_tracks) * 100
        
        passed = continuity_rate >= 90.0
        category = 'warning' if continuity_rate >= 80.0 else 'critical'
        
        self.checks.append(QualityCheck(
            check_name="Track Continuity",
            category=category,
            passed=passed,
            score=continuity_rate,
            message=f"{tracks_with_gaps}/{total_tracks} tracks have gaps (max gap: {max_gap_size} frames)",
            details={
                'tracks_with_gaps': tracks_with_gaps,
                'max_gap_size': max_gap_size,
                'continuity_rate': continuity_rate
            }
        ))
    
    def _check_spatial_outliers(self, tracks_df: pd.DataFrame, pixel_size: float):
        """Detect spatial outliers using IQR method."""
        x_um = tracks_df['x'] * pixel_size
        y_um = tracks_df['y'] * pixel_size
        
        # Calculate IQR for x and y
        x_q1, x_q3 = x_um.quantile([0.25, 0.75])
        y_q1, y_q3 = y_um.quantile([0.25, 0.75])
        x_iqr = x_q3 - x_q1
        y_iqr = y_q3 - y_q1
        
        # Identify outliers (values beyond 3*IQR)
        x_outliers = ((x_um < (x_q1 - 3 * x_iqr)) | (x_um > (x_q3 + 3 * x_iqr))).sum()
        y_outliers = ((y_um < (y_q1 - 3 * y_iqr)) | (y_um > (y_q3 + 3 * y_iqr))).sum()
        total_outliers = x_outliers + y_outliers
        
        outlier_rate = (total_outliers / (len(tracks_df) * 2)) * 100
        score = max(0, 100 - outlier_rate * 10)  # Penalize outliers
        
        passed = outlier_rate < 1.0
        category = 'warning'
        
        self.checks.append(QualityCheck(
            check_name="Spatial Outliers",
            category=category,
            passed=passed,
            score=score,
            message=f"Spatial outlier rate: {outlier_rate:.2f}% ({total_outliers} outliers)",
            details={
                'x_outliers': int(x_outliers),
                'y_outliers': int(y_outliers),
                'outlier_rate': outlier_rate
            }
        ))
    
    def _check_temporal_consistency(self, tracks_df: pd.DataFrame, frame_interval: float):
        """Check temporal consistency of frame numbers."""
        frame_gaps = tracks_df['frame'].diff().dropna()
        
        # Most gaps should be 1 (consecutive frames)
        gap_counts = frame_gaps.value_counts()
        consecutive_rate = (gap_counts.get(1, 0) / len(frame_gaps)) * 100 if len(frame_gaps) > 0 else 100
        
        passed = consecutive_rate >= 80.0
        category = 'info'
        
        self.checks.append(QualityCheck(
            check_name="Temporal Consistency",
            category=category,
            passed=passed,
            score=consecutive_rate,
            message=f"Consecutive frame rate: {consecutive_rate:.1f}%",
            details={'consecutive_rate': consecutive_rate}
        ))
    
    def _check_track_length_distribution(self, tracks_df: pd.DataFrame):
        """Analyze track length distribution."""
        track_lengths = tracks_df.groupby('track_id').size()
        
        mean_length = track_lengths.mean()
        median_length = track_lengths.median()
        min_length = track_lengths.min()
        max_length = track_lengths.max()
        
        # Check if we have reasonable track lengths
        short_tracks = (track_lengths < 5).sum()
        short_track_rate = (short_tracks / len(track_lengths)) * 100
        
        score = max(0, 100 - short_track_rate * 2)
        passed = short_track_rate < 30.0
        category = 'warning'
        
        self.checks.append(QualityCheck(
            check_name="Track Length Distribution",
            category=category,
            passed=passed,
            score=score,
            message=f"Mean length: {mean_length:.1f} frames, {short_track_rate:.1f}% short tracks (<5 frames)",
            details={
                'mean_length': float(mean_length),
                'median_length': float(median_length),
                'min_length': int(min_length),
                'max_length': int(max_length),
                'short_track_rate': short_track_rate
            }
        ))
    
    def _check_displacement_plausibility(self, tracks_df: pd.DataFrame, 
                                        pixel_size: float, frame_interval: float):
        """Check if displacements are physically plausible."""
        displacements = []
        
        for track_id, track_data in tracks_df.groupby('track_id'):
            track = track_data.sort_values('frame')
            dx = track['x'].diff() * pixel_size
            dy = track['y'].diff() * pixel_size
            dist = np.sqrt(dx**2 + dy**2)
            displacements.extend(dist.dropna().values)
        
        if not displacements:
            return
        
        displacements = np.array(displacements)
        
        # Calculate velocities (μm/s)
        velocities = displacements / frame_interval
        
        # Check for unrealistic velocities (>1000 μm/s for typical biological particles)
        unrealistic = (velocities > 1000).sum()
        unrealistic_rate = (unrealistic / len(velocities)) * 100
        
        score = max(0, 100 - unrealistic_rate * 20)
        passed = unrealistic_rate < 5.0
        category = 'warning'
        
        self.checks.append(QualityCheck(
            check_name="Displacement Plausibility",
            category=category,
            passed=passed,
            score=score,
            message=f"Unrealistic velocity rate: {unrealistic_rate:.2f}% (>1000 μm/s)",
            details={
                'max_velocity': float(velocities.max()),
                'mean_velocity': float(velocities.mean()),
                'unrealistic_count': int(unrealistic),
                'unrealistic_rate': unrealistic_rate
            }
        ))
    
    def _check_duplicate_detections(self, tracks_df: pd.DataFrame):
        """Check for duplicate detections (same position in same frame)."""
        duplicates = tracks_df.duplicated(subset=['frame', 'x', 'y'], keep=False).sum()
        duplicate_rate = (duplicates / len(tracks_df)) * 100
        
        score = max(0, 100 - duplicate_rate * 10)
        passed = duplicate_rate < 1.0
        category = 'warning'
        
        self.checks.append(QualityCheck(
            check_name="Duplicate Detections",
            category=category,
            passed=passed,
            score=score,
            message=f"Duplicate detection rate: {duplicate_rate:.2f}% ({duplicates} duplicates)",
            details={
                'duplicate_count': int(duplicates),
                'duplicate_rate': duplicate_rate
            }
        ))
    
    def _check_statistical_properties(self, tracks_df: pd.DataFrame):
        """Check statistical properties of coordinates."""
        # Check for normal-ish distribution of positions
        x_skew = abs(stats.skew(tracks_df['x']))
        y_skew = abs(stats.skew(tracks_df['y']))
        x_kurtosis = abs(stats.kurtosis(tracks_df['x']))
        y_kurtosis = abs(stats.kurtosis(tracks_df['y']))
        
        # High skewness or kurtosis might indicate issues
        max_skew = max(x_skew, y_skew)
        max_kurtosis = max(x_kurtosis, y_kurtosis)
        
        score = 100.0
        if max_skew > 2.0:
            score -= 20
        if max_kurtosis > 10.0:
            score -= 20
        
        passed = score >= 60.0
        category = 'info'
        
        self.checks.append(QualityCheck(
            check_name="Statistical Properties",
            category=category,
            passed=passed,
            score=score,
            message=f"Skewness: {max_skew:.2f}, Kurtosis: {max_kurtosis:.2f}",
            details={
                'x_skew': float(x_skew),
                'y_skew': float(y_skew),
                'x_kurtosis': float(x_kurtosis),
                'y_kurtosis': float(y_kurtosis)
            }
        ))
    
    def _check_coordinate_ranges(self, tracks_df: pd.DataFrame):
        """Check if coordinate ranges are reasonable."""
        x_range = tracks_df['x'].max() - tracks_df['x'].min()
        y_range = tracks_df['y'].max() - tracks_df['y'].min()
        
        # Check for suspicious ranges (too small or negative coordinates)
        min_x = tracks_df['x'].min()
        min_y = tracks_df['y'].min()
        
        issues = []
        if x_range < 10:
            issues.append("Very small X range")
        if y_range < 10:
            issues.append("Very small Y range")
        if min_x < 0:
            issues.append("Negative X coordinates")
        if min_y < 0:
            issues.append("Negative Y coordinates")
        
        score = 100.0 if not issues else 50.0
        passed = len(issues) == 0
        category = 'info'
        
        message = "Coordinate ranges OK" if not issues else f"Issues: {', '.join(issues)}"
        
        self.checks.append(QualityCheck(
            check_name="Coordinate Ranges",
            category=category,
            passed=passed,
            score=score,
            message=message,
            details={
                'x_range': float(x_range),
                'y_range': float(y_range),
                'min_x': float(min_x),
                'min_y': float(min_y),
                'issues': issues
            }
        ))
    
    def _calculate_track_statistics(self, tracks_df: pd.DataFrame,
                                    pixel_size: float, frame_interval: float) -> Dict[str, Any]:
        """Calculate comprehensive track statistics."""
        track_lengths = tracks_df.groupby('track_id').size()
        
        # Calculate frame range
        min_frame = int(tracks_df['frame'].min())
        max_frame = int(tracks_df['frame'].max())
        
        # Calculate spatial ranges
        x_min = tracks_df['x'].min()
        x_max = tracks_df['x'].max()
        y_min = tracks_df['y'].min()
        y_max = tracks_df['y'].max()
        
        # Calculate displacement and velocity statistics
        displacements = []
        velocities = []
        for track_id, track_data in tracks_df.groupby('track_id'):
            if len(track_data) > 1:
                track_data = track_data.sort_values('frame')
                dx = track_data['x'].diff()
                dy = track_data['y'].diff()
                # Calculate displacement in physical units
                step_displacements = np.sqrt(dx**2 + dy**2) * pixel_size
                displacements.extend(step_displacements.dropna().tolist())
                # Calculate velocity (displacement / time)
                step_velocities = step_displacements / frame_interval
                velocities.extend(step_velocities.dropna().tolist())
        
        mean_displacement = float(np.mean(displacements)) if displacements else 0.0
        mean_velocity = float(np.mean(velocities)) if velocities else 0.0
        
        return {
            # Keys expected by app.py
            'n_tracks': int(tracks_df['track_id'].nunique()),
            'n_points': len(tracks_df),
            'mean_track_length': float(track_lengths.mean()),
            'median_track_length': float(track_lengths.median()),
            'frame_range': [min_frame, max_frame],
            'x_range': [float(x_min * pixel_size), float(x_max * pixel_size)],
            'y_range': [float(y_min * pixel_size), float(y_max * pixel_size)],
            'mean_displacement': mean_displacement,
            'mean_velocity': mean_velocity,
            # Additional keys for backward compatibility
            'total_tracks': int(tracks_df['track_id'].nunique()),
            'total_detections': len(tracks_df),
            'min_track_length': int(track_lengths.min()),
            'max_track_length': int(track_lengths.max()),
            'total_frames': max_frame - min_frame + 1,
            'x_range_um': float((x_max - x_min) * pixel_size),
            'y_range_um': float((y_max - y_min) * pixel_size),
            'has_z': 'z' in tracks_df.columns
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on failed checks."""
        recommendations = []
        
        for check in self.checks:
            if not check.passed:
                if check.check_name == "Track Continuity":
                    recommendations.append(
                        "⚠️ Tracks have gaps. Consider using gap-closing tracking algorithms "
                        "or adjusting detection parameters."
                    )
                elif check.check_name == "Spatial Outliers":
                    recommendations.append(
                        "⚠️ Spatial outliers detected. Review detection thresholds and "
                        "consider filtering outliers before analysis."
                    )
                elif check.check_name == "Track Length Distribution":
                    recommendations.append(
                        "⚠️ Many short tracks detected. Consider increasing minimum track "
                        "length filter or improving tracking parameters."
                    )
                elif check.check_name == "Displacement Plausibility":
                    recommendations.append(
                        "⚠️ Unrealistic displacements detected. Verify pixel size and "
                        "frame interval settings, or check for tracking errors."
                    )
                elif check.check_name == "Duplicate Detections":
                    recommendations.append(
                        "⚠️ Duplicate detections found. Review detection algorithm and "
                        "consider duplicate removal filters."
                    )
        
        if not recommendations:
            recommendations.append("✅ Data quality is good! No major issues detected.")
        
        return recommendations


def show_quality_checker_ui():
    """
    Display data quality checker interface in Streamlit.
    """
    st.title("🔍 Data Quality Checker")
    st.markdown("Automated validation and quality assessment of particle tracking data")
    
    # Get track data
    tracks_df, has_data = get_track_data()
    
    if not has_data:
        st.warning("No track data loaded. Please load tracking data first.")
        return
    
    st.success(f"Loaded {len(tracks_df)} detections from {tracks_df['track_id'].nunique()} tracks")
    
    # Get parameters
    col1, col2 = st.columns(2)
    with col1:
        pixel_size = st.number_input("Pixel Size (μm)", value=0.1, min_value=0.001, step=0.01)
    with col2:
        frame_interval = st.number_input("Frame Interval (s)", value=0.1, min_value=0.001, step=0.01)
    
    # Run quality checks
    if st.button("🔍 Run Quality Checks", type="primary"):
        with st.spinner("Running comprehensive quality checks..."):
            checker = DataQualityChecker()
            report = checker.run_all_checks(tracks_df, pixel_size, frame_interval)
            
            # Store report in session state
            st.session_state['quality_report'] = report
    
    # Display report if available
    if 'quality_report' in st.session_state:
        report = st.session_state['quality_report']
        
        # Overall score
        st.markdown("---")
        st.markdown("### 📊 Overall Quality Score")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Score with color coding
        score_color = "green" if report.overall_score >= 80 else "orange" if report.overall_score >= 60 else "red"
        
        with col1:
            st.metric("Overall Score", f"{report.overall_score:.1f}/100")
        with col2:
            st.metric("Passed Checks", f"{report.passed_checks}/{report.total_checks}")
        with col3:
            st.metric("Failed (Critical)", report.failed_checks)
        with col4:
            st.metric("Warnings", report.warnings)
        with col5:
            timestamp_str = report.timestamp.strftime("%H:%M:%S")
            st.metric("Check Time", timestamp_str)
        
        # Score gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=report.overall_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Quality Score"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': score_color},
                'steps': [
                    {'range': [0, 60], 'color': "lightgray"},
                    {'range': [60, 80], 'color': "lightblue"},
                    {'range': [80, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed checks
        st.markdown("---")
        st.markdown("### 📋 Detailed Check Results")
        
        # Categorize checks
        critical_checks = [c for c in report.checks if c.category == 'critical']
        warning_checks = [c for c in report.checks if c.category == 'warning']
        info_checks = [c for c in report.checks if c.category == 'info']
        
        tab1, tab2, tab3 = st.tabs(["Critical", "Warnings", "Info"])
        
        with tab1:
            if critical_checks:
                for check in critical_checks:
                    icon = "✅" if check.passed else "❌"
                    with st.expander(f"{icon} {check.check_name} - Score: {check.score:.1f}"):
                        st.write(check.message)
                        if check.details:
                            st.json(check.details)
            else:
                st.info("No critical checks defined")
        
        with tab2:
            if warning_checks:
                for check in warning_checks:
                    icon = "✅" if check.passed else "⚠️"
                    with st.expander(f"{icon} {check.check_name} - Score: {check.score:.1f}"):
                        st.write(check.message)
                        if check.details:
                            st.json(check.details)
            else:
                st.info("No warnings")
        
        with tab3:
            if info_checks:
                for check in info_checks:
                    icon = "✅" if check.passed else "ℹ️"
                    with st.expander(f"{icon} {check.check_name} - Score: {check.score:.1f}"):
                        st.write(check.message)
                        if check.details:
                            st.json(check.details)
            else:
                st.info("No info checks")
        
        # Track statistics
        st.markdown("---")
        st.markdown("### 📈 Track Statistics")
        
        stats = report.track_statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Track Counts:**")
            st.write(f"- Total Tracks: {stats['total_tracks']}")
            st.write(f"- Total Detections: {stats['total_detections']}")
            st.write(f"- Total Frames: {stats['total_frames']}")
            st.write(f"- Has Z-coordinate: {'Yes' if stats['has_z'] else 'No'}")
        
        with col2:
            st.write("**Track Lengths:**")
            st.write(f"- Mean: {stats['mean_track_length']:.1f} frames")
            st.write(f"- Median: {stats['median_track_length']:.1f} frames")
            st.write(f"- Min: {stats['min_track_length']} frames")
            st.write(f"- Max: {stats['max_track_length']} frames")
        
        st.write("**Spatial Range:**")
        st.write(f"- X Range: {stats['x_range_um']:.2f} μm")
        st.write(f"- Y Range: {stats['y_range_um']:.2f} μm")
        
        # Recommendations
        st.markdown("---")
        st.markdown("### 💡 Recommendations")
        
        for rec in report.recommendations:
            st.markdown(rec)
        
        # Export report
        st.markdown("---")
        st.markdown("### 💾 Export Report")
        
        # Create JSON export
        export_data = {
            'timestamp': report.timestamp.isoformat(),
            'overall_score': report.overall_score,
            'total_checks': report.total_checks,
            'passed_checks': report.passed_checks,
            'failed_checks': report.failed_checks,
            'warnings': report.warnings,
            'track_statistics': report.track_statistics,
            'recommendations': report.recommendations,
            'checks': [
                {
                    'name': c.check_name,
                    'category': c.category,
                    'passed': c.passed,
                    'score': c.score,
                    'message': c.message,
                    'details': c.details
                }
                for c in report.checks
            ]
        }
        
        import json
        json_str = json.dumps(export_data, indent=2)
        
        st.download_button(
            label="Download Report (JSON)",
            data=json_str,
            file_name=f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


if __name__ == "__main__":
    # Example usage
    # Create sample data
    np.random.seed(42)
    n_tracks = 100
    frames_per_track = 50
    
    data = []
    for track_id in range(n_tracks):
        x_start = np.random.uniform(0, 512)
        y_start = np.random.uniform(0, 512)
        
        for frame in range(frames_per_track):
            x = x_start + np.cumsum(np.random.randn(1))[0] * 2
            y = y_start + np.cumsum(np.random.randn(1))[0] * 2
            
            data.append({
                'track_id': track_id,
                'frame': frame,
                'x': x,
                'y': y
            })
    
    tracks_df = pd.DataFrame(data)
    
    # Run quality check
    checker = DataQualityChecker()
    report = checker.run_all_checks(tracks_df, pixel_size=0.1, frame_interval=0.1)
    
    print(f"\nQuality Report:")
    print(f"Overall Score: {report.overall_score:.1f}/100")
    print(f"Passed: {report.passed_checks}/{report.total_checks}")
    print(f"\nRecommendations:")
    for rec in report.recommendations:
        print(f"  {rec}")
