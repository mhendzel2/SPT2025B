# New Features Documentation

## Overview

Three major enhancements have been added to SPT2025B to improve quality, monitoring, and automation:

1. **Performance Profiler Dashboard** - Real-time monitoring and profiling
2. **Data Quality Checker** - Automated validation and quality scoring
3. **CI/CD Pipeline** - Automated testing on every commit

---

## 1. Performance Profiler Dashboard

### Purpose
Real-time monitoring and profiling system for analyzing SPT analysis operations, tracking CPU/memory usage, identifying bottlenecks, and visualizing performance metrics.

### File
`performance_profiler.py`

### Key Features

#### Real-Time System Monitoring
- **CPU Usage**: Track CPU utilization during operations
- **Memory Usage**: Monitor RAM consumption with MB-level precision
- **Disk Usage**: Track available storage
- **Background Monitoring**: Continuous sampling at configurable intervals

#### Operation Profiling
- **Automatic Timing**: Decorator-based function profiling
- **Memory Tracking**: Per-operation memory delta measurement
- **Track/Frame Counting**: Automatic extraction of processing statistics
- **Success/Failure Tracking**: Error capture and logging

#### Bottleneck Detection
- **Automatic Identification**: Detects operations > 2x average time
- **Ranking**: Orders bottlenecks by severity
- **Historical Analysis**: 24-hour rolling window

#### Interactive Dashboard
- **Summary Metrics**: Total operations, success rate, averages
- **System Resource Charts**: CPU, memory, disk over time
- **Operation Timeline**: Scatter plot with size = memory usage
- **Distribution Analysis**: Box plots by operation type
- **Export Capabilities**: JSON and CSV export

### Usage

#### Basic Setup
```python
from performance_profiler import get_profiler

# Get global profiler instance
profiler = get_profiler()

# Start background monitoring (1 second intervals)
profiler.start_monitoring(interval=1.0)
```

#### Profile Functions
```python
# Use decorator to profile any function
@profiler.profile_operation("MSD Calculation")
def calculate_msd(tracks_df, **kwargs):
    # Your analysis code
    return results

# Or manually profile
with profiler.profile_operation("Custom Analysis"):
    # Your code here
    pass
```

#### Get Performance Metrics
```python
# Get summary statistics (last 24 hours)
summary = profiler.get_metrics_summary(hours=24)
print(f"Average duration: {summary['avg_duration']:.3f}s")
print(f"Success rate: {summary['success_rate']:.1f}%")
print(f"Bottlenecks: {len(summary['bottlenecks'])}")

# Get as DataFrame for analysis
metrics_df = profiler.get_metrics_dataframe(hours=24)
system_df = profiler.get_system_dataframe(hours=24)
```

#### Export Data
```python
# Export to JSON
profiler.export_metrics("performance_data.json")

# Clear history
profiler.clear_history()
```

#### Streamlit UI
```python
from performance_profiler import show_performance_dashboard

# Add to your Streamlit app
show_performance_dashboard()
```

### Dashboard Controls
- **🔄 Start Monitoring**: Begin background system monitoring
- **⏸️ Stop Monitoring**: Pause background monitoring
- **🗑️ Clear History**: Reset all performance data
- **Time Range**: Select 1, 6, 12, or 24 hour view

### Metrics Tracked

| Metric | Description | Unit |
|--------|-------------|------|
| `duration` | Operation execution time | seconds |
| `memory_used` | Memory delta during operation | MB |
| `cpu_percent` | Average CPU utilization | % |
| `track_count` | Number of tracks processed | count |
| `frame_count` | Number of frames processed | count |
| `success` | Operation success status | boolean |

### Performance Benefits
- **Identify Bottlenecks**: Automatically detect slow operations
- **Optimize Resource Usage**: Monitor memory and CPU consumption
- **Track Improvements**: Compare performance before/after optimizations
- **Production Monitoring**: Real-time health checks in production

---

## 2. Data Quality Checker

### Purpose
Automated validation system that performs comprehensive quality checks on particle tracking data, assigns quality scores, and provides actionable recommendations.

### File
`data_quality_checker.py`

### Key Features

#### Comprehensive Validation Checks

##### Critical Checks (60% weight)
1. **Required Columns**: Validates presence of `track_id`, `frame`, `x`, `y`
2. **Data Completeness**: Checks for null/missing values (target: >99%)

##### Warning Checks (30% weight)
3. **Track Continuity**: Detects frame gaps in tracks (target: >90% continuous)
4. **Spatial Outliers**: IQR-based outlier detection (target: <1% outliers)
5. **Track Length Distribution**: Analyzes short track prevalence (target: <30% short)
6. **Displacement Plausibility**: Checks for unrealistic velocities (target: <5% >1000 μm/s)
7. **Duplicate Detections**: Finds same position in same frame (target: <1%)

##### Info Checks (10% weight)
8. **Temporal Consistency**: Consecutive frame rate analysis
9. **Statistical Properties**: Skewness and kurtosis of coordinates
10. **Coordinate Ranges**: Validates reasonable spatial ranges

#### Quality Scoring System
- **Overall Score**: 0-100 weighted composite score
- **Per-Check Scoring**: Individual 0-100 scores
- **Color Coding**: Green (≥80), Orange (60-80), Red (<60)
- **Pass/Fail Status**: Binary result for each check

#### Track Statistics
- Total tracks and detections
- Track length statistics (mean, median, min, max)
- Frame and spatial coverage
- Short track rate

#### Actionable Recommendations
Auto-generated suggestions based on failed checks:
- Tracking parameter adjustments
- Detection threshold tuning
- Filtering strategies
- Data preprocessing steps

### Usage

#### Basic Quality Check
```python
from data_quality_checker import DataQualityChecker

# Initialize checker
checker = DataQualityChecker()

# Run all checks
report = checker.run_all_checks(
    tracks_df,
    pixel_size=0.1,
    frame_interval=0.1
)

# Access results
print(f"Overall Score: {report.overall_score:.1f}/100")
print(f"Passed: {report.passed_checks}/{report.total_checks}")
print(f"Recommendations: {len(report.recommendations)}")
```

#### Access Detailed Results
```python
# Iterate through checks
for check in report.checks:
    print(f"{check.check_name}: {'PASS' if check.passed else 'FAIL'}")
    print(f"  Score: {check.score:.1f}")
    print(f"  Message: {check.message}")
    print(f"  Details: {check.details}")

# Get recommendations
for rec in report.recommendations:
    print(rec)

# Get statistics
stats = report.track_statistics
print(f"Total tracks: {stats['total_tracks']}")
print(f"Mean track length: {stats['mean_track_length']:.1f} frames")
```

#### Streamlit UI
```python
from data_quality_checker import show_quality_checker_ui

# Add to your Streamlit app
show_quality_checker_ui()
```

### Quality Score Interpretation

| Score Range | Rating | Interpretation |
|-------------|--------|----------------|
| 90-100 | Excellent | High-quality data, ready for analysis |
| 80-89 | Good | Minor issues, generally acceptable |
| 60-79 | Fair | Some concerns, review recommendations |
| 40-59 | Poor | Significant issues, improvements needed |
| 0-39 | Critical | Major problems, data may be unusable |

### Check Details

#### Track Continuity
- **Method**: Identifies missing frames in track sequences
- **Target**: ≥90% of tracks without gaps
- **Impact**: Gaps can affect MSD calculation accuracy

#### Spatial Outliers
- **Method**: IQR method with 3×IQR threshold
- **Target**: <1% outlier rate
- **Impact**: Outliers may indicate detection errors

#### Displacement Plausibility
- **Method**: Calculates frame-to-frame velocities
- **Threshold**: >1000 μm/s flagged as unrealistic
- **Impact**: Unrealistic jumps suggest tracking failures

### Export Options
- **JSON Export**: Full report with all check details
- **Downloadable Reports**: Time-stamped quality reports
- **Session State Storage**: Report persistence in Streamlit

---

## 3. CI/CD Pipeline

### Purpose
Automated testing, quality analysis, and validation pipeline that runs on every commit to ensure code quality and prevent regressions.

### File
`.github/workflows/ci-cd.yml`

### Pipeline Jobs

#### 1. Test Job
**Purpose**: Run comprehensive tests across multiple Python versions and operating systems

**Matrix Testing**:
- Python versions: 3.9, 3.10, 3.11
- OS: Ubuntu, Windows, macOS
- Total: 9 combinations

**Steps**:
1. **Checkout code**: Get latest repository state
2. **Setup Python**: Install specified Python version with pip caching
3. **Install dependencies**: Install from `requirements.txt` plus testing tools
4. **Lint with flake8**: Check for syntax errors and code style
5. **Format check with black**: Verify code formatting consistency
6. **Run pytest**: Execute unit tests with coverage reporting
7. **Upload coverage**: Send results to Codecov
8. **Run standalone tests**: Execute `test_functionality.py` and `test_comprehensive.py`
9. **Test imports**: Verify critical packages import correctly

**Testing Tools**:
- `pytest`: Main testing framework
- `pytest-cov`: Coverage measurement
- `pytest-xdist`: Parallel test execution
- `flake8`: Linting
- `black`: Code formatting

#### 2. Quality Check Job
**Purpose**: Perform in-depth code quality and security analysis

**Steps**:
1. **Pylint**: Code quality and style analysis
   - Max line length: 127 characters
   - Disabled: C0103, C0114, C0115, C0116 (naming conventions)
2. **Mypy**: Static type checking
   - Ignore missing imports
   - No strict optional
3. **Bandit**: Security vulnerability scanning
   - Recursive scan
   - JSON and screen output
4. **Safety**: Dependency vulnerability check
   - Checks against known CVE database
5. **Radon**: Code complexity metrics
   - Cyclomatic complexity
   - Maintainability index

**Artifacts**:
- `bandit-report.json`: Security scan results
- `safety-report.json`: Dependency vulnerabilities
- Retention: 30 days

#### 3. Performance Test Job
**Purpose**: Run performance benchmarks to detect regressions

**Steps**:
1. Run `performance_benchmark.py`
2. Check for significant slowdowns
3. Report results for review

**Tools**:
- `pytest-benchmark`: Automated benchmarking
- `memory_profiler`: Memory usage profiling

#### 4. Documentation Job
**Purpose**: Verify documentation completeness

**Checks**:
- README.md exists and contains required sections
- CHANGELOG.md present
- LICENSE file present
- requirements.txt present

#### 5. Integration Test Job
**Purpose**: Test end-to-end workflows and component integration

**Tests**:
1. **Data Loading Pipeline**: Test file loading and parsing
2. **Analysis Pipeline**: Test MSD calculation on synthetic data
3. **State Management**: Test StateManager functionality
4. **Security Utilities**: Test SecureFileHandler validation
5. **Logging Configuration**: Test logger initialization

**Dependencies**: Runs after Test and Quality Check jobs pass

#### 6. Notify Job
**Purpose**: Summarize pipeline results

**Outputs**:
- Status of all jobs
- Overall pipeline pass/fail
- Runs even if previous jobs fail (`if: always()`)

### Trigger Events

| Event | Branches | Description |
|-------|----------|-------------|
| `push` | main, develop | Automatic on code push |
| `pull_request` | main, develop | Automatic on PR creation |
| `workflow_dispatch` | any | Manual trigger |

### Workflow Strategy
- **fail-fast: false**: Continue testing all matrix combinations even if one fails
- **continue-on-error**: Some jobs are informational and don't fail the build
- **Caching**: Pip packages cached for faster runs

### Using the Pipeline

#### Local Testing (Before Pushing)
```bash
# Run tests locally
python -m pytest tests/ -v

# Run linting
flake8 . --count --select=E9,F63,F7,F82 --show-source

# Check formatting
black --check .

# Run security check
bandit -r .

# Run standalone tests
python test_functionality.py
python test_comprehensive.py
```

#### Viewing Results
1. Go to repository on GitHub
2. Click "Actions" tab
3. Select workflow run
4. View job results and logs
5. Download artifacts (quality reports)

#### Badge Integration
Add to README.md:
```markdown
![CI/CD Pipeline](https://github.com/mhendzel2/SPT2025B/workflows/CI%2FCD%20Pipeline/badge.svg)
```

### Coverage Reporting
- Uploads to Codecov for tracking test coverage
- Shows coverage per file and function
- Tracks coverage trends over time

### Best Practices
1. **Always run tests locally first**: Use `pytest tests/` before pushing
2. **Check black formatting**: Run `black .` to auto-format
3. **Review linting warnings**: Address flake8 issues when possible
4. **Monitor coverage**: Aim for >80% code coverage
5. **Update tests**: Add tests for new features

---

## Integration with SPT2025B

### Adding to Main App

#### Option 1: Add New Pages
```python
# In app.py, add new pages
import performance_profiler as perf
import data_quality_checker as qc

# Create new page options
page = st.sidebar.radio(
    "Select Page",
    ["Analysis", "Performance Dashboard", "Quality Checker", ...]
)

if page == "Performance Dashboard":
    perf.show_performance_dashboard()
elif page == "Quality Checker":
    qc.show_quality_checker_ui()
```

#### Option 2: Add to Existing Workflows
```python
# Add profiling to existing analysis functions
from performance_profiler import get_profiler

profiler = get_profiler()

@profiler.profile_operation("MSD Analysis")
def analyze_msd():
    # Existing analysis code
    pass

# Add quality check before analysis
from data_quality_checker import DataQualityChecker

def validate_before_analysis():
    checker = DataQualityChecker()
    report = checker.run_all_checks(tracks_df, pixel_size, frame_interval)
    
    if report.overall_score < 60:
        st.warning(f"Data quality score is low: {report.overall_score:.1f}/100")
        st.write("Recommendations:")
        for rec in report.recommendations:
            st.write(rec)
        
        if not st.checkbox("Continue anyway?"):
            return False
    
    return True
```

### Recommended Workflow

1. **Data Loading** → Run Data Quality Checker
2. **Show Quality Report** → Review score and recommendations
3. **If Score < 60** → Display warnings and recommendations
4. **If Approved** → Continue to analysis
5. **During Analysis** → Profile operations with Performance Profiler
6. **After Analysis** → Review performance metrics
7. **On Every Commit** → CI/CD pipeline validates changes

---

## Configuration

### Performance Profiler Settings
```python
# In performance_profiler.py
profiler = PerformanceProfiler(
    max_history=1000  # Maximum metrics to store
)

# Start monitoring with custom interval
profiler.start_monitoring(interval=2.0)  # Every 2 seconds
```

### Quality Checker Thresholds
Edit thresholds in `data_quality_checker.py`:
```python
# Track continuity target
continuity_rate >= 90.0  # Line ~181

# Spatial outlier threshold
outlier_rate < 1.0  # Line ~230

# Displacement threshold
velocities > 1000  # Line ~301 (μm/s)

# Duplicate rate
duplicate_rate < 1.0  # Line ~322
```

### CI/CD Configuration
Edit `.github/workflows/ci-cd.yml`:
```yaml
# Change Python versions
python-version: ['3.9', '3.10', '3.11', '3.12']

# Change OS matrix
os: [ubuntu-latest, windows-latest, macos-latest]

# Adjust flake8 line length
--max-line-length=100  # Default: 127

# Change coverage threshold
--cov-fail-under=80  # Require 80% coverage
```

---

## Troubleshooting

### Performance Profiler

**Issue**: Monitoring not capturing data
- **Solution**: Ensure `profiler.start_monitoring()` is called
- **Check**: System permissions for psutil

**Issue**: Memory tracking shows zero
- **Solution**: Verify tracemalloc is started
- **Fix**: Call `tracemalloc.start()` before profiling

### Data Quality Checker

**Issue**: All checks fail with "Missing required columns"
- **Solution**: Verify DataFrame has `track_id`, `frame`, `x`, `y` columns
- **Check**: Use `get_track_data()` from `data_access_utils`

**Issue**: Score seems too low
- **Solution**: Review individual check scores and messages
- **Note**: Score is weighted: critical (60%), warning (30%), info (10%)

### CI/CD Pipeline

**Issue**: Tests fail on GitHub but pass locally
- **Solution**: Check Python version compatibility
- **Fix**: Test locally with multiple Python versions

**Issue**: Coverage upload fails
- **Solution**: Verify Codecov token in repository secrets
- **Note**: Coverage upload failure doesn't fail the build

**Issue**: Lint errors block merge
- **Solution**: Run `black .` and `flake8 .` locally before pushing
- **Auto-fix**: `black .` automatically formats code

---

## Performance Impact

### Performance Profiler
- **CPU Overhead**: ~1-2% during monitoring
- **Memory Overhead**: ~5-10 MB for history storage
- **Decorator Overhead**: <0.001s per function call

### Data Quality Checker
- **Typical Runtime**: 0.5-2 seconds for 10,000 detections
- **Memory Usage**: ~2x DataFrame size during checks
- **Scales**: Linearly with number of tracks

### CI/CD Pipeline
- **Typical Duration**: 5-10 minutes for full pipeline
- **Parallel Execution**: 9 matrix combinations run simultaneously
- **Cost**: Free for public repositories, uses GitHub Actions minutes

---

## Future Enhancements

### Performance Profiler
- [ ] Real-time alerts for memory/CPU thresholds
- [ ] Comparison mode for A/B testing
- [ ] GPU monitoring support
- [ ] Automatic bottleneck recommendations

### Data Quality Checker
- [ ] Machine learning-based anomaly detection
- [ ] Custom check definitions via config
- [ ] Batch quality assessment for multiple files
- [ ] Historical quality tracking

### CI/CD Pipeline
- [ ] Automatic dependency updates (Dependabot)
- [ ] Performance regression detection
- [ ] Automatic changelog generation
- [ ] Docker container builds

---

## Support

For questions or issues:
1. Check this documentation
2. Review code comments in source files
3. Check GitHub Issues
4. Run with `--help` flag for CLI tools

## License

These features are part of SPT2025B and follow the same license as the main project.
