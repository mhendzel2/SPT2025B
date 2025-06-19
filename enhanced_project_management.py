"""
Enhanced Project Management Interface for SPT Analysis
Provides improved UI/UX with guided workflow and statistical test suggestions.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from enhanced_error_handling import suggest_statistical_test, validate_statistical_analysis
from project_management import ProjectManager, Project, Condition
from state_manager import get_state_manager

def create_guided_project_setup():
    """Create a guided, step-by-step project setup interface."""
    st.title("🧪 Project Management - Guided Setup")
    
    # Progress indicator
    if 'setup_step' not in st.session_state:
        st.session_state.setup_step = 1
    
    progress_steps = ["Create Project", "Define Conditions", "Upload Data", "Ready for Analysis"]
    current_step = st.session_state.setup_step
    
    # Progress bar
    progress = current_step / len(progress_steps)
    st.progress(progress)
    st.write(f"**Step {current_step}/{len(progress_steps)}: {progress_steps[current_step-1]}**")
    
    if current_step == 1:
        show_project_creation_step()
    elif current_step == 2:
        show_condition_definition_step()
    elif current_step == 3:
        show_data_upload_step()
    elif current_step == 4:
        show_analysis_ready_step()

def show_project_creation_step():
    """Step 1: Project Creation"""
    st.markdown("""
    ### Welcome to SPT Analysis Project Setup
    
    Create a new project to organize your single particle tracking experiments. 
    You'll be able to compare different experimental conditions and generate comprehensive reports.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        project_name = st.text_input(
            "Project Name",
            placeholder="e.g., 'Chromatin Dynamics Study' or 'Drug Treatment Effects'",
            help="Choose a descriptive name for your experiment"
        )
        
        project_description = st.text_area(
            "Project Description (Optional)",
            placeholder="Brief description of your experiment, cell type, conditions, etc.",
            help="This will help you remember the context of your experiment"
        )
    
    with col2:
        st.info("""
        **Project Organization**
        
        Your project will contain:
        - Multiple experimental conditions
        - Track data files for each condition
        - Analysis results and comparisons
        - Generated reports
        """)
    
    if st.button("Create Project", type="primary", disabled=not project_name):
        # Create project
        pm = ProjectManager()
        try:
            project = Project(
                name=project_name,
                description=project_description
            )
            
            # Save project
            import os
            projects_dir = "projects"
            os.makedirs(projects_dir, exist_ok=True)
            project_file_path = os.path.join(projects_dir, f"{project.id}.json")
            pm.save_project(project, project_file_path)
            
            # Store in session state
            st.session_state.current_project = project
            st.session_state.setup_step = 2
            st.success(f"✅ Project '{project_name}' created successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Failed to create project: {str(e)}")

def show_condition_definition_step():
    """Step 2: Define Experimental Conditions"""
    if 'current_project' not in st.session_state:
        st.error("No project found. Please start over.")
        if st.button("Start Over"):
            st.session_state.setup_step = 1
            st.rerun()
        return
    
    project = st.session_state.current_project
    
    st.markdown(f"""
    ### Define Experimental Conditions for '{project.name}'
    
    Set up the different experimental conditions you want to compare 
    (e.g., Control, Treatment A, Treatment B, etc.)
    """)
    
    # Show existing conditions
    if project.conditions:
        st.write("**Current Conditions:**")
        for i, condition in enumerate(project.conditions):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"• {condition.name}")
                if condition.description:
                    st.caption(condition.description)
            with col2:
                st.caption(f"{len(condition.files)} files")
            with col3:
                if st.button("🗑️", key=f"delete_condition_{i}", help="Delete condition"):
                    project.conditions.pop(i)
                    st.rerun()
    
    # Add new condition
    st.markdown("---")
    st.write("**Add New Condition:**")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        condition_name = st.text_input(
            "Condition Name",
            placeholder="e.g., 'Control', 'Treatment 100nM', 'Knockout'",
            key="new_condition_name"
        )
        
        condition_description = st.text_input(
            "Description (Optional)",
            placeholder="Brief description of this condition",
            key="new_condition_desc"
        )
    
    with col2:
        st.info("""
        **Condition Examples**
        
        - Control
        - Drug Treatment (specify concentration)
        - Genetic Knockout/Knockdown
        - Different cell lines
        - Time points
        """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Add Condition", disabled=not condition_name):
            condition = Condition(
                name=condition_name,
                description=condition_description
            )
            project.conditions.append(condition)
            st.success(f"Added condition: {condition_name}")
            st.rerun()
    
    with col2:
        if st.button("Continue to Data Upload", disabled=len(project.conditions) == 0):
            st.session_state.setup_step = 3
            st.rerun()
    
    with col3:
        if st.button("Back"):
            st.session_state.setup_step = 1
            st.rerun()
    
    if len(project.conditions) == 0:
        st.warning("Please add at least one experimental condition to continue.")

def show_data_upload_step():
    """Step 3: Upload Data Files"""
    if 'current_project' not in st.session_state:
        st.error("No project found. Please start over.")
        return
    
    project = st.session_state.current_project
    
    st.markdown(f"""
    ### Upload Data Files for '{project.name}'
    
    Upload tracking data files for each experimental condition.
    """)
    
    # Condition selection
    condition_names = [c.name for c in project.conditions]
    selected_condition = st.selectbox(
        "Select condition to upload data for:",
        condition_names,
        help="Choose which experimental condition these files belong to"
    )
    
    if selected_condition:
        condition_idx = condition_names.index(selected_condition)
        condition = project.conditions[condition_idx]
        
        # Show existing files for this condition
        if condition.files:
            st.write(f"**Current files for '{selected_condition}':**")
            for i, file_info in enumerate(condition.files):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"• {file_info.get('name', 'Unknown')}")
                    st.caption(f"Type: {file_info.get('type', 'Unknown')}")
                with col2:
                    st.caption(f"Size: {file_info.get('size', 0)/1024:.1f} KB")
                with col3:
                    if st.button("🗑️", key=f"delete_file_{i}", help="Remove file"):
                        condition.files.pop(i)
                        st.rerun()
        
        # File upload
        st.markdown("---")
        uploaded_files = st.file_uploader(
            f"Upload tracking data for '{selected_condition}'",
            type=['csv', 'xml', 'mvd2', 'xlsx'],
            accept_multiple_files=True,
            help="Supported formats: CSV (recommended), XML (Imaris), MVD2 (Imaris), Excel"
        )
        
        if uploaded_files:
            if st.button("Add Files to Condition"):
                for uploaded_file in uploaded_files:
                    file_info = {
                        'name': uploaded_file.name,
                        'type': uploaded_file.type,
                        'size': uploaded_file.size,
                        'data': uploaded_file.getvalue()
                    }
                    condition.files.append(file_info)
                
                st.success(f"Added {len(uploaded_files)} files to '{selected_condition}'")
                st.rerun()
    
    # Navigation
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Back to Conditions"):
            st.session_state.setup_step = 2
            st.rerun()
    
    with col2:
        # Check if at least one condition has files
        has_data = any(len(c.files) > 0 for c in project.conditions)
        if st.button("Complete Setup", disabled=not has_data):
            st.session_state.setup_step = 4
            st.rerun()
    
    with col3:
        if st.button("Save Project"):
            save_current_project()
    
    if not any(len(c.files) > 0 for c in project.conditions):
        st.warning("Please upload at least one data file to continue.")

def show_analysis_ready_step():
    """Step 4: Project Ready for Analysis"""
    if 'current_project' not in st.session_state:
        st.error("No project found. Please start over.")
        return
    
    project = st.session_state.current_project
    
    st.markdown(f"""
    ### 🎉 Project Setup Complete!
    
    Your project **'{project.name}'** is ready for analysis.
    """)
    
    # Project summary
    st.subheader("Project Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Conditions", len(project.conditions))
    
    with col2:
        total_files = sum(len(c.files) for c in project.conditions)
        st.metric("Total Files", total_files)
    
    with col3:
        total_size = sum(
            sum(f.get('size', 0) for f in c.files) 
            for c in project.conditions
        )
        st.metric("Total Size", f"{total_size/1024/1024:.1f} MB")
    
    # Condition details
    st.subheader("Condition Details")
    for condition in project.conditions:
        with st.expander(f"📊 {condition.name} ({len(condition.files)} files)"):
            if condition.description:
                st.write(f"**Description:** {condition.description}")
            
            if condition.files:
                st.write("**Files:**")
                files_df = pd.DataFrame([
                    {
                        'File Name': f.get('name', 'Unknown'),
                        'Type': f.get('type', 'Unknown'),
                        'Size (KB)': f"{f.get('size', 0)/1024:.1f}"
                    }
                    for f in condition.files
                ])
                st.dataframe(files_df, use_container_width=True)
    
    # Next steps
    st.subheader("Next Steps")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔬 Start Analysis", type="primary"):
            save_current_project()
            st.success("Project saved! Navigate to the Analysis Dashboard to begin.")
            # Reset setup state
            st.session_state.setup_step = 1
            if 'current_project' in st.session_state:
                del st.session_state.current_project
    
    with col2:
        if st.button("🔄 Create Another Project"):
            st.session_state.setup_step = 1
            if 'current_project' in st.session_state:
                del st.session_state.current_project
            st.rerun()
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        if st.button("💾 Save Project"):
            save_current_project()
        
        if st.button("📤 Export Project Configuration"):
            export_project_config(project)

def save_current_project():
    """Save the current project to disk."""
    if 'current_project' not in st.session_state:
        st.error("No current project to save.")
        return
    
    try:
        project = st.session_state.current_project
        pm = ProjectManager()
        
        import os
        projects_dir = "projects"
        os.makedirs(projects_dir, exist_ok=True)
        project_file_path = os.path.join(projects_dir, f"{project.id}.json")
        pm.save_project(project, project_file_path)
        
        st.success(f"✅ Project '{project.name}' saved successfully!")
        
    except Exception as e:
        st.error(f"Failed to save project: {str(e)}")

def export_project_config(project: Project):
    """Export project configuration as downloadable file."""
    try:
        import json
        
        # Create export data (without file contents for size)
        export_data = {
            'project_name': project.name,
            'project_description': project.description,
            'conditions': [
                {
                    'name': c.name,
                    'description': c.description,
                    'file_count': len(c.files),
                    'file_names': [f.get('name', 'Unknown') for f in c.files]
                }
                for c in project.conditions
            ],
            'created_date': project.created_date,
            'total_files': sum(len(c.files) for c in project.conditions)
        }
        
        config_json = json.dumps(export_data, indent=2, default=str)
        
        st.download_button(
            label="📥 Download Configuration",
            data=config_json,
            file_name=f"{project.name.replace(' ', '_')}_config.json",
            mime="application/json"
        )
        
    except Exception as e:
        st.error(f"Failed to export configuration: {str(e)}")

def create_enhanced_comparison_interface():
    """Enhanced comparative analysis interface with statistical test suggestions."""
    st.title("📊 Enhanced Comparative Analysis")
    
    # Load project
    pm = ProjectManager()
    projects = pm.list_projects()
    
    if not projects:
        st.warning("No projects found. Please create a project first using the Guided Setup.")
        return
    
    # Project selection
    project_names = [p['name'] for p in projects]
    selected_project_name = st.selectbox("Select Project", project_names)
    
    if selected_project_name:
        # Load project details
        selected_project = next(p for p in projects if p['name'] == selected_project_name)
        project_path = selected_project['path']
        
        try:
            project = pm.load_project(project_path)
            show_enhanced_comparison_analysis(project)
            
        except Exception as e:
            st.error(f"Failed to load project: {str(e)}")

def show_enhanced_comparison_analysis(project: Project):
    """Show enhanced comparison analysis with statistical suggestions."""
    st.subheader(f"Comparative Analysis: {project.name}")
    
    if len(project.conditions) < 2:
        st.warning("At least 2 conditions are required for comparative analysis.")
        return
    
    # Load and process data for all conditions
    condition_data = {}
    
    with st.spinner("Loading and processing data..."):
        for condition in project.conditions:
            if condition.files:
                try:
                    # Load first file for each condition (simplified)
                    file_data = condition.files[0].get('data')
                    if file_data:
                        import io
                        df = pd.read_csv(io.BytesIO(file_data))
                        
                        # Basic analysis
                        if all(col in df.columns for col in ['x', 'y', 'track_id']):
                            # Calculate basic metrics
                            track_lengths = df.groupby('track_id').size()
                            displacements = []
                            
                            for track_id, track_df in df.groupby('track_id'):
                                if len(track_df) > 1:
                                    track_df = track_df.sort_values('frame') if 'frame' in track_df.columns else track_df
                                    dx = track_df['x'].diff().dropna()
                                    dy = track_df['y'].diff().dropna()
                                    track_displacements = np.sqrt(dx**2 + dy**2)
                                    displacements.extend(track_displacements)
                            
                            condition_data[condition.name] = {
                                'track_lengths': track_lengths.values,
                                'displacements': displacements,
                                'n_tracks': len(track_lengths),
                                'total_points': len(df)
                            }
                        
                except Exception as e:
                    st.error(f"Error processing data for {condition.name}: {str(e)}")
    
    if not condition_data:
        st.error("No valid data found in any condition.")
        return
    
    # Analysis selection
    st.subheader("Analysis Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        analysis_metric = st.selectbox(
            "Metric to Compare",
            ["Track Lengths", "Step Displacements", "Number of Tracks"],
            help="Choose which metric to compare between conditions"
        )
    
    with col2:
        conditions_to_compare = st.multiselect(
            "Conditions to Compare",
            list(condition_data.keys()),
            default=list(condition_data.keys())[:2],
            help="Select which conditions to include in the comparison"
        )
    
    if len(conditions_to_compare) < 2:
        st.warning("Please select at least 2 conditions to compare.")
        return
    
    # Extract data for comparison
    comparison_data = {}
    
    for condition_name in conditions_to_compare:
        if analysis_metric == "Track Lengths":
            data = condition_data[condition_name]['track_lengths']
        elif analysis_metric == "Step Displacements":
            data = condition_data[condition_name]['displacements']
        else:  # Number of Tracks
            data = [condition_data[condition_name]['n_tracks']]
        
        comparison_data[condition_name] = data
    
    # Statistical Analysis with Suggestions
    st.subheader("Statistical Analysis")
    
    # Suggest appropriate test
    if len(conditions_to_compare) == 2:
        data1 = comparison_data[conditions_to_compare[0]]
        data2 = comparison_data[conditions_to_compare[1]]
        
        try:
            validate_statistical_analysis(data1, "comparative analysis")
            validate_statistical_analysis(data2, "comparative analysis")
            
            suggested_test = suggest_statistical_test(data1, data2)
            
            st.info(f"**Recommended Statistical Test:** {suggested_test}")
            
            # Perform multiple tests
            test_results = perform_statistical_tests(data1, data2, conditions_to_compare)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Test Results:**")
                for test_name, result in test_results.items():
                    st.write(f"• **{test_name}**: p = {result['p_value']:.4f}")
                    if result['p_value'] < 0.05:
                        st.write("  ✅ Significant difference detected")
                    else:
                        st.write("  ❌ No significant difference")
            
            with col2:
                st.write("**Effect Size:**")
                effect_size = calculate_effect_size(data1, data2)
                st.metric("Cohen's d", f"{effect_size:.3f}")
                
                if abs(effect_size) < 0.2:
                    st.caption("Small effect")
                elif abs(effect_size) < 0.8:
                    st.caption("Medium effect")
                else:
                    st.caption("Large effect")
        
        except Exception as e:
            st.error(f"Statistical analysis failed: {str(e)}")
    
    # Visualization
    st.subheader("Visualization")
    
    viz_type = st.selectbox(
        "Visualization Type",
        ["Box Plot", "Violin Plot", "Histogram", "Strip Plot"],
        help="Choose how to visualize the comparison"
    )
    
    fig = create_comparison_plot(comparison_data, analysis_metric, viz_type)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Summary
    create_comparison_summary(comparison_data, analysis_metric, conditions_to_compare)

def perform_statistical_tests(data1: List, data2: List, condition_names: List[str]) -> Dict[str, Any]:
    """Perform multiple statistical tests and return results."""
    results = {}
    
    try:
        # Mann-Whitney U Test (non-parametric)
        statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        results['Mann-Whitney U'] = {'statistic': statistic, 'p_value': p_value}
        
        # Welch's t-test (assumes unequal variances)
        statistic, p_value = stats.ttest_ind(data1, data2, equal_var=False)
        results["Welch's t-test"] = {'statistic': statistic, 'p_value': p_value}
        
        # Kolmogorov-Smirnov test (distribution comparison)
        statistic, p_value = stats.ks_2samp(data1, data2)
        results['Kolmogorov-Smirnov'] = {'statistic': statistic, 'p_value': p_value}
        
    except Exception as e:
        st.warning(f"Some statistical tests failed: {str(e)}")
    
    return results

def calculate_effect_size(data1: List, data2: List) -> float:
    """Calculate Cohen's d effect size."""
    try:
        mean1, mean2 = np.mean(data1), np.mean(data2)
        std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
        n1, n2 = len(data1), len(data2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
        
        # Cohen's d
        cohens_d = (mean1 - mean2) / pooled_std
        return cohens_d
        
    except:
        return 0.0

def create_comparison_plot(data: Dict[str, List], metric: str, plot_type: str):
    """Create comparison visualization."""
    try:
        # Prepare data for plotting
        plot_data = []
        for condition, values in data.items():
            for value in values:
                plot_data.append({'Condition': condition, 'Value': value})
        
        df = pd.DataFrame(plot_data)
        
        if plot_type == "Box Plot":
            fig = px.box(df, x='Condition', y='Value', title=f'{metric} Comparison')
        elif plot_type == "Violin Plot":
            fig = px.violin(df, x='Condition', y='Value', title=f'{metric} Comparison')
        elif plot_type == "Histogram":
            fig = px.histogram(df, x='Value', color='Condition', 
                             title=f'{metric} Distribution', opacity=0.7)
        else:  # Strip Plot
            fig = px.strip(df, x='Condition', y='Value', title=f'{metric} Comparison')
        
        fig.update_layout(
            xaxis_title="Experimental Condition",
            yaxis_title=metric,
            showlegend=True if plot_type == "Histogram" else False
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Failed to create plot: {str(e)}")
        return None

def create_comparison_summary(data: Dict[str, List], metric: str, conditions: List[str]):
    """Create summary statistics table."""
    st.subheader("Summary Statistics")
    
    summary_data = []
    for condition in conditions:
        values = data[condition]
        summary_data.append({
            'Condition': condition,
            'Count': len(values),
            'Mean': np.mean(values),
            'Median': np.median(values),
            'Std Dev': np.std(values),
            'Min': np.min(values),
            'Max': np.max(values)
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Format numerical columns
    numeric_cols = ['Mean', 'Median', 'Std Dev', 'Min', 'Max']
    for col in numeric_cols:
        summary_df[col] = summary_df[col].round(3)
    
    st.dataframe(summary_df, use_container_width=True)
    
    # Download option
    csv = summary_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Summary",
        data=csv,
        file_name=f"{metric.replace(' ', '_')}_summary.csv",
        mime="text/csv"
    )