import streamlit as st
import os
import numpy as np
import pandas as pd
import project_management as pm
from enhanced_report_generator import EnhancedSPTReportGenerator
from datetime import datetime
import json
from ui_utils import generate_batch_html_report as _generate_batch_html_report
from utils import calculate_track_statistics

def show_project_management_page():
    st.title("Project Management: Group cells into experimental conditions")
    pmgr = pm.ProjectManager()
    if "pm_current" not in st.session_state:
        st.session_state.pm_current = None  # holds pm.Project
    # Project selector/creator
    with st.expander("Project Selection", expanded=True):
        existing = pmgr.list_projects()
        options = [f"{p['name']} ({p['id'][:8]})" for p in existing]
        sel = st.selectbox("Select a project", options + ["<New Project>"])
        if sel == "<New Project>":
            c1, c2 = st.columns([2,1])
            with c1:
                new_name = st.text_input("Project name", value="My Experiment")
            with c2:
                if st.button("Create Project", use_container_width=True):
                    proj = pmgr.create_project(new_name)
                    st.session_state.pm_current = proj
                    st.success("Project created.")
                    st.rerun()
        else:
            idx = options.index(sel)
            meta = existing[idx]
            st.session_state.pm_current = pmgr.get_project(meta['id'])

    proj = st.session_state.pm_current
    if proj is None:
        st.info("Create or select a project to manage conditions and files.")
    else:
        st.subheader(f"Project: {proj.name}")

        if st.button("Delete Project", key="pm_delete_project"):
            st.session_state.confirm_delete = True

        if st.session_state.get("confirm_delete"):
            st.warning(f"Are you sure you want to delete project '{proj.name}'? This cannot be undone.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Yes, delete it", type="primary"):
                    pmgr.delete_project(proj.id)
                    st.session_state.pm_current = None
                    st.session_state.confirm_delete = False
                    st.success("Project deleted.")
                    st.rerun()
            with c2:
                if st.button("Cancel"):
                    st.session_state.confirm_delete = False
                    st.rerun()

        # Add condition
        with st.expander("Add Condition", expanded=True):
            cname = st.text_input("Condition name", key="pm_new_cond_name")
            cdesc = st.text_input("Description", key="pm_new_cond_desc")
            if st.button("Add Condition", key="pm_add_condition_btn"):
                if cname and cname.strip():
                    pmgr.add_condition(proj, cname.strip(), cdesc.strip())
                    pmgr.save_project(proj, os.path.join(pmgr.projects_dir, f"{proj.id}.json"))
                    st.success("Condition added.")
                    st.rerun()
                else:
                    st.error("Please enter a condition name.")

        # List conditions with file upload per condition
        for cond in list(proj.conditions):
            with st.expander(f"Condition: {cond.name} ({len(cond.files)} files)", expanded=True):

                # Delete condition button
                if st.button("Delete Condition", key=f"delete_cond_{cond.id}"):
                    if "confirm_delete_condition" not in st.session_state:
                        st.session_state.confirm_delete_condition = {}
                    st.session_state.confirm_delete_condition[cond.id] = True

                if st.session_state.get("confirm_delete_condition", {}).get(cond.id):
                    st.warning(f"Are you sure you want to delete condition '{cond.name}'?")
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("Yes, delete", type="primary", key=f"confirm_delete_cond_{cond.id}"):
                            pmgr.remove_condition(proj, cond.id)
                            pmgr.save_project(proj, os.path.join(pmgr.projects_dir, f"{proj.id}.json"))
                            st.session_state.confirm_delete_condition[cond.id] = False
                            st.success("Condition deleted.")
                            st.rerun()
                    with c2:
                        if st.button("Cancel", key=f"cancel_delete_cond_{cond.id}"):
                            st.session_state.confirm_delete_condition[cond.id] = False
                            st.rerun()

                uploaded = st.file_uploader(
                    "Add cell files (CSV, Excel, XML)",
                    type=["csv", "xlsx", "xls", "xml"],
                    accept_multiple_files=True,
                    key=f"pm_up_{cond.id}",
                    help="Upload track data in CSV, Excel, or XML (TrackMate) format"
                )

                # Track which files have been processed to avoid duplicates
                upload_key = f"pm_upload_processed_{cond.id}"
                if upload_key not in st.session_state:
                    st.session_state[upload_key] = set()

                if uploaded:
                    # Check if these are new files (not already processed)
                    new_files = []
                    for uf in uploaded:
                        file_id = f"{uf.name}_{uf.size}"
                        if file_id not in st.session_state[upload_key]:
                            new_files.append((uf, file_id))

                    # Only process new files
                    if new_files:
                        for uf, file_id in new_files:
                            try:
                                file_extension = os.path.splitext(uf.name)[1].lower()

                                # Handle different file types
                                if file_extension == '.csv':
                                    import pandas as _pd
                                    df = _pd.read_csv(uf)
                                elif file_extension in ['.xlsx', '.xls']:
                                    # Use the existing load_tracks_file function for Excel
                                    from data_loader import load_tracks_file
                                    df = load_tracks_file(uf)
                                    if df is None or df.empty:
                                        st.warning(f"No track data found in {uf.name}")
                                        continue
                                elif file_extension == '.xml':
                                    # Use the existing load_tracks_file function for XML
                                    from data_loader import load_tracks_file
                                    df = load_tracks_file(uf)
                                    if df is None or df.empty:
                                        st.warning(f"No track data found in {uf.name}")
                                        continue
                                else:
                                    st.warning(f"Unsupported file type: {file_extension}")
                                    continue

                                pmgr.add_file_to_condition(proj, cond.id, uf.name, df)
                                st.session_state[upload_key].add(file_id)
                            except Exception as e:
                                st.warning(f"Failed to add {uf.name}: {e}")
                        pmgr.save_project(proj, os.path.join(pmgr.projects_dir, f"{proj.id}.json"))
                        st.success(f"{len(new_files)} file(s) added.")
                        st.rerun()

                # Show files and remove option
                if cond.files:
                    for f in list(cond.files):
                        fname = f.get('name') or f.get('file_name') or f.get('id')
                        cols = st.columns([6,2,2])
                        cols[0].write(fname)
                        if cols[1].button("Preview", key=f"pv_{cond.id}_{f.get('id')}"):
                            try:
                                import pandas as _pd, io as _io, os as _os
                                if f.get('data'):
                                    df = _pd.read_csv(_io.BytesIO(f['data']))
                                elif f.get('data_path') and _os.path.exists(f['data_path']):
                                    df = _pd.read_csv(f['data_path'])
                                else:
                                    df = None
                                if df is not None:
                                    st.dataframe(df.head())
                                else:
                                    st.info("No data available for preview.")
                            except Exception as e:
                                st.warning(f"Preview failed: {e}")
                        if cols[2].button("Remove", key=f"rm_{cond.id}_{f.get('id')}"):
                            pmgr.remove_file_from_project(proj, cond.id, f.get('id'))
                            pmgr.save_project(proj, os.path.join(pmgr.projects_dir, f"{proj.id}.json"))
                            st.rerun()

                    # Clear All Files button
                    st.write("")  # Spacing
                    col1, col2 = st.columns([3, 1])
                    with col2:
                        if st.button("🗑️ Clear All Files", key=f"clear_all_{cond.id}", type="secondary"):
                            # Remove all files from this condition
                            for f in list(cond.files):
                                pmgr.remove_file_from_project(proj, cond.id, f.get('id'))
                            # Clear the processed tracking for this condition
                            upload_key = f"pm_upload_processed_{cond.id}"
                            if upload_key in st.session_state:
                                del st.session_state[upload_key]
                            pmgr.save_project(proj, os.path.join(pmgr.projects_dir, f"{proj.id}.json"))
                            st.success(f"All files removed from '{cond.name}'")
                            st.rerun()

                # Subpopulation detection and pooling workflow
                st.write("---")
                st.write("**Analysis Workflow:**")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Step 1: Detect Subpopulations**")
                    if st.button("🔬 Detect Subpopulations", key=f"subpop_{cond.id}", type="primary"):
                        with st.spinner("Analyzing single-cell heterogeneity..."):
                            try:
                                # Strategy: Each file represents one cell/nucleus
                                # Analyze track-level variation within each file to characterize cells

                                st.info(f"Analyzing {len(cond.files)} cells (files) in condition '{cond.name}'")

                                # Collect per-cell (per-file) features
                                cell_features_list = []
                                all_tracks_with_cell_id = []

                                for file_idx, file_info in enumerate(cond.files):
                                    try:
                                        # Load this cell's data
                                        import pandas as pd_local, io as io_local

                                        # Try multiple ways to get the data
                                        df = None
                                        if file_info.get('data'):
                                            # Data stored as bytes
                                            df = pd_local.read_csv(io_local.BytesIO(file_info['data']))
                                        elif file_info.get('data_path') and os.path.exists(file_info['data_path']):
                                            # Data stored in file path
                                            df = pd_local.read_csv(file_info['data_path'])
                                        elif file_info.get('path') and os.path.exists(file_info['path']):
                                            # Legacy path field
                                            from data_loader import load_tracks_file
                                            df = load_tracks_file(file_info['path'])

                                        if df is None or df.empty:
                                            st.warning(f"Could not load data from {file_info.get('name', 'unknown file')}")
                                            continue

                                        # Assign cell_id based on file
                                        cell_id = f"cell_{file_idx}"
                                        df['cell_id'] = cell_id
                                        df['source_file'] = file_info['name']
                                        all_tracks_with_cell_id.append(df)

                                        # Calculate per-cell features from track variation
                                        n_tracks = df['track_id'].nunique()

                                        if n_tracks < 5:  # Need minimum tracks for meaningful statistics
                                            continue

                                        # Get track-level statistics
                                        track_stats = df.groupby('track_id').agg({
                                            'x': ['mean', 'std'],
                                            'y': ['mean', 'std'],
                                            'frame': ['min', 'max', 'count']
                                        }).reset_index()

                                        track_stats.columns = ['track_id', 'x_mean', 'x_std', 'y_mean', 'y_std',
                                                              'frame_min', 'frame_max', 'track_length']

                                        # Calculate displacements per track
                                        track_displacements = []
                                        for tid, track_df in df.groupby('track_id'):
                                            if len(track_df) < 2:
                                                continue
                                            dx = np.diff(track_df['x'].values)
                                            dy = np.diff(track_df['y'].values)
                                            disp = np.sqrt(dx**2 + dy**2)
                                            track_displacements.append({
                                                'track_id': tid,
                                                'mean_displacement': np.mean(disp),
                                                'total_displacement': np.sum(disp)
                                            })

                                        disp_df = pd.DataFrame(track_displacements)

                                        # Aggregate to cell-level features
                                        cell_features = {
                                            'cell_id': cell_id,
                                            'source_file': file_info['name'],
                                            'n_tracks': n_tracks,

                                            # Track length statistics (measures track stability/lifetime)
                                            'mean_track_length': track_stats['track_length'].mean(),
                                            'std_track_length': track_stats['track_length'].std(),
                                            'cv_track_length': track_stats['track_length'].std() / track_stats['track_length'].mean() if track_stats['track_length'].mean() > 0 else 0,

                                            # Displacement statistics (measures mobility)
                                            'mean_displacement_per_track': disp_df['mean_displacement'].mean() if not disp_df.empty else 0,
                                            'std_displacement_per_track': disp_df['mean_displacement'].std() if not disp_df.empty else 0,
                                            'cv_displacement': (disp_df['mean_displacement'].std() / disp_df['mean_displacement'].mean()) if not disp_df.empty and disp_df['mean_displacement'].mean() > 0 else 0,

                                            # Spatial characteristics
                                            'spatial_extent_x': df['x'].max() - df['x'].min(),
                                            'spatial_extent_y': df['y'].max() - df['y'].min(),

                                            # Temporal characteristics
                                            'total_frames': df['frame'].max() - df['frame'].min() + 1,

                                            # Heterogeneity measures (variation within cell)
                                            'track_length_heterogeneity': track_stats['track_length'].std() / track_stats['track_length'].mean() if track_stats['track_length'].mean() > 0 else 0,
                                            'displacement_heterogeneity': disp_df['mean_displacement'].std() / disp_df['mean_displacement'].mean() if not disp_df.empty and disp_df['mean_displacement'].mean() > 0 else 0
                                        }

                                        cell_features_list.append(cell_features)

                                    except Exception as e:
                                        st.warning(f"Could not analyze file {file_info['name']}: {str(e)}")
                                        continue

                                if len(cell_features_list) < 5:
                                    st.error(f"Insufficient cells for analysis. Found {len(cell_features_list)} cells, need at least 5.")
                                    st.info("Each file should represent one cell with multiple tracks.")
                                    continue

                                # Create cell-level dataframe
                                cell_df = pd.DataFrame(cell_features_list)

                                # Prepare features for clustering
                                feature_cols = [
                                    'mean_track_length', 'cv_track_length',
                                    'mean_displacement_per_track', 'cv_displacement',
                                    'track_length_heterogeneity', 'displacement_heterogeneity'
                                ]

                                available_features = [f for f in feature_cols if f in cell_df.columns]
                                X = cell_df[available_features].fillna(0)

                                # Standardize features
                                from sklearn.preprocessing import StandardScaler
                                scaler = StandardScaler()
                                X_scaled = scaler.fit_transform(X)

                                # Try different numbers of clusters
                                from sklearn.cluster import KMeans
                                from sklearn.metrics import silhouette_score

                                best_k = 2
                                best_score = -1

                                for k in range(2, min(5, len(cell_df))):
                                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                                    labels = kmeans.fit_predict(X_scaled)

                                    if len(np.unique(labels)) > 1:
                                        score = silhouette_score(X_scaled, labels)
                                        if score > best_score:
                                            best_score = score
                                            best_k = k

                                # Final clustering with optimal k
                                kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                                cell_df['subpopulation'] = kmeans.fit_predict(X_scaled)

                                # Check if subpopulations are meaningful (silhouette > 0.25)
                                subpops_detected = best_score > 0.25

                                # Create result structure
                                result = {
                                    'success': True,
                                    'n_cells_total': len(cell_df),
                                    'n_subpopulations': best_k if subpops_detected else 1,
                                    'subpopulations_detected': subpops_detected,
                                    'clustering_method': 'kmeans',
                                    'silhouette_score': best_score,
                                    'cell_level_data': cell_df,
                                    'features_used': available_features
                                }

                                # Characterize subpopulations
                                if subpops_detected:
                                    subpop_chars = {}
                                    for subpop_id in range(best_k):
                                        subpop_cells = cell_df[cell_df['subpopulation'] == subpop_id]
                                        subpop_chars[f'subpop_{subpop_id}'] = {
                                            'subpopulation_id': int(subpop_id),
                                            'n_cells': len(subpop_cells),
                                            'fraction_of_total': len(subpop_cells) / len(cell_df),
                                            'feature_means': {f: float(subpop_cells[f].mean()) for f in available_features}
                                        }
                                    result['subpopulation_characteristics'] = subpop_chars

                                # Concatenate all tracks with cell_id for later use
                                if all_tracks_with_cell_id:
                                    pooled = pd.concat(all_tracks_with_cell_id, ignore_index=True)
                                    # Map subpopulation labels
                                    cell_to_subpop = dict(zip(cell_df['cell_id'], cell_df['subpopulation']))
                                    pooled['subpopulation'] = pooled['cell_id'].map(cell_to_subpop)
                                    result['pooled_tracks'] = pooled

                                # Store results
                                if 'subpopulation_results' not in st.session_state:
                                    st.session_state.subpopulation_results = {}
                                st.session_state.subpopulation_results[cond.id] = result

                                if result['subpopulations_detected']:
                                    st.success(f"✓ Detected {result['n_subpopulations']} subpopulations in '{cond.name}'")
                                    st.info(f"📊 Analyzed {result['n_cells_total']} cells (files) with silhouette score: {result['silhouette_score']:.3f}")

                                    # Show subpopulation breakdown
                                    subpop_chars = result.get('subpopulation_characteristics', {})
                                    if subpop_chars:
                                        st.write("**Subpopulation Distribution:**")
                                        for subpop_name, chars in subpop_chars.items():
                                            st.write(f"- Subpop {chars['subpopulation_id']}: {chars['n_cells']} cells ({chars['fraction_of_total']:.1%})")
                                else:
                                    st.info(f"'{cond.name}' appears homogeneous (silhouette score: {result['silhouette_score']:.3f} < 0.25)")

                            except ImportError as e:
                                st.error(f"Required module not available: {str(e)}")
                                st.info("Install required packages: pip install scikit-learn scipy numpy")
                            except Exception as e:
                                st.error(f"Error in subpopulation analysis: {str(e)}")
                                import traceback
                                with st.expander("Error Details"):
                                    st.code(traceback.format_exc())

                with col2:
                    st.write("**Step 2: Pool by Subpopulation**")

                    # Check if subpopulation analysis has been run
                    has_subpop_results = (
                        'subpopulation_results' in st.session_state and
                        cond.id in st.session_state.subpopulation_results
                    )

                    if has_subpop_results:
                        result = st.session_state.subpopulation_results[cond.id]

                        if result.get('subpopulations_detected'):
                            # Show pooling options
                            pool_option = st.radio(
                                "Pooling Strategy:",
                                ["Pool all data together", "Pool by subpopulation"],
                                key=f"pool_option_{cond.id}"
                            )

                            if st.button("📊 Pool & Load", key=f"pool_{cond.id}"):
                                cell_df = result.get('cell_level_data')

                                if pool_option == "Pool by subpopulation":
                                    # Pool each subpopulation separately - use pre-loaded data with cell_ids
                                    st.info("Loading data with subpopulation assignments...")

                                    # Use the pooled tracks from subpopulation detection (already has cell_id and subpopulation labels)
                                    pooled = result.get('pooled_tracks')

                                    if pooled is not None and not pooled.empty and cell_df is not None:
                                        if 'cell_id' in pooled.columns and 'subpopulation' in pooled.columns:
                                            pooled['group'] = cond.name

                                            st.session_state.tracks_data = pooled
                                            try:
                                                st.session_state.track_statistics = calculate_track_statistics(pooled)
                                            except Exception:
                                                pass

                                            st.success(f"✓ Loaded {len(pooled)} tracks with subpopulation labels")

                                            # Show distribution
                                            subpop_dist = pooled['subpopulation'].value_counts()
                                            st.write("**Track distribution by subpopulation:**")
                                            for subpop_id, count in subpop_dist.items():
                                                st.write(f"- Subpopulation {int(subpop_id)}: {count} tracks")
                                        else:
                                            st.error("cell_id or subpopulation column not found in pooled data")
                                    else:
                                        st.error("Failed to load pooled tracks from subpopulation detection")
                                else:
                                    # Pool all together
                                    pooled_result = cond.pool_tracks()
                                    if isinstance(pooled_result, tuple):
                                        pooled, errors = pooled_result
                                    else:
                                        pooled = pooled_result
                                        errors = []

                                    if pooled is not None and not pooled.empty:
                                        st.session_state.tracks_data = pooled
                                        try:
                                            st.session_state.track_statistics = calculate_track_statistics(pooled)
                                        except Exception:
                                            pass
                                        st.success(f"✓ Pooled {len(pooled)} rows (ignoring subpopulations)")
                                        if errors:
                                            st.warning(f"Encountered {len(errors)} errors during pooling")
                        else:
                            # Homogeneous - just pool normally
                            st.info("Condition is homogeneous")
                            if st.button("📊 Pool Data", key=f"pool_{cond.id}"):
                                pooled_result = cond.pool_tracks()
                                if isinstance(pooled_result, tuple):
                                    pooled, errors = pooled_result
                                else:
                                    pooled = pooled_result
                                    errors = []

                                if pooled is not None and not pooled.empty:
                                    st.session_state.tracks_data = pooled
                                    try:
                                        st.session_state.track_statistics = calculate_track_statistics(pooled)
                                    except Exception:
                                        pass
                                    st.success(f"✓ Pooled {len(pooled)} rows")
                                    if errors:
                                        st.warning(f"Encountered {len(errors)} errors")
                    else:
                        st.info("👈 Run subpopulation detection first")
                        st.caption("Or click below to pool without subpopulation analysis:")
                        if st.button("⚡ Quick Pool (Skip Analysis)", key=f"quick_pool_{cond.id}"):
                            pooled_result = cond.pool_tracks()
                            if isinstance(pooled_result, tuple):
                                pooled, errors = pooled_result
                            else:
                                pooled = pooled_result
                                errors = []

                            if pooled is not None and not pooled.empty:
                                st.session_state.tracks_data = pooled
                                try:
                                    st.session_state.track_statistics = calculate_track_statistics(pooled)
                                except Exception:
                                    pass
                                st.success(f"✓ Pooled {len(pooled)} rows")
                                if errors:
                                    st.warning(f"Encountered {len(errors)} errors")

        # Batch Analysis Section
        st.divider()
        st.header("📊 Batch Analysis & Comparison")

        if proj.conditions and any(cond.files for cond in proj.conditions):
            st.info("Analyze and compare data across all conditions in this project")

            # Analysis options
            analysis_col1, analysis_col2 = st.columns(2)

            with analysis_col1:
                st.subheader("Select Conditions to Compare")
                conditions_to_analyze = []
                for cond in proj.conditions:
                    if cond.files:
                        if st.checkbox(f"{cond.name} ({len(cond.files)} files)",
                                     value=True,
                                     key=f"analyze_cond_{cond.id}"):
                            conditions_to_analyze.append(cond)

            with analysis_col2:
                st.subheader("Analysis Options")

                # Select analyses to run
                with st.expander("📋 Select Analyses", expanded=False):
                    st.write("**Choose analyses to run on pooled condition data:**")

                    # Always available even if imported
                    from enhanced_report_generator import EnhancedSPTReportGenerator
                    temp_gen = EnhancedSPTReportGenerator(pd.DataFrame(), 0.1, 0.1)

                    # Group analyses by category
                    analyses_by_category = {}
                    for key, analysis in temp_gen.available_analyses.items():
                        category = analysis.get('category', 'Other')
                        if category not in analyses_by_category:
                            analyses_by_category[category] = []
                        analyses_by_category[category].append((key, analysis['name']))

                    selected_analyses = []
                    for category, analyses in sorted(analyses_by_category.items()):
                        st.write(f"**{category}**")
                        for key, name in analyses:
                            if st.checkbox(name, key=f"batch_analysis_{key}", value=key in ['basic_statistics', 'diffusion_analysis']):
                                selected_analyses.append(key)

                # Quick analysis buttons
                if st.button("📊 Generate Individual Reports", type="primary"):
                    if len(conditions_to_analyze) < 1:
                        st.error("Select at least one condition to analyze")
                    else:
                        with st.spinner("Generating reports for each condition..."):
                            try:
                                # Pool data from each condition, breaking down by subpopulation if detected
                                condition_datasets = {}
                                pooling_errors = {}
                                subpop_info = {}

                                for cond in conditions_to_analyze:
                                    # Check if subpopulation results exist for this condition
                                    has_subpop = ('subpopulation_results' in st.session_state and
                                                 cond.id in st.session_state.subpopulation_results)

                                    if has_subpop:
                                        result = st.session_state.subpopulation_results[cond.id]

                                        if result.get('subpopulations_detected'):
                                            # Use pre-loaded pooled data with subpopulation labels
                                            pooled_df = result.get('pooled_tracks')

                                            if pooled_df is not None and not pooled_df.empty and 'subpopulation' in pooled_df.columns:
                                                # Split into separate datasets by subpopulation
                                                n_subpops = result['n_subpopulations']
                                                subpop_info[cond.name] = {'n_subpopulations': n_subpops}

                                                for subpop_id in range(n_subpops):
                                                    subpop_df = pooled_df[pooled_df['subpopulation'] == subpop_id].copy()
                                                    if not subpop_df.empty:
                                                        dataset_name = f"{cond.name} - Subpop {subpop_id}"
                                                        condition_datasets[dataset_name] = subpop_df
                                                continue

                                    # No subpopulations detected or no subpopulation analysis - pool normally
                                    pooled_result = cond.pool_tracks()
                                    if isinstance(pooled_result, tuple):
                                        pooled_df, errors = pooled_result
                                        if errors:
                                            pooling_errors[cond.name] = errors
                                    else:
                                        pooled_df = pooled_result

                                    if pooled_df is not None and not pooled_df.empty:
                                        condition_datasets[cond.name] = pooled_df

                                if not condition_datasets:
                                    st.error("No valid data in selected conditions")
                                else:
                                    # Show pooling summary
                                    if subpop_info:
                                        st.success(f"✅ Pooled data from {len(conditions_to_analyze)} conditions ({len(condition_datasets)} datasets including subpopulations)")
                                        with st.expander("ℹ️ Subpopulation Breakdown", expanded=False):
                                            for cond_name, info in subpop_info.items():
                                                st.write(f"**{cond_name}:** {info['n_subpopulations']} subpopulations detected")
                                    else:
                                        st.success(f"✅ Pooled data from {len(condition_datasets)} conditions")

                                    if pooling_errors:
                                        with st.expander("⚠️ Pooling Warnings", expanded=False):
                                            for cond_name, errors in pooling_errors.items():
                                                st.warning(f"**{cond_name}:** {len(errors)} files had errors")

                                    # Get units
                                    pixel_size = st.session_state.get('pixel_size', 0.1)
                                    frame_interval = st.session_state.get('frame_interval', 0.1)

                                    # Generate reports
                                    generator = EnhancedSPTReportGenerator(pd.DataFrame(), pixel_size, frame_interval)

                                    analyses_to_run = selected_analyses if selected_analyses else ['basic_statistics', 'diffusion_analysis']

                                    report_results = generator.generate_condition_reports(
                                        condition_datasets,
                                        analyses_to_run,
                                        pixel_size,
                                        frame_interval
                                    )

                                    # Store results in session state
                                    st.session_state['batch_report_results'] = report_results

                                    # Display results
                                    st.subheader("📊 Analysis Results")

                                    # Summary table
                                    summary_data = []
                                    for name, df in condition_datasets.items():
                                        n_tracks = df['track_id'].nunique() if 'track_id' in df.columns else 0
                                        n_frames = df['frame'].nunique() if 'frame' in df.columns else 0
                                        n_points = len(df)

                                        cond_results = report_results['conditions'].get(name, {})

                                        if cond_results.get('success', False):
                                            status = "✅ Success"
                                            error_msg = ""
                                        else:
                                            status = "❌ Failed"
                                            error_msg = cond_results.get('error', 'Unknown error')

                                        summary_row = {
                                            'Condition': name,
                                            'Status': status,
                                            'Tracks': n_tracks,
                                            'Frames': n_frames,
                                            'Data Points': n_points
                                        }

                                        if error_msg:
                                            summary_row['Error'] = error_msg

                                        summary_data.append(summary_row)

                                    summary_df = pd.DataFrame(summary_data)
                                    st.dataframe(summary_df, use_container_width=True)

                                    # Show individual condition results
                                    for cond_name, cond_result in report_results['conditions'].items():
                                        with st.expander(f"📈 {cond_name} - Detailed Results", expanded=False):
                                            if cond_result.get('success', False):
                                                st.write(f"**Analyses completed:** {len(cond_result.get('analysis_results', {}))}")
                                                st.write(f"**Figures generated:** {len(cond_result.get('figures', {}))}")

                                                # Show individual analysis statuses
                                                analysis_results = cond_result.get('analysis_results', {})
                                                if analysis_results:
                                                    st.write("**Analysis Status:**")
                                                    for analysis_key, analysis_result in analysis_results.items():
                                                        if isinstance(analysis_result, dict):
                                                            if analysis_result.get('success', True) and 'error' not in analysis_result:
                                                                st.write(f"- ✅ {analysis_key}")
                                                            else:
                                                                st.write(f"- ❌ {analysis_key}: {analysis_result.get('error', 'Unknown error')}")
                                                        else:
                                                            st.write(f"- ✅ {analysis_key}")

                                                # Show figures - handle both Plotly and Matplotlib
                                                for analysis_key, fig in cond_result.get('figures', {}).items():
                                                    if fig:
                                                        try:
                                                            # Check if it's a matplotlib figure
                                                            import matplotlib.figure
                                                            if isinstance(fig, matplotlib.figure.Figure):
                                                                st.pyplot(fig, use_container_width=True, key=f"fig_{cond_name}_{analysis_key}")
                                                            else:
                                                                # Assume it's a Plotly figure
                                                                st.plotly_chart(fig, use_container_width=True, key=f"fig_{cond_name}_{analysis_key}")
                                                        except Exception as e:
                                                            st.warning(f"Could not display figure for {analysis_key}: {e}")
                                            else:
                                                st.error(f"Analysis failed: {cond_result.get('error', 'Unknown error')}")

                                    # Show comparison results if available
                                    if report_results.get('comparisons') and len(condition_datasets) >= 2:
                                        st.divider()
                                        st.subheader("🔬 Statistical Comparisons")

                                        comparisons = report_results['comparisons']

                                        if comparisons.get('success', False):
                                            # Show metrics summary
                                            if 'metrics' in comparisons:
                                                st.write("**Summary Metrics:**")
                                                metrics_df = pd.DataFrame(comparisons['metrics']).T
                                                st.dataframe(metrics_df, use_container_width=True)

                                            # Show statistical tests
                                            if 'statistical_tests' in comparisons and comparisons['statistical_tests']:
                                                st.write("**Pairwise Statistical Tests:**")
                                                for comparison, tests in comparisons['statistical_tests'].items():
                                                    with st.expander(f"📊 {comparison}", expanded=False):
                                                        for metric, test_results in tests.items():
                                                            st.write(f"**{metric.replace('_', ' ').title()}:**")
                                                            if 't_test' in test_results:
                                                                p_val = test_results['t_test']['p_value']
                                                                significant = test_results.get('significant', False)
                                                                sig_text = "✅ Significant" if significant else "❌ Not significant"
                                                                st.write(f"- t-test p-value: {p_val:.4f} ({sig_text})")
                                                            if 'mann_whitney' in test_results:
                                                                p_val = test_results['mann_whitney']['p_value']
                                                                st.write(f"- Mann-Whitney p-value: {p_val:.4f}")

                                            # Show comparison figures
                                            if 'figures' in comparisons and comparisons['figures'].get('comparison_boxplots'):
                                                fig = comparisons['figures']['comparison_boxplots']
                                                try:
                                                    # Check if it's a matplotlib figure
                                                    import matplotlib.figure
                                                    if isinstance(fig, matplotlib.figure.Figure):
                                                        st.pyplot(fig, use_container_width=True, key="comparison_boxplots")
                                                    else:
                                                        # Assume it's a Plotly figure
                                                        st.plotly_chart(fig, use_container_width=True, key="comparison_boxplots")
                                                except Exception as e:
                                                    st.warning(f"Could not display comparison figure: {e}")
                                        else:
                                            st.warning(f"Comparison analysis failed: {comparisons.get('error', 'Unknown error')}")

                                    # Download options
                                    st.divider()
                                    st.subheader("💾 Export Results")
                                    col1, col2, col3, col4 = st.columns(4)

                                    with col1:
                                        # JSON export
                                        report_json = json.dumps(report_results, indent=2, default=str)
                                        st.download_button(
                                            "📄 JSON Report",
                                            data=report_json,
                                            file_name=f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                            mime="application/json"
                                        )

                                    with col2:
                                        # CSV export of metrics
                                        if report_results.get('comparisons', {}).get('metrics'):
                                            metrics_csv = pd.DataFrame(report_results['comparisons']['metrics']).T.to_csv()
                                            st.download_button(
                                                "📊 Metrics CSV",
                                                data=metrics_csv,
                                                file_name=f"batch_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                                mime="text/csv"
                                            )

                                    with col3:
                                        # HTML export (static)
                                        try:
                                            html_report = _generate_batch_html_report(
                                                report_results,
                                                condition_datasets,
                                                pixel_size,
                                                frame_interval,
                                                interactive=False
                                            )
                                            st.download_button(
                                                "📰 HTML Report",
                                                data=html_report,
                                                file_name=f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                                mime="text/html"
                                            )
                                        except Exception as e:
                                            st.error(f"HTML export error: {e}")

                                    with col4:
                                        # Interactive HTML export
                                        try:
                                            interactive_html = _generate_batch_html_report(
                                                report_results,
                                                condition_datasets,
                                                pixel_size,
                                                frame_interval,
                                                interactive=True
                                            )
                                            st.download_button(
                                                "🎨 Interactive HTML",
                                                data=interactive_html,
                                                file_name=f"batch_report_interactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                                mime="text/html"
                                            )
                                        except Exception as e:
                                            st.error(f"Interactive HTML export error: {e}")

                            except Exception as e:
                                st.error(f"Error generating reports: {e}")
                                import traceback
                                with st.expander("🐛 Error Details"):
                                    st.code(traceback.format_exc())

                st.divider()

                # Legacy comparative report button (simpler version)
                if st.button("📈 Generate Comparative Report", type="secondary"):
                    if len(conditions_to_analyze) < 1:
                        st.error("Select at least one condition to analyze")
                    else:
                        with st.spinner("Generating comparative report..."):
                            try:
                                # Pool data from each condition, breaking down by subpopulation if detected
                                condition_datasets = {}
                                subpop_info_comp = {}

                                for cond in conditions_to_analyze:
                                    # Check if subpopulation results exist for this condition
                                    has_subpop = ('subpopulation_results' in st.session_state and
                                                 cond.id in st.session_state.subpopulation_results)

                                    if has_subpop:
                                        result = st.session_state.subpopulation_results[cond.id]

                                        if result.get('subpopulations_detected'):
                                            # Use pre-loaded pooled data with subpopulation labels
                                            pooled_df = result.get('pooled_tracks')

                                            if pooled_df is not None and not pooled_df.empty and 'subpopulation' in pooled_df.columns:
                                                # Split into separate datasets by subpopulation
                                                n_subpops = result['n_subpopulations']
                                                subpop_info_comp[cond.name] = {'n_subpopulations': n_subpops}

                                                for subpop_id in range(n_subpops):
                                                    subpop_df = pooled_df[pooled_df['subpopulation'] == subpop_id].copy()
                                                    if not subpop_df.empty:
                                                        dataset_name = f"{cond.name} - Subpop {subpop_id}"
                                                        condition_datasets[dataset_name] = subpop_df
                                                continue

                                    # No subpopulations detected or no subpopulation analysis - pool normally
                                    pooled_result = cond.pool_tracks()
                                    if isinstance(pooled_result, tuple):
                                        pooled_df, _ = pooled_result
                                    else:
                                        pooled_df = pooled_result
                                    if pooled_df is not None and not pooled_df.empty:
                                        condition_datasets[cond.name] = pooled_df

                                if not condition_datasets:
                                    st.error("No valid data in selected conditions")
                                else:
                                    if subpop_info_comp:
                                        st.success(f"✅ Pooled data from {len(conditions_to_analyze)} conditions ({len(condition_datasets)} datasets including subpopulations)")
                                        with st.expander("ℹ️ Subpopulation Breakdown", expanded=False):
                                            for cond_name, info in subpop_info_comp.items():
                                                st.write(f"**{cond_name}:** {info['n_subpopulations']} subpopulations detected")
                                    else:
                                        st.success(f"✅ Pooled data from {len(condition_datasets)} conditions")

                                    # Show summary
                                    st.subheader("Condition Summaries")
                                    summary_data = []
                                    for name, df in condition_datasets.items():
                                        n_tracks = df['track_id'].nunique() if 'track_id' in df.columns else 0
                                        n_frames = df['frame'].nunique() if 'frame' in df.columns else 0
                                        n_points = len(df)
                                        summary_data.append({
                                            'Condition': name,
                                            'Tracks': n_tracks,
                                            'Frames': n_frames,
                                            'Data Points': n_points
                                        })

                                    import pandas as pd
                                    summary_df = pd.DataFrame(summary_data)
                                    st.dataframe(summary_df, use_container_width=True)

                                    # Option to run enhanced report generator on each condition
                                    st.subheader("Advanced Analysis")
                                    st.info("💡 Use the Enhanced Report Generator tab to run detailed analyses on individual conditions")

                                    # Quick access to load condition into main analysis
                                    st.write("**Load condition for detailed analysis:**")
                                    selected_cond_name = st.selectbox(
                                        "Select condition to load into main workspace",
                                        options=list(condition_datasets.keys()),
                                        key="load_cond_to_workspace"
                                    )

                                    if st.button("Load Selected Condition", key="load_cond_btn"):
                                        st.session_state.tracks_data = condition_datasets[selected_cond_name]
                                        try:
                                            st.session_state.track_statistics = calculate_track_statistics(
                                                condition_datasets[selected_cond_name]
                                            )
                                        except Exception:
                                            pass
                                        st.success(f"✅ Loaded '{selected_cond_name}' into main workspace. Go to 'Enhanced Report Generator' tab to run analyses.")

                            except Exception as e:
                                st.error(f"Error generating report: {e}")
                                import traceback
                                st.code(traceback.format_exc())

                # Export options
                st.divider()
                if st.button("💾 Export All Condition Data"):
                    try:
                        export_data = {}
                        for cond in conditions_to_analyze:
                            pooled_result = cond.pool_tracks()
                            if isinstance(pooled_result, tuple):
                                pooled_df, _ = pooled_result
                            else:
                                pooled_df = pooled_result
                            if pooled_df is not None and not pooled_df.empty:
                                export_data[cond.name] = pooled_df

                        if export_data:
                            # Create a combined CSV with condition labels
                            combined_rows = []
                            for cond_name, df in export_data.items():
                                df_copy = df.copy()
                                df_copy['condition'] = cond_name
                                combined_rows.append(df_copy)

                            import pandas as pd
                            combined_df = pd.concat(combined_rows, ignore_index=True)

                            csv = combined_df.to_csv(index=False)
                            st.download_button(
                                label="Download Combined CSV",
                                data=csv,
                                file_name=f"{proj.name}_all_conditions.csv",
                                mime="text/csv"
                            )
                            st.success(f"✅ Prepared {len(combined_df)} rows from {len(export_data)} conditions")
                        else:
                            st.error("No data to export")
                    except Exception as e:
                        st.error(f"Export failed: {e}")
        else:
            st.info("Add files to conditions to enable batch analysis")

        # Save project explicitly
        if st.button("Save Project"):
            pmgr.save_project(proj, os.path.join(pmgr.projects_dir, f"{proj.id}.json"))
            st.success("Project saved.")
