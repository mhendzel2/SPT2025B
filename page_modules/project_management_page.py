"""
Project Management Page Module

Provides tools for organizing tracking data into projects with experimental conditions.
"""

import os
import streamlit as st
from page_modules import register_page

# Import project management module
try:
    import project_management as pm
    PM_AVAILABLE = True
except ImportError:
    PM_AVAILABLE = False


@register_page("Project Management")
def render():
    """
    Render the project management page.
    
    Allows users to:
    - Create and manage projects
    - Define experimental conditions
    - Associate tracking data files with conditions
    - Compare results across conditions
    """
    if not PM_AVAILABLE:
        st.error("Project management module is not available.")
        return
    
    st.title("Project Management: Group cells into experimental conditions")
    
    pmgr = pm.ProjectManager()
    
    # Initialize session state for current project
    if "pm_current" not in st.session_state:
        st.session_state.pm_current = None
    
    # Project selector/creator
    _render_project_selector(pmgr)
    
    # Project details and condition management
    proj = st.session_state.pm_current
    if proj is None:
        st.info("Create or select a project to manage conditions and files.")
    else:
        _render_project_details(pmgr, proj)


def _render_project_selector(pmgr):
    """Render project selection and creation UI."""
    with st.expander("Project Selection", expanded=True):
        existing = pmgr.list_projects()
        options = [f"{p['name']} ({p['id'][:8]})" for p in existing]
        sel = st.selectbox("Select a project", options + ["<New Project>"])
        
        if sel == "<New Project>":
            c1, c2 = st.columns([2, 1])
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


def _render_project_details(pmgr, proj):
    """Render project details and condition management."""
    st.subheader(f"Project: {proj.name}")
    
    # Delete project button
    if st.button("Delete Project", key="pm_delete_project"):
        st.session_state.confirm_delete = True
    
    if st.session_state.get("confirm_delete"):
        _render_delete_confirmation(pmgr, proj)
    
    # Add condition
    _render_add_condition(pmgr, proj)
    
    # List and manage conditions
    _render_conditions(pmgr, proj)


def _render_delete_confirmation(pmgr, proj):
    """Render project deletion confirmation dialog."""
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


def _render_add_condition(pmgr, proj):
    """Render UI for adding new conditions."""
    with st.expander("Add Condition", expanded=True):
        cname = st.text_input("Condition name", key="pm_new_cond_name")
        cdesc = st.text_input("Description", key="pm_new_cond_desc")
        if st.button("Add Condition") and cname.strip():
            pmgr.add_condition(proj, cname.strip(), cdesc.strip())
            pmgr.save_project(proj, os.path.join(pmgr.projects_dir, f"{proj.id}.json"))
            st.success("Condition added.")
            st.rerun()


def _render_conditions(pmgr, proj):
    """Render list of conditions with file management."""
    for cond in list(proj.conditions):
        with st.expander(f"Condition: {cond.name} ({len(cond.files)} files)", expanded=True):
            # Delete condition button
            _render_condition_delete(pmgr, proj, cond)
            
            # File uploader
            _render_file_uploader(pmgr, proj, cond)
            
            # Show files
            _render_condition_files(pmgr, proj, cond)


def _render_condition_delete(pmgr, proj, cond):
    """Render condition deletion UI."""
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


def _render_file_uploader(pmgr, proj, cond):
    """Render file uploader for condition."""
    uploaded = st.file_uploader(
        "Add cell files (CSV)", 
        type=["csv"], 
        accept_multiple_files=True, 
        key=f"pm_up_{cond.id}"
    )
    
    # Track processed files to avoid duplicates
    upload_key = f"pm_upload_processed_{cond.id}"
    if upload_key not in st.session_state:
        st.session_state[upload_key] = set()
    
    if uploaded:
        # Filter for new files only
        new_files = []
        for uf in uploaded:
            file_id = f"{uf.name}_{uf.size}"
            if file_id not in st.session_state[upload_key]:
                new_files.append((uf, file_id))
        
        # Process new files
        if new_files:
            for uf, file_id in new_files:
                try:
                    import pandas as pd
                    df = pd.read_csv(uf)
                    pmgr.add_file_to_condition(proj, cond.id, uf.name, df)
                    st.session_state[upload_key].add(file_id)
                except Exception as e:
                    st.warning(f"Failed to add {uf.name}: {e}")
            pmgr.save_project(proj, os.path.join(pmgr.projects_dir, f"{proj.id}.json"))
            st.success(f"{len(new_files)} file(s) added.")
            st.rerun()


def _render_condition_files(pmgr, proj, cond):
    """Render list of files in condition with preview and remove options."""
    if not cond.files:
        return
    
    for f in list(cond.files):
        fname = f.get('name') or f.get('file_name') or f.get('id')
        cols = st.columns([6, 2, 2])
        cols[0].write(fname)
        
        # Preview button
        if cols[1].button("Preview", key=f"pv_{cond.id}_{f.get('id')}"):
            try:
                import pandas as pd
                import io
                
                if f.get('data'):
                    df = pd.read_csv(io.BytesIO(f['data']))
                elif f.get('data_path') and os.path.exists(f['data_path']):
                    df = pd.read_csv(f['data_path'])
                else:
                    df = None
                
                if df is not None:
                    st.dataframe(df.head())
                else:
                    st.info("No data available for preview.")
            except Exception as e:
                st.warning(f"Preview failed: {e}")
        
        # Remove button
        if cols[2].button("Remove", key=f"rm_{cond.id}_{f.get('id')}"):
            pmgr.remove_file_from_project(proj, cond.id, f.get('id'))
            pmgr.save_project(proj, os.path.join(pmgr.projects_dir, f"{proj.id}.json"))
            st.rerun()
    
    # Clear all files button
    st.write("")  # Spacing
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🗑️ Clear All Files", key=f"clear_all_{cond.id}", type="secondary"):
            for f in list(cond.files):
                pmgr.remove_file_from_project(proj, cond.id, f.get('id'))
            
            # Clear processed tracking
            upload_key = f"pm_upload_processed_{cond.id}"
            if upload_key in st.session_state:
                del st.session_state[upload_key]
            
            pmgr.save_project(proj, os.path.join(pmgr.projects_dir, f"{proj.id}.json"))
            st.success("All files cleared.")
            st.rerun()
