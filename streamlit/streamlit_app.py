"""
INTEGRATE Rejection Sampling Web Interface

A Streamlit web application for running probabilistic inversion using rejection sampling
through the integrate_rejection function. This interface allows users to upload HDF5 files,
configure parameters, and monitor execution progress.
"""

import streamlit as st
import numpy as np
import h5py
import os
import tempfile
import sys
import traceback
from datetime import datetime
import logging

# Add the current directory to Python path to import integrate
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import integrate as ig
except ImportError:
    st.error("Could not import integrate module. Make sure it's installed and available in the Python path.")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Removed complex subprocess and progress parsing functions
# Using simple direct execution instead

# Page configuration
st.set_page_config(
    page_title="INTEGRATE Rejection Sampling",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

def validate_h5_file_path(file_path, file_type="prior"):
    """
    Validate HDF5 file structure from file path.
    
    Parameters
    ----------
    file_path : str
        Path to HDF5 file on disk
    file_type : str
        Type of file: "prior" or "data"
        
    Returns
    -------
    bool, str
        Validation status and message
    """
    try:
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}"
        
        if not file_path.lower().endswith(('.h5', '.hdf5')):
            return False, f"File must be HDF5 format (.h5 or .hdf5): {file_path}"
        
        with h5py.File(file_path, 'r') as f:
            if file_type == "prior":
                required_datasets = ['D1']  # At least D1 should exist
                for dataset in required_datasets:
                    if dataset not in f:
                        return False, f"Missing required dataset: {dataset}"
                        
            elif file_type == "data":
                # Check for data structure - look for d_obs in D1, D2, etc. or at root
                found_d_obs = False
                if 'd_obs' in f:
                    found_d_obs = True
                else:
                    # Check in D1, D2, etc.
                    for key in f.keys():
                        if key.startswith('D') and hasattr(f[key], 'keys'):
                            if 'd_obs' in f[key]:
                                found_d_obs = True
                                break
                
                if not found_d_obs:
                    return False, "Missing required dataset: d_obs (not found in root or D1/D2/etc.)"
        
        return True, "File validation successful"
        
    except Exception as e:
        return False, f"File validation error: {str(e)}"

def get_file_info_from_path(file_path, file_type="prior"):
    """
    Extract information from HDF5 file path.
    
    Parameters
    ----------
    file_path : str
        Path to HDF5 file on disk
    file_type : str
        Type of file: "prior" or "data"
        
    Returns
    -------
    dict
        File information dictionary
    """
    try:
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}
        
        info = {}
        with h5py.File(file_path, 'r') as f:
            if file_type == "prior":
                # Get available data types
                data_types = [key for key in f.keys() if key.startswith('D')]
                info['data_types'] = sorted(data_types)
                
                # Get sample size from first data type
                if data_types:
                    first_data = f[data_types[0]]
                    info['n_samples'] = first_data.shape[0]
                    info['n_features'] = first_data.shape[1] if len(first_data.shape) > 1 else 1
                    
            elif file_type == "data":
                # Look for d_obs at root level first
                if 'd_obs' in f:
                    d_obs = f['d_obs']
                    if len(d_obs.shape) == 3:  # Multiple data types
                        info['n_data_points'] = d_obs.shape[1]
                        info['n_data_types'] = d_obs.shape[0]
                        info['n_features'] = d_obs.shape[2]
                    else:  # Single data type
                        info['n_data_points'] = d_obs.shape[0]
                        info['n_data_types'] = 1
                        info['n_features'] = d_obs.shape[1] if len(d_obs.shape) > 1 else 1
                else:
                    # Look for d_obs in D1, D2, etc.
                    data_types = []
                    total_points = 0
                    total_features = 0
                    
                    for key in sorted(f.keys()):
                        if key.startswith('D') and hasattr(f[key], 'keys'):
                            if 'd_obs' in f[key]:
                                data_types.append(key)
                                d_obs = f[key]['d_obs']
                                if total_points == 0:  # First data type sets the reference
                                    total_points = d_obs.shape[0]
                                    total_features = d_obs.shape[1] if len(d_obs.shape) > 1 else 1
                    
                    info['n_data_points'] = total_points
                    info['n_data_types'] = len(data_types)
                    info['n_features'] = total_features
                    info['data_types_found'] = data_types
        
        # Add file size info
        file_size = os.path.getsize(file_path) / (1024*1024)  # MB
        info['file_size_mb'] = file_size
        
        return info
        
    except Exception as e:
        return {"error": str(e)}

# File upload function removed - using direct file paths instead

def main():
    """Main Streamlit application."""
    
    # Title and description
    st.title("🧮 INTEGRATE Rejection Sampling")
    st.markdown("""
    Web interface for probabilistic inversion using rejection sampling methodology.
    Enter file paths to your prior model and observed data files to perform Bayesian inversion.
    **Supports large files (>200MB)** by working directly with files on disk.
    """)
    
    # Initialize session state
    if 'execution_complete' not in st.session_state:
        st.session_state.execution_complete = False
    if 'result_file' not in st.session_state:
        st.session_state.result_file = None
    
    # Sidebar for file paths
    with st.sidebar:
        st.header("📁 File Paths")
        st.info("💡 **For Large Files**: Enter file paths directly instead of uploading. Supports files >200MB.")
        
        # Prior file path
        prior_path = st.text_input(
            "Prior HDF5 File Path",
            value="",
            help="Full path to HDF5 file containing prior model and forward modeled data (D1, D2, etc.)",
            key="prior_path",
            placeholder="/path/to/prior.h5"
        )
        
        # Data file path
        data_path = st.text_input(
            "Data HDF5 File Path", 
            value="",
            help="Full path to HDF5 file containing observed data (d_obs, d_std, etc.)",
            key="data_path",
            placeholder="/path/to/data.h5"
        )
        
        # Output filename
        output_name = st.text_input(
            "Output Filename (optional)",
            value="",
            help="Leave empty for auto-generated filename",
            key="output_name",
            placeholder="results.h5"
        )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col2:
        # File information display
        if prior_path.strip():
            st.subheader("Prior File Info")
            with st.spinner("Validating prior file..."):
                valid, msg = validate_h5_file_path(prior_path.strip(), "prior")
                if valid:
                    st.success("✅ Valid prior file")
                    info = get_file_info_from_path(prior_path.strip(), "prior")
                    if 'error' not in info:
                        st.write(f"**File Size:** {info.get('file_size_mb', 0):.1f} MB")
                        st.write(f"**Samples:** {info.get('n_samples', 'N/A'):,}")
                        st.write(f"**Features:** {info.get('n_features', 'N/A')}")
                        st.write(f"**Data Types:** {len(info.get('data_types', []))}")
                        if info.get('data_types'):
                            st.write("Available:", ", ".join(info['data_types']))
                    else:
                        st.error(f"Error: {info['error']}")
                else:
                    st.error(f"❌ {msg}")
        
        if data_path.strip():
            st.subheader("Data File Info")
            with st.spinner("Validating data file..."):
                valid, msg = validate_h5_file_path(data_path.strip(), "data")
                if valid:
                    st.success("✅ Valid data file")
                    info = get_file_info_from_path(data_path.strip(), "data")
                    if 'error' not in info:
                        st.write(f"**File Size:** {info.get('file_size_mb', 0):.1f} MB")
                        st.write(f"**Data Points:** {info.get('n_data_points', 'N/A'):,}")
                        st.write(f"**Data Types:** {info.get('n_data_types', 'N/A')}")
                        st.write(f"**Features:** {info.get('n_features', 'N/A')}")
                        if info.get('data_types_found'):
                            st.write("Available:", ", ".join(info['data_types_found']))
                    else:
                        st.error(f"Error: {info['error']}")
                else:
                    st.error(f"❌ {msg}")
    
    with col1:
        # Core parameters section
        st.header("⚙️ Core Parameters")
        
        col_param1, col_param2 = st.columns(2)
        
        with col_param1:
            N_use = st.number_input(
                "Maximum Prior Samples (N_use)",
                min_value=1,
                max_value=10**12,
                value=100000,
                step=1000,
                format="%d",
                help="Maximum number of prior samples to use for inversion"
            )
            
            id_use_str = st.text_input(
                "Data Types to Use (id_use)",
                value="",
                help="Comma-separated list of data type IDs (e.g., '1,2,3'). Leave empty to use all.",
                placeholder="1,2,3 or leave empty for all"
            )
        
        with col_param2:
            nr = st.number_input(
                "Posterior Samples per Data Point (nr)",
                min_value=1,
                max_value=10000,
                value=400,
                step=10,
                help="Number of posterior samples to retain per data point"
            )
            
            ip_range_str = st.text_input(
                "Data Point Range (ip_range)",
                value="",
                help="Range of data point indices to invert (e.g., '0,100' or '0:100'). Leave empty for all.",
                placeholder="0,100 or 0:100 or leave empty for all"
            )
        
        # Advanced options (collapsible)
        with st.expander("🔧 Advanced Options", expanded=False):
            st.subheader("Temperature Control")
            col_temp1, col_temp2 = st.columns(2)
            
            with col_temp1:
                autoT = st.selectbox(
                    "Automatic Temperature (autoT)",
                    options=[1, 0],
                    format_func=lambda x: "Enabled" if x == 1 else "Disabled",
                    index=0,
                    help="Automatic temperature estimation for rejection sampling"
                )
            
            with col_temp2:
                T_base = st.number_input(
                    "Base Temperature (T_base)",
                    min_value=0.1,
                    max_value=100.0,
                    value=1.0,
                    step=0.1,
                    help="Base temperature when autoT is disabled",
                    disabled=(autoT == 1)
                )
            
            st.subheader("Parallel Processing")
            col_par1, col_par2, col_par3 = st.columns(3)
            
            with col_par1:
                parallel = st.checkbox(
                    "Enable Parallel Processing",
                    value=True,
                    help="Use multiprocessing for faster execution"
                )
            
            with col_par2:
                Ncpu = st.number_input(
                    "Number of CPUs (Ncpu)",
                    min_value=0,
                    max_value=64,
                    value=0,
                    step=1,
                    help="Number of CPU cores to use (0 = auto-detect)",
                    disabled=not parallel
                )
            
            with col_par3:
                Nchunks = st.number_input(
                    "Number of Chunks (Nchunks)",
                    min_value=0,
                    max_value=100,
                    value=0,
                    step=1,
                    help="Number of processing chunks (0 = auto-determine)",
                    disabled=not parallel
                )
            
            st.subheader("Performance Tuning")
            col_perf1, col_perf2 = st.columns(2)
            
            with col_perf1:
                use_N_best = st.number_input(
                    "Use N Best Samples (use_N_best)",
                    min_value=0,
                    max_value=10000,
                    value=0,
                    step=100,
                    help="Use only the N best-fitting samples (0 = disabled)"
                )
            
            with col_perf2:
                updatePostStat = st.checkbox(
                    "Update Posterior Statistics",
                    value=True,
                    help="Calculate and save posterior statistics"
                )
            
            st.subheader("Debug Options")
            showInfo = st.selectbox(
                "Information Level (showInfo)",
                options=[0, 1, 2, 3],
                format_func=lambda x: {0: "Minimal", 1: "Standard", 2: "Detailed", 3: "Debug"}[x],
                index=1,
                help="Level of information to display during execution"
            )
        
        # Execution section
        st.header("▶️ Run Analysis")
        
        # Check if file paths are provided
        files_ready = bool(prior_path.strip()) and bool(data_path.strip())
        
        if not files_ready:
            st.warning("Please enter both Prior HDF5 and Data HDF5 file paths to continue.")
        
        # Parse parameters
        def parse_list_param(param_str):
            """Parse comma-separated string to list of integers."""
            if not param_str.strip():
                return []
            try:
                return [int(x.strip()) for x in param_str.split(',')]
            except ValueError:
                st.error(f"Invalid format: {param_str}. Use comma-separated integers.")
                return None
        
        def parse_range_param(range_str):
            """Parse range string to list of integers."""
            if not range_str.strip():
                return []
            try:
                if ':' in range_str:
                    start, end = map(int, range_str.split(':'))
                    return list(range(start, end))
                elif ',' in range_str:
                    return [int(x.strip()) for x in range_str.split(',')]
                else:
                    return [int(range_str.strip())]
            except ValueError:
                st.error(f"Invalid format: {range_str}. Use 'start:end' or 'n1,n2,n3'.")
                return None
        
        # Run button
        if st.button("🚀 Start Rejection Sampling", disabled=not files_ready, type="primary"):
            # Parse parameters
            id_use_list = parse_list_param(id_use_str) if id_use_str else []
            ip_range_list = parse_range_param(ip_range_str) if ip_range_str else []
            
            if (id_use_str and id_use_list is None) or (ip_range_str and ip_range_list is None):
                st.stop()  # Stop if parameter parsing failed
            
            # Use file paths directly (no need to copy files)
            prior_file_path = prior_path.strip()
            data_file_path = data_path.strip()
            
            # Set output path
            if output_name.strip():
                output_path = os.path.join(tempfile.mkdtemp(), output_name)
                if not output_path.endswith('.h5'):
                    output_path += '.h5'
            else:
                output_path = ""  # Let function auto-generate
            
            # Execute integrate_rejection directly with simple progress indication
            try:
                st.info("🔄 Running rejection sampling... This may take several minutes.")
                
                # Create a simple progress indicator
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("⏳ Initializing rejection sampling...")
                
                # Show the function is running
                with st.spinner("Running integrate_rejection..."):
                    
                    # Execute the function directly
                    result_file = ig.integrate_rejection(
                        f_prior_h5=prior_file_path,
                        f_data_h5=data_file_path,
                        f_post_h5=output_path,
                        N_use=N_use,
                        id_use=id_use_list,
                        ip_range=ip_range_list,
                        nr=nr,
                        autoT=autoT,
                        T_base=T_base,
                        Nchunks=Nchunks,
                        Ncpu=Ncpu,
                        parallel=parallel,
                        use_N_best=use_N_best,
                        showInfo=showInfo,
                        updatePostStat=updatePostStat,
                    )
                
                # Store results in session state
                st.session_state.execution_complete = True
                st.session_state.result_file = result_file
                
                # Update final progress
                progress_bar.progress(100)
                status_text.text("✅ Rejection sampling completed successfully!")
                
                st.success("✅ Rejection sampling completed successfully!")
                
                # No cleanup needed - using files directly from disk
                
            except Exception as e:
                st.error(f"❌ Error during execution: {str(e)}")
                st.code(traceback.format_exc())
                
                # No cleanup needed - using files directly from disk
        
        # Results section
        if st.session_state.execution_complete and st.session_state.result_file:
            st.header("📊 Results")
            
            result_file = st.session_state.result_file
            
            if os.path.exists(result_file):
                st.success(f"Results saved to: `{result_file}`")
                
                # Display file info
                try:
                    file_size = os.path.getsize(result_file) / (1024*1024)  # MB
                    st.info(f"Output file size: {file_size:.2f} MB")
                    
                    # Show basic results summary
                    with h5py.File(result_file, 'r') as f:
                        st.subheader("Result Summary")
                        
                        # Basic info
                        if 'T' in f:
                            T_values = f['T'][:]
                            T_mean = np.nanmean(T_values)
                            st.metric("Average Temperature", f"{T_mean:.2f}")
                        
                        if 'EV' in f:
                            EV_values = f['EV'][:]
                            EV_mean = np.nanmean(EV_values)
                            st.metric("Average Evidence", f"{EV_mean:.2f}")
                        
                        if 'N_UNIQUE' in f:
                            N_unique = f['N_UNIQUE'][:]
                            N_unique_mean = np.nanmean(N_unique)
                            st.metric("Average Unique Samples", f"{N_unique_mean:.0f}")
                        
                        # Show attributes
                        if f.attrs:
                            st.subheader("Execution Details")
                            for key, value in f.attrs.items():
                                if isinstance(value, (str, int, float)):
                                    st.text(f"{key}: {value}")
                
                except Exception as e:
                    st.warning(f"Could not read result file details: {e}")
                
                # Download button
                try:
                    with open(result_file, 'rb') as f:
                        st.download_button(
                            label="📥 Download Results",
                            data=f.read(),
                            file_name=os.path.basename(result_file),
                            mime="application/octet-stream"
                        )
                except Exception as e:
                    st.error(f"Could not create download: {e}")
            else:
                st.error("Result file not found!")

if __name__ == "__main__":
    main()