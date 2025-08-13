"""
INTEGRATE Plotting Interface

Streamlit interface for visualization and plotting functions from integrate_plot.py.
Provides access to profile plots, 2D mapping, and data analysis tools.

Author: Generated for the INTEGRATE module
"""

import streamlit as st
import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

# Add the parent directory to Python path to import integrate module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import integrate as ig
    import integrate.integrate_plot as igp
except ImportError:
    st.error("Could not import integrate module. Please ensure it is properly installed.")
    st.stop()

def display_h5_info(file_path):
    """Display basic information about an H5 file"""
    try:
        with h5py.File(file_path, 'r') as f:
            st.subheader(f"File Info: {os.path.basename(file_path)}")
            
            # Show datasets
            datasets = []
            def visit_func(name, obj):
                if isinstance(obj, h5py.Dataset):
                    datasets.append((name, obj.shape, obj.dtype))
            
            f.visititems(visit_func)
            
            if datasets:
                st.write("**Datasets:**")
                for name, shape, dtype in datasets[:10]:  # Show first 10
                    st.text(f"  {name}: {shape} {dtype}")
                if len(datasets) > 10:
                    st.text(f"  ... and {len(datasets) - 10} more datasets")
            
            # Show attributes
            if f.attrs:
                st.write("**File Attributes:**")
                for key, value in list(f.attrs.items())[:5]:  # Show first 5
                    st.text(f"  {key}: {value}")
                if len(f.attrs) > 5:
                    st.text(f"  ... and {len(f.attrs) - 5} more attributes")
                    
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")

def get_h5_files():
    """Get list of H5 files in current directory"""
    return [f for f in os.listdir('.') if f.endswith('.h5')]

def get_post_files():
    """Get H5 files that look like posterior files"""
    h5_files = get_h5_files()
    post_files = []
    for f in h5_files:
        if 'POST' in f.upper():
            post_files.append(f)
    return post_files if post_files else h5_files

def get_data_files():
    """Get H5 files that look like data files"""
    h5_files = get_h5_files()
    data_files = []
    for f in h5_files:
        if any(x in f.upper() for x in ['DATA', 'OBS', 'TEM', 'DAUGAARD', 'AVG']):
            data_files.append(f)
    return data_files if data_files else h5_files

def get_prior_files():
    """Get H5 files that look like prior files"""
    h5_files = get_h5_files()
    prior_files = []
    for f in h5_files:
        if 'PRIOR' in f.upper():
            prior_files.append(f)
    return prior_files if prior_files else h5_files

def capture_plot():
    """Capture current matplotlib plot as image for Streamlit display"""
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    return buf

def run_plot_app():
    st.header("Visualization & Plotting")
    
    st.markdown("""
    Create publication-quality visualizations of INTEGRATE results including
    profiles, 2D maps, data comparisons, and statistical plots.
    """)
    
    # Plot type selection
    plot_type = st.selectbox(
        "Select Plot Type",
        [
            "Profile Plots",
            "2D Feature Mapping", 
            "Temperature & Evidence",
            "Data Visualization",
            "Geometry Plots",
            "Prior Statistics",
            "Data-Model Comparison"
        ]
    )
    
    st.markdown("---")
    
    if plot_type == "Profile Plots":
        st.subheader("1D Profile Plots")
        
        # File selection
        post_files = get_post_files()
        if post_files:
            f_post_h5 = st.selectbox("Select posterior file:", post_files)
            
            col1, col2 = st.columns(2)
            with col1:
                i1 = st.number_input("Start data point", value=1, min_value=1)
                i2 = st.number_input("End data point (large number = all)", value=100, min_value=1)
                im = st.number_input("Model index (0 = all models)", value=1, min_value=0)
            
            with col2:
                hardcopy = st.checkbox("Save plot to file", value=False)
                txt = st.text_input("Additional text for filename", "")
                showInfo = st.selectbox("Verbosity level", [0, 1, 2], index=0)
            
            if st.button("Generate Profile Plot"):
                try:
                    plt.ioff()  # Turn off interactive mode
                    fig = plt.figure(figsize=(12, 8))
                    
                    igp.plot_profile(
                        f_post_h5=f_post_h5,
                        i1=i1,
                        i2=i2,
                        im=im,
                        hardcopy=hardcopy,
                        txt=txt,
                        showInfo=showInfo
                    )
                    
                    buf = capture_plot()
                    st.image(buf, caption=f"Profile plot from {os.path.basename(f_post_h5)}")
                    plt.close()
                    
                except Exception as e:
                    st.error(f"Error generating plot: {str(e)}")
        else:
            st.warning("No posterior files found")
    
    elif plot_type == "2D Feature Mapping":
        st.subheader("2D Spatial Feature Maps")
        
        post_files = get_post_files()
        if post_files:
            f_post_h5 = st.selectbox("Select posterior file:", post_files)
            
            col1, col2 = st.columns(2)
            with col1:
                im = st.number_input("Model index", value=1, min_value=1)
                iz = st.number_input("Feature/layer index", value=0, min_value=0)
                key = st.text_input("Dataset key (empty = auto-detect)", "")
                
            with col2:
                i1 = st.number_input("Start data point", value=1, min_value=1)
                i2 = st.number_input("End data point", value=1000, min_value=1)
                uselog = st.checkbox("Use logarithmic color scale", value=True)
                hardcopy = st.checkbox("Save plot to file", value=False)
            
            title_text = st.text_input("Additional title text", "")
            
            if st.button("Generate 2D Feature Map"):
                try:
                    plt.ioff()
                    fig = plt.figure(figsize=(12, 8))
                    
                    igp.plot_feature_2d(
                        f_post_h5=f_post_h5,
                        key=key,
                        i1=i1,
                        i2=i2,
                        im=im,
                        iz=iz,
                        uselog=int(uselog),
                        title_text=title_text,
                        hardcopy=hardcopy
                    )
                    
                    buf = capture_plot()
                    st.image(buf, caption=f"2D feature map from {os.path.basename(f_post_h5)}")
                    plt.close()
                    
                except Exception as e:
                    st.error(f"Error generating plot: {str(e)}")
        else:
            st.warning("No posterior files found")
    
    elif plot_type == "Temperature & Evidence":
        st.subheader("Temperature and Evidence Fields")
        
        post_files = get_post_files()
        if post_files:
            f_post_h5 = st.selectbox("Select posterior file:", post_files)
            
            col1, col2 = st.columns(2)
            with col1:
                i1 = st.number_input("Start data point", value=1, min_value=1)
                i2 = st.number_input("End data point", value=1000, min_value=1)
                s = st.number_input("Marker size", value=5, min_value=1)
                
            with col2:
                pl = st.selectbox("Plot type", ["all", "T", "EV", "ND"])
                T_min = st.number_input("Min temperature", value=1, min_value=1)
                T_max = st.number_input("Max temperature", value=100, min_value=1)
                hardcopy = st.checkbox("Save plot to file", value=False)
            
            if st.button("Generate T/EV Plot"):
                try:
                    plt.ioff()
                    
                    igp.plot_T_EV(
                        f_post_h5=f_post_h5,
                        i1=i1,
                        i2=i2,
                        s=s,
                        T_min=T_min,
                        T_max=T_max,
                        pl=pl,
                        hardcopy=hardcopy
                    )
                    
                    buf = capture_plot()
                    st.image(buf, caption=f"Temperature/Evidence from {os.path.basename(f_post_h5)}")
                    plt.close()
                    
                except Exception as e:
                    st.error(f"Error generating plot: {str(e)}")
        else:
            st.warning("No posterior files found")
    
    elif plot_type == "Data Visualization":
        st.subheader("Observational Data Plots")
        
        data_files = get_data_files()
        if data_files:
            f_data_h5 = st.selectbox("Select data file:", data_files)
            
            col1, col2 = st.columns(2)
            with col1:
                plType = st.selectbox("Plot style", ["imshow", "plot"])
                Dkey = st.text_input("Data key (empty = auto-detect)", "")
                
            with col2:
                i_plot_text = st.text_input("Data indices to plot (comma-separated, empty = all)", "")
                hardcopy = st.checkbox("Save plot to file", value=False)
            
            # Parse data indices
            i_plot = []
            if i_plot_text:
                try:
                    i_plot = [int(x.strip()) for x in i_plot_text.split(',')]
                except:
                    st.error("Invalid data index format")
            
            if st.button("Generate Data Plot"):
                try:
                    plt.ioff()
                    
                    igp.plot_data(
                        f_data_h5=f_data_h5,
                        i_plot=i_plot,
                        Dkey=Dkey,
                        plType=plType,
                        hardcopy=hardcopy
                    )
                    
                    buf = capture_plot()
                    st.image(buf, caption=f"Data plot from {os.path.basename(f_data_h5)}")
                    plt.close()
                    
                except Exception as e:
                    st.error(f"Error generating plot: {str(e)}")
        else:
            st.warning("No data files found")
    
    elif plot_type == "Geometry Plots":
        st.subheader("Survey Geometry Visualization")
        
        data_files = get_data_files()
        if data_files:
            f_data_h5 = st.selectbox("Select data file:", data_files)
            
            col1, col2 = st.columns(2)
            with col1:
                pl = st.selectbox("Geometry type", ["all", "LINE", "ELEVATION", "id"])
                s = st.number_input("Marker size", value=5, min_value=1)
                
            with col2:
                i1 = st.number_input("Start index", value=0, min_value=0)
                i2 = st.number_input("End index (0 = all)", value=0, min_value=0)
                hardcopy = st.checkbox("Save plot to file", value=False)
            
            if st.button("Generate Geometry Plot"):
                try:
                    plt.ioff()
                    
                    igp.plot_geometry(
                        f_data_h5=f_data_h5,
                        i1=i1,
                        i2=i2,
                        s=s,
                        pl=pl,
                        hardcopy=hardcopy
                    )
                    
                    buf = capture_plot()
                    st.image(buf, caption=f"Geometry plot from {os.path.basename(f_data_h5)}")
                    plt.close()
                    
                except Exception as e:
                    st.error(f"Error generating plot: {str(e)}")
        else:
            st.warning("No data files found")
    
    elif plot_type == "Prior Statistics":
        st.subheader("Prior Model Statistics")
        
        prior_files = get_prior_files()
        if prior_files:
            f_prior_h5 = st.selectbox("Select prior file:", prior_files)
            
            col1, col2 = st.columns(2)
            with col1:
                Mkey = st.text_input("Model key (empty = all models)", "")
                nr = st.number_input("Number of realizations to show", value=100, min_value=10)
                
            with col2:
                hardcopy = st.checkbox("Save plot to file", value=True)
            
            if st.button("Generate Prior Statistics"):
                try:
                    plt.ioff()
                    
                    igp.plot_prior_stats(
                        f_prior_h5=f_prior_h5,
                        Mkey=Mkey,
                        nr=nr,
                        hardcopy=hardcopy
                    )
                    
                    buf = capture_plot()
                    st.image(buf, caption=f"Prior statistics from {os.path.basename(f_prior_h5)}")
                    plt.close()
                    
                except Exception as e:
                    st.error(f"Error generating plot: {str(e)}")
        else:
            st.warning("No prior files found")
    
    elif plot_type == "Data-Model Comparison":
        st.subheader("Compare Data with Model Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Prior/Forward Model File**")
            prior_files = get_h5_files()
            if prior_files:
                f_prior_data_h5 = st.selectbox("Select prior/forward file:", prior_files)
            else:
                st.warning("No files found")
                f_prior_data_h5 = None
        
        with col2:
            st.write("**Observational Data File**") 
            data_files = get_data_files()
            if data_files:
                f_data_h5 = st.selectbox("Select data file:", data_files)
            else:
                st.warning("No data files found")
                f_data_h5 = None
        
        if f_prior_data_h5 and f_data_h5:
            col1, col2 = st.columns(2)
            with col1:
                nr = st.number_input("Number of realizations", value=1000, min_value=10)
                id = st.number_input("Data identifier", value=1, min_value=1)
                alpha = st.slider("Line transparency", 0.1, 1.0, 0.5)
                
            with col2:
                d_str = st.selectbox("Data array", ["d_obs", "d_std"])
                hardcopy = st.checkbox("Save plot to file", value=True)
                ylim_text = st.text_input("Y-axis limits (min,max)", "")
            
            # Parse y limits
            ylim = None
            if ylim_text:
                try:
                    ylim = [float(x.strip()) for x in ylim_text.split(',')]
                    if len(ylim) != 2:
                        ylim = None
                except:
                    st.error("Invalid y-limit format")
            
            if st.button("Generate Data-Model Comparison"):
                try:
                    plt.ioff()
                    
                    result = igp.plot_data_prior(
                        f_prior_data_h5=f_prior_data_h5,
                        f_data_h5=f_data_h5,
                        nr=nr,
                        id=id,
                        d_str=d_str,
                        alpha=alpha,
                        ylim=ylim,
                        hardcopy=hardcopy
                    )
                    
                    if result:
                        buf = capture_plot()
                        st.image(buf, caption="Data-Model Comparison")
                        plt.close()
                    
                except Exception as e:
                    st.error(f"Error generating plot: {str(e)}")
    
    # Information section
    st.markdown("---")
    st.subheader("Plotting Information")
    
    with st.expander("Plot Types Overview"):
        st.markdown("""
        **Profile Plots**: 1D vertical sections showing parameter variation with depth
        - Discrete models: Mode, entropy, temperature/evidence curves
        - Continuous models: Mean/median, standard deviation, confidence intervals
        
        **2D Feature Mapping**: Spatial distribution of model parameters
        - Color-coded scatter plots of parameter values
        - Support for logarithmic scaling and custom color limits
        
        **Temperature & Evidence**: Quality control plots
        - Temperature: sampling efficiency (lower = better convergence)
        - Evidence: data fit quality (higher = better fit)
        
        **Data Visualization**: Observational data analysis
        - Time series, image displays, signal-to-noise ratios
        - Support for different plot styles and data selection
        
        **Geometry Plots**: Survey layout visualization  
        - Survey lines, elevation data, data point indices
        - Useful for quality control and survey planning
        """)

if __name__ == "__main__":
    run_plot_app()