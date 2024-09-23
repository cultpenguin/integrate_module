import streamlit as st
import os 
import integrate as ig
import matplotlib.pyplot as plt
import h5py

def app():
    st.title("DATA")

        
    # Add a list box with all *.h5 files in the current directory that do not start with 'PRIOR' or 'POST' and end with '.h5'
    # st.write("Select a file")
    files = [f for f in os.listdir('.') if not f.startswith(('PRIOR', 'POST')) and f.endswith('.h5')]
    selected_files = []
    for file in files:
        with h5py.File(file, 'r') as f:
            if '/LINE' in f:
                selected_files.append(file)
    f_data_h5 = st.selectbox("Select a HDF5 file with data/geometry", selected_files)
    st.write("You selected", f_data_h5)

    fig = ig.plot_data_xy(f_data_h5)
    st.pyplot(fig)

    # Show the number of data points, the number of lines, and the total length of the all lines
    with h5py.File(f_data_h5, 'r') as f:
        LINE = f['/LINE'][:]
        n_data = LINE.shape[0]

    lines = set(LINE.flatten())
    n_lines = len(lines)
            
    #n_lines = len(set(LINE))
    #st.write("Number of times an entry in LINE changes:", line_changes)


    st.write("Number of data points:", n_data)
    st.write("Number of lines:", n_lines)
    st.write("Total line kilomteres:")

