import streamlit as st

import os
import h5py
import integrate as ig

def app():
    st.title("Posterior")
    
    # Add a list box with all *.h5 files in the current directory that starts with 'POST'
    st.write("Select a file")
    files = [f for f in os.listdir('.') if f.startswith('POST') and f.endswith('.h5')]
    f_post_h5 = st.selectbox("Select a file", files)
    st.write("You selected", f_post_h5)

    # Get 'f_prior' and 'f_data' from the selected file 
    # and display them in the sidebar

    with h5py.File(f_post_h5,'r') as f_post:
        f_prior_h5 = f_post['/'].attrs['f5_prior']
        f_data_h5 = f_post['/'].attrs['f5_data']

        st.sidebar.write("f_prior_h5:", f_prior_h5)
        st.sidebar.write("f_data_h5:", f_data_h5)
        st.write("f_prior_h5:", f_prior_h5)
        st.write("f_data_h5:", f_data_h5)

        #ig.plot_T_EV(f_post_h5,i1=0, i2=100)    

    


# If this script is run directly, call the app() function
if __name__ == "__main__":
    app()        
        
    
