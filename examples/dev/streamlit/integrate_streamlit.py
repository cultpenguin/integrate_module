import streamlit as st
import integrate_streamlit_data
import integrate_streamlit_prior
import integrate_streamlit_posterior
import integrate_streamlit_forward

# Set page layout to wide
st.set_page_config(layout="wide")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["App 1", "App 2", "App 3,", "Posterior"])

# Display content in each tab
with tab1:
    integrate_streamlit_data.app()

with tab2:
    integrate_streamlit_prior.app()

with tab3:
    integrate_streamlit_forward.app()

with tab4:
    integrate_streamlit_posterior.app()
