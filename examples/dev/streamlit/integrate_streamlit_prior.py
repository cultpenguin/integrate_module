import streamlit as st

def app():
    st.title("Application 1")
    st.write("This is the content of the first application.")
    # Add your specific content and functionality here.
    st.slider("Select a number", 0, 100, key="slider1")