import streamlit as st

def app():
    st.title("FORWARD")
    st.write("This is the content of the first application.")
    # Add your specific content and functionality here.
    st.slider("Select a number", 0, 100, key="N")