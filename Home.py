import streamlit as st
import libs as glib
import json
import traceback

# Set up the page configuration
st.set_page_config(page_title="AWS Stock Analysis Assistant", layout="wide")

# Load CSS for styling
try:
    with open('./style.css') as f:
        css = f.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.error("CSS file not found. Please ensure 'style.css' is present in the correct directory.")

# Title and description
st.title(":rainbow[AWS Stock Analysis Assistant]")
st.write(":rainbow[Welcome to the Stock Analysis Assistant. Enter your stock-related queries below and receive instant insights!]")

# Chat input
input_text = st.text_input(":rainbow[Enter your query here...]")

# Process the input
if input_text:
    try:
        # Call the custom library function to get a response (assuming it's a generator)
        response_generator = glib.call_claude_sonet_stream(input_text)
        
        # Handle None values in the generator and join the response
        response = "".join(item if item is not None else "" for item in response_generator)
        
        # Display the response
        if response:
            st.write(response)
        else:
            st.warning("No response generated. Please try a different query.")
    
    except Exception as e:
        # Print full error details for debugging
        st.error(f"An error occurred: {str(e)}")
        st.write("Please try again later.")
        st.write(traceback.format_exc())

        # Optional: Log the error to a file for further investigation
        with open('error_log.txt', 'a') as f:
            f.write(traceback.format_exc() + "\n")
