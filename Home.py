import streamlit as st
import libs as glib
import json

# Set up the page configuration
st.set_page_config(page_title="Home", layout="centered", initial_sidebar_state="collapsed")

# Title and description
st.title("Chatbot Q&A")
st.write("Welcome to our AI-powered chat interface. Type your query below and get instant responses!")

# Chat input
input_text = st.chat_input("Enter your message here...")
if input_text:
    try:
        # Call the custom library function to get a response
        response = glib.call_claude_sonet_stream(input_text)
        # Stream the response
        st.write_stream(response)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please try again later.")
