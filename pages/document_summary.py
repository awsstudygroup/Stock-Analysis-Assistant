import streamlit as st 
from PyPDF2 import PdfReader
import base
import libs as glib 

# Set the page configuration
st.set_page_config(page_title="Document Summary", layout="wide")

# Title of the application
st.title('Document Summary')

# File uploader for PDF documents
uploaded_file = st.file_uploader("Upload a PDF document")

# Option to select the summary length
summary_length = st.selectbox("Select summary length", options=["Short", "Medium", "Long"])

# Option to choose the format of the summary output
summary_format = st.selectbox("Select summary format", options=["Text", "Bullet Points"])

# Initialize an empty list to store the document text
docs = []

# Check if a file has been uploaded
if uploaded_file is not None:
    # Create a PDF reader object
    reader = PdfReader(uploaded_file)
    
    # Extract text from each page and append to the docs list
    for page in reader.pages:
        docs.append(page.extract_text())

    # Generate the summary using the summary_stream function
    try:
        # Attempt to call the summary_stream function with additional options
        response = glib.summary_stream(docs, summary_length, summary_format)
    except TypeError:
        # Fallback if the summary_stream function does not accept additional options
        response = glib.summary_stream(docs)

    # Convert the generator response to a list of strings
    response_list = list(response)

    # Display the summary in the selected format
    if summary_format == "Text":
        st.write(" ".join(response_list))
    elif summary_format == "Bullet Points":
        st.write('\n'.join(f"- {point}" for point in response_list))

# Display the summary length and format selected
st.write(f"Summary Length: {summary_length}")
st.write(f"Summary Format: {summary_format}")
