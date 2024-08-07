import streamlit as st
from PyPDF2 import PdfReader
import libs as glib

# Set page configuration
st.set_page_config(page_title="Document Q&A", layout="wide")

# Title of the app
st.title('Document Q&A')

# File uploader for PDF documents
uploaded_file = st.file_uploader("Upload a PDF document for Q&A", type="pdf")

# Text input for user's question
input_text = st.text_input("Your question!")

# Initialize an empty list to store the document text
docs = []

# Check if a file is uploaded and a question is entered
if uploaded_file is not None and input_text:
    # Read the uploaded PDF file
    reader = PdfReader(uploaded_file)
    
    # Extract text from each page of the PDF and add it to the docs list
    for page in reader.pages:
        docs.append(page.extract_text())
    
    # Query the document using the user's question and the extracted text
    response = glib.query_document(input_text, docs)
    
    # Display the response in the Streamlit app
    st.write(response)
