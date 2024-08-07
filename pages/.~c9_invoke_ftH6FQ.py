from langchain_community.retrievers import AmazonKnowledgeBasesRetriever
import streamlit as st
from streamlit_chat import message
import boto3
import json
from anthropic import Anthropic
import base

# Initialize the Anthropic client
anthropic = Anthropic()

# Constants
KNOWLEDGE_BASE_ID = 'F2QMSWL0AX'
MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
BEDROCK_SERVICE_NAME = "bedrock-runtime"
MAX_TOKENS = 2000
TEMPERATURE = 0.7
TOP_P = 1

# Set up Streamlit page configuration
st.set_page_config(page_title="Document Summary Assistant", layout="wide")
st.title('Document Summary Assistant')

# Function to count tokens
def count_tokens(text):
    return len(anthropic.get_tokenizer().encode(text))

# Initialize home state
base.init_home_state("Your AI document summary assistant")

# Function to generate responses
def generate_response(prompt):
    try:
        # Initialize the Bedrock client
        bedrock = boto3.client(service_name=BEDROCK_SERVICE_NAME)
        
        # Initialize the retriever
        retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            top_k=5,
            retrieval_config={
                "vectorSearchConfiguration": {
                    "numberOfResults": 5,
                    'overrideSearchType': "SEMANTIC"
                }
            }
        )
        
        # Retrieve relevant documents
        retrieved_docs = retriever.get_relevant_documents(prompt)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        conversation_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.get('messages', [])])
        
        query = f"""Human: {base.system_prompt}. Based on the provided context, provide the answer to the following question:
        <context>{context}</context>
        <conversation_history>{conversation_history}</conversation_history>
        <question>{prompt}</question>
        Assistant: 
        """
        
        prompt_config = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": MAX_TOKENS,
            "messages": [
                {"role": "user", "content": query}
            ],
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
        }
        
        # Invoke the model
        response = bedrock.invoke_model_with_response_stream(
            body=json.dumps(prompt_config),
            modelId=MODEL_ID,
            accept="application/json", 
            contentType="application/json"
        )
        
        stream = response.get('body')
        if stream:
            for event in stream:
                chunk = event.get('chunk')
                if chunk:
                    chunk_obj = json.loads(chunk.get('bytes').decode())
                    if 'delta' in chunk_obj:
                        delta_obj = chunk_obj.get('delta', None)
                        if delta_obj:
                            text = delta_obj.get('text', None)
                            if text:
                                yield text
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Initialize message history if not present
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# User input handling
prompt = st.text_input("Enter your question:", "")
if prompt:
    st.session_state['messages'].append({"role": "user", "content": prompt})
    response = generate_response(prompt)
    full_response = "".join(response)
    st.session_state['messages'].append({"role": "assistant", "content": full_response})

# Display conversation history
for msg in st.session_state['messages']:
    message(msg['content'], is_user=msg['role'] == 'user')

# Optional: Add stock analysis functionality
def analyze_stock(ticker):
    # Example function to analyze stock data (dummy implementation)
    # Replace with actual stock analysis logic
    return f"Analysis for {ticker}: Stock price is currently XYZ."

# Add stock analysis feature
ticker = st.text_input("Enter stock ticker for analysis:", "")
if ticker:
    analysis_result = analyze_stock(ticker)
    st.write(analysis_result)
