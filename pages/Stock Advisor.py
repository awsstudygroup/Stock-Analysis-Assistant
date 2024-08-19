import streamlit as st
from streamlit_chat import message
import boto3
import json
from anthropic import Anthropic
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever

# Initialize the Anthropic client
anthropic = Anthropic()

# Constants
KNOWLEDGE_BASE_ID = 'ZWPXJNPCQL'
MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
BEDROCK_SERVICE_NAME = "bedrock-runtime"
MAX_TOKENS = 10000
TEMPERATURE = 0.5
TOP_P = 0.9

# Streamlit page configuration

st.title(":rainbow[AWS Stock Advisor]")

# Custom CSS for improved UI
with open('./style.css') as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Sidebar with instructions
with st.sidebar:
    st.markdown("""
        :rainbow[**Ask a question**]: Enter your stock-related query in the input box.
        
        :rainbow[**Get Advice**]: Receive expert advice based on real-time data and financial models.
        
        :rainbow[**Review History**]: Scroll through the conversation to review previous advice.
    """)

# Function to count tokens
def count_tokens(text):
    return len(anthropic.get_tokenizer().encode(text))

# Initialize conversation history
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "assistant", "content": "Welcome to Stock Advisor! I am your professional Stock Advisor, ready to assist you with your stock inquiries. How can I help you today?"}
    ]

# Function to generate AI responses
def generate_response(prompt):
    try:
        # Initialize the Bedrock client
        bedrock = boto3.client(service_name=BEDROCK_SERVICE_NAME)
        
        # Retrieve relevant documents from the knowledge base
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
        retrieved_docs = retriever.get_relevant_documents(prompt)
        context = "\n".join(doc.page_content for doc in retrieved_docs)
        
        # Compile conversation history
        conversation_history = "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state['messages'])
        
        # Formulate the AI model query with a professional tone, technical analysis, and charting
        query = f"""
        Human: As a seasoned Stock Advisor, you are tasked with providing expert stock analysis using the following technical indicators:
        - Moving Average (MA)
        - Relative Strength Index (RSI)
        - Moving Average Convergence Divergence (MACD)
        - Bollinger Bands

        For each indicator, please provide a detailed analysis, and plot the corresponding charts to visualize these indicators based on the recent data. After the analysis, give a comprehensive recommendation on the stock's performance and possible strategies.

        <context>{context}</context>
        <conversation_history>{conversation_history}</conversation_history>
        <question>{prompt}</question>
        Advisor:
        """
        
        # AI model prompt configuration
        prompt_config = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": query}],
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
        }
        
        # Invoke the model and get the response
        response = bedrock.invoke_model_with_response_stream(
            body=json.dumps(prompt_config),
            modelId=MODEL_ID,
            accept="application/json", 
            contentType="application/json"
        )
        
        # Process the streamed response
        full_response = ""
        stream = response.get('body')
        if stream:
            for event in stream:
                chunk = event.get('chunk')
                if chunk:
                    chunk_obj = json.loads(chunk.get('bytes').decode())
                    delta_obj = chunk_obj.get('delta', None)
                    if delta_obj:
                        text = delta_obj.get('text', None)
                        if text:
                            full_response += text
        return full_response

    except boto3.exceptions.Boto3Error as e:
        st.error(f"An error occurred with Boto3: {e}")
        return "Sorry, something went wrong. Please try again later."
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return "Sorry, something went wrong. Please try again later."

# Handle user input
prompt = st.text_input(":rainbow[Enter your stock-related question or ask for advice:]", "")
if prompt:
    st.session_state['messages'].append({"role": "user", "content": prompt})
    response = generate_response(prompt)
    st.session_state['messages'].append({"role": "assistant", "content": response})

# Display conversation history with styled message bubbles
for msg in st.session_state['messages']:
    role_class = "message-user" if msg['role'] == 'user' else "message-assistant"
    st.markdown(f'<div class="{role_class}">{msg["content"]}</div>', unsafe_allow_html=True)
