import streamlit as st
import pandas as pd
from transformers import AutoTokenizer
import streamlit.components.v1 as components
import random

message = "Xin chào! Tôi là trợ lý AI giúp bạn hiểu metadata hình ảnh. Hỏi tôi bất cứ điều gì!"

system_prompt = """
    You are an advanced AI financial advisor with extensive market knowledge and analytical capabilities. Your name is CRobo Advisor. The current date is July 2024. Your role is to assist traders and investors with stock analysis, market insights, and trading strategies. Please adhere to the following guidelines:
    1. Expertise: Demonstrate deep understanding of financial markets, trading strategies, technical analysis, and market structures.
    2. Analysis: Provide thorough, data-driven analysis of market trends, specific stocks, or trading strategies as requested.
    3. Explanations: Offer clear, comprehensive explanations suitable for traders of all experience levels. Break down complex concepts when necessary.
    4. Asset Evaluation: When asked, use your knowledge to identify potential trading assets. Provide a detailed list with supporting data and rationale for each recommendation.
    5. Current Information: Ensure all advice and analysis is based on the most up-to-date market information available to you. If you need to access real-time data, inform the user and proceed to retrieve the latest information.
    6. Honesty: If you're unsure about something or don't have the necessary information, clearly state this. Do not provide speculative or potentially misleading information.
    7. Language: Provide all responses in Vietnamese.
    8. Adaptability: Tailor your responses to the specific needs and questions of each user, whether they're seeking general market insights or detailed analysis of particular stocks or strategies.
    9. Markdown Format: Provide all responses in Markdown format, highlighting key points using bold text.
    """

def init_home_state(_message): 
  if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": message}]
  if "exif_df" not in st.session_state:
      st.session_state["exif_df"] = pd.DataFrame()
  if "url_exif_df" not in st.session_state:
      st.session_state["url_exif_df"] = pd.DataFrame()
  if "show_expanders" not in st.session_state:
      st.session_state.show_expanders = True
  if "reset_trigger" not in st.session_state:
      st.session_state.reset_trigger = False
  if "image_url" not in st.session_state:
      st.session_state["image_url"] = ""
  if "follow_up" not in st.session_state:
      st.session_state.follow_up = False
  if "show_animation" not in st.session_state:
      st.session_state.show_animation = True


def init_stock_advisor(): 
  if "messages" not in st.session_state:
      st.session_state["messages"] = [{"role": "assistant", "content": message}]
  if 'generated' not in st.session_state:
      st.session_state['generated'] = []
  if 'past' not in st.session_state:
      st.session_state['past'] = []
  if 'model_name' not in st.session_state:
      st.session_state['model_name'] = []
  if 'cost' not in st.session_state:
      st.session_state['cost'] = []
  if 'total_tokens' not in st.session_state:
      st.session_state['total_tokens'] = []
  if 'total_cost' not in st.session_state:
      st.session_state['total_cost'] = 0.0  
                

def clear_chat_history():
    st.session_state.reset_trigger = not st.session_state.reset_trigger
    st.session_state.show_expanders = True
    st.session_state.show_animation = True
    st.session_state.messages = [{"role": "assistant", "content": message}]
    st.session_state["exif_df"] = pd.DataFrame()
    st.session_state["url_exif_df"] = pd.DataFrame()
    st.cache_data.clear()
    st.success("Chat History Cleared!")

def clear_stock_advisor():
    st.session_state.show_animation = True
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state.messages = [{"role": "assistant", "content": message}]
    st.session_state['number_tokens'] = []
    st.session_state['model_name'] = []
    st.session_state['cost'] = []
    st.session_state['total_cost'] = 0.0
    st.session_state['total_tokens'] = []


@st.cache_resource(show_spinner=False)
def get_tokenizer():
    return AutoTokenizer.from_pretrained("huggyllama/llama-7b")

def get_num_tokens(prompt):
    tokenizer = get_tokenizer()
    tokens = tokenizer.tokenize(prompt)
    return len(tokens)


