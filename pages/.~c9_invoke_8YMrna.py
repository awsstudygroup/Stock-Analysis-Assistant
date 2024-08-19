import streamlit as st
import boto3
from langchain_aws import ChatBedrock

import json
from anthropic import Anthropic
from datetime import date, datetime, timedelta
import pandas as pd
import os
from vnstock3 import Vnstock
from bs4 import BeautifulSoup
import re
import requests
from langchain.agents import initialize_agent, Tool
from langchain.callbacks import StreamlitCallbackHandler
import base

# Setting page title and header
st.set_page_config(page_title="Trợ lý chứng khoán", page_icon="img/favicon.ico", layout="wide")
st.title('Trợ lý chứng khoán')
base.init_home_state("Your 24/7 AI financial companion")

anthropic = Anthropic()
knowledge_base_id = "DTINVPCWDO"

modelId = "anthropic.claude-3-5-sonnet-20240620-v1:0"

clear_button = st.sidebar.button("Xoá lịch sử chat", key="clear")
if clear_button:
    base.clear_chat_history()

def get_llm():
    model_kwargs = {
        "max_tokens": 2048,
        "temperature": 0.2,
        "top_k": 250,
        "top_p": 1,
    }
    bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name=os.environ.get("BWB_REGION_NAME"))
    llm = ChatBedrock(
        client=bedrock_runtime,
        model_id=modelId,
        model_kwargs=model_kwargs,
        streaming=True
    )
    return llm

# Function to parse and validate response
def parse_response(content):
    try:
        parsed_data = json.loads(content)
        company_name = parsed_data[0].get('company_name', 'Unknown')
        company_ticker = parsed_data[0].get('company_ticker', 'Unknown')
        return company_name, company_ticker
    except (json.JSONDecodeError, KeyError):
        return 'Unknown', 'Unknown'

# Function to invoke Bedrock model
def invoke_bedrock_model(prompt, max_tokens=2000):
    bedrock_runtime = boto3.client('bedrock-runtime')
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 1,
    }

    response = bedrock_runtime.invoke_model(
        modelId=modelId,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(request_body)
    )
    response_body = json.loads(response['body'].read().decode())
    return response_body['content'][0]['text']

def get_stock_ticker(question):
    with open('tickers.csv', 'r') as file:
        company_data = file.read()

    prompt = f"""Your AI Assistant prompt with the full instructions and context..."""
    initial_response = invoke_bedrock_model(prompt)
    company_name, company_ticker = parse_response(initial_response)

    with open("company.json", 'w') as file:
        file.write(json.dumps({'company_name': company_name, 'company_ticker': company_ticker}))

    return company_name, company_ticker

# get stock history of the company in 3 years
def get_stock_price(ticker, history=1000):
    with open("company.json", 'a') as file:
        file.write(f'\nget stock price for ticker: {ticker}')
    today = date.today()
    start_date = today - timedelta(days=history)
    stock = Vnstock().stock(symbol=ticker.strip(), source='TCBS')
    data = stock.quote.history(start=start_date.strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'))
    return data

# Function to safely get data and handle exceptions
def safe_get_data(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"Error getting data: {e}")
        return pd.DataFrame()

# Function to get financial data
def get_financial_data(ticker):
    ticker = ticker.strip().upper()
    stock_finance = Vnstock().stock(symbol=ticker, source='VCI')

    company_data = {
        'Balance Sheet Yearly': safe_get_data(stock_finance.finance.balance_sheet, period='year', lang='en'),
        'Balance Sheet Quarterly': safe_get_data(stock_finance.finance.balance_sheet, period='quarter', lang='en'),
        'Income Statement Yearly': safe_get_data(stock_finance.finance.income_statement, period='year', lang='en'),
        'Income Statement Quarterly': safe_get_data(stock_finance.finance.income_statement, period='quarter', lang='en'),
        'Cash Flow Yearly': safe_get_data(stock_finance.finance.cash_flow, period='year', lang='en'),
        'Cash Flow Quarterly': safe_get_data(stock_finance.finance.cash_flow, period='quarter', lang='en'),
        'Financial Ratios Yearly': safe_get_data(stock_finance.finance.ratio, period='year', lang='en'),
        'Financial Ratios Quarterly': safe_get_data(stock_finance.finance.ratio, period='quarter', lang='en'),
    }
    return company_data

def get_financial_statements(ticker):
    stock = Vnstock().stock(symbol=ticker.upper(), source='VCI')
    data = stock.finance.balance_sheet(period='year', lang='en')
    return data

# Script to scrap top 5 Google news for a given company name
def google_query(search_term):
    if "news" not in search_term:
        search_term = search_term + " stock news"
    url = f"https://www.google.com/search?q={search_term}+tin+tức&hl=vi&tbm=nws"
    url = re.sub(r"\s", "+", url)
    return url

def get_recent_news(ticker):
    ticker = ticker.strip().upper()

    headers = {'User-Agent': 'Mozilla/5.0'}
    company = Vnstock().stock(symbol=ticker.upper(), source='TCBS').company
    company_name = company.profile()['company_name'][0]
    g_query = google_query(company_name)
    res = requests.get(g_query, headers=headers).text
    soup = BeautifulSoup(res, "html.parser")
    news = [n.text for n in soup.find_all("div", class_="n0jPhd ynAwRc tNxQIb nDgy9d")]
    news.extend([n.text for n in soup.find_all("div", class_="IJl0Z")])

    news = news[:4] if len(news) > 6 else news
    news_string = "\n".join([f"{i}. {n}" for i, n in enumerate(news)])
    
    return "Recent News:\n\n" + news_string

def get_article_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([p.text for p in paragraphs])
        return content
    except Exception as e:
        return f"Could not retrieve content: {e}"

def initializeAgent():
    zero_shot_agent = initialize_agent(
        llm=get_llm(),
        agent="zero-shot-react-description",
        tools=tools,
        verbose=True,
        max_iteration=1,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        output_key="output",
    )
    CURRENT_DATE = datetime.today().strftime("%Y-%m-%d")
    prompt = f"""
        Your agent prompt with the full instructions...
    """

    zero_shot_agent.agent.llm_chain.prompt.template = prompt 
    return zero_shot_agent

tools = [
    Tool(
        name="get company ticker",
        func=get_stock_ticker,
        description="Extract company name and ticker from the user's question. Input: user question. Output: company name and ticker."
    ),
    Tool(
        name="get stock data",
        func=get_stock_price,
        description="Retrieve historical share price data for stock analysis. Input: EXACT company ticker. Output: historic share price data."
    ),
    Tool(
        name="get recent stock news",
        func=get_recent_stock_news,
        description="Fetch recent news about the stock. Input: EXACT company ticker. Output: recent news articles related to the stock."
    ),
    Tool(
        name="get financial data",
        func=get_financial_data,
        description="""Retrieve comprehensive financial data for a company. Input: EXACT company ticker. 
                Output: A dictionary containing financial reports..."""
    )
]

zero_shot_agent = initializeAgent()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Main interaction function
def generate_response(prompt, st_callback):
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    st.session_state['messages'].append({"role": "user", "content": prompt})

    response = zero_shot_agent({
        "input": prompt,
        "chat_history": st.session_state.chat_history,
    },
    callbacks=[st_callback])

    return response.get('body'), prompt

# Containers
response_container = st.container()
input_container = st.container()

with input_container:
    user_input = st.text_area("You:", key='input', height=100)
    submit_button = st.button(label='Send')

    if submit_button and user_input:
        st_callback = StreamlitCallbackHandler(st.container())
        response_stream, query = generate_response(user_input, st_callback)

with st.chat_input():
    if prompt := st.chat_input("Send a message..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        base.right_message(st, prompt)

        st_callback = StreamlitCallbackHandler()
        response = generate_response(prompt, st_callback)
        full_response = st.write_stream(response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
