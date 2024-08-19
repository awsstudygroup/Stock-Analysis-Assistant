import pandas as pd
import yfinance as yf
import streamlit as st
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import boto3, json
import pandas_datareader as pdr
import numpy as np

# st.set_page_config(page_title="Phân tích cổ phiếu", page_icon="img/favicon.ico", layout="wide")
st.title(":rainbow[Phân tích cổ phiếu]")

# Custom CSS for improved UI
with open('./style.css') as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

snp500 = pd.read_csv("SP500.csv")
symbols = snp500['Symbol'].sort_values().tolist()

ticker = st.sidebar.selectbox('Chọn mã chứng khoán', symbols)
infoType = st.sidebar.radio("Chọn kiểu phân tích", ('Phân tích kỹ thuật', 'Phân tích cơ bản'))

def get_stock_price(ticker, history=500):
    today = dt.datetime.today()
    start_date = today - dt.timedelta(days=history)
    df_price = pdr.get_data_yahoo(ticker, start=start_date, end=today)
    return df_price

def plot_stock_price(ticker, history=500):
    df_price = get_stock_price(ticker, history)
    fig = go.Figure([go.Scatter(x=df_price.index, y=df_price['Adj Close'], mode='lines', name='Adjusted Close')])
    fig.update_layout(
        title=f':rainbow[Stock Price for {ticker}]',
        xaxis_title=':rainbow[Date]',
        yaxis_title=':rainbow[Price ($)]',
        yaxis_tickprefix='$',
        hovermode="x"
    )
    fig.update_yaxes(tickprefix="$")
    st.plotly_chart(fig, use_container_width=True)

def call_claude_sonet_stream(prompt):
    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 10000,
        "temperature": 0.5,
        "top_k": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }

    body = json.dumps(prompt_config)

    modelId = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    accept = "application/json"
    contentType = "application/json"

    bedrock = boto3.client(service_name="bedrock-runtime")
    response = bedrock.invoke_model_with_response_stream(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )

    stream = response['body']
    full_response = ""
    if stream:
        for event in stream:
            chunk = event.get('chunk')
            if chunk:
                delta = json.loads(chunk.get('bytes').decode()).get("delta")
                if delta:
                    full_response += delta.get("text", "")
    return full_response

def forecast_price(question, docs):
    prompt = """Human: here is the data price:
        <text>""" + str(docs) + """</text>
        Question: """ + question + """ 
    \n\nAssistant: """
    return call_claude_sonet_stream(prompt)

def calcMovingAverage(data, size):
    df = data.copy()
    df['sma'] = df['Adj Close'].rolling(size).mean()
    df['ema'] = df['Adj Close'].ewm(span=size, min_periods=size).mean()
    df.dropna(inplace=True)
    return df

def calc_macd(data):
    df = data.copy()
    df['ema12'] = df['Adj Close'].ewm(span=12, min_periods=12).mean()
    df['ema26'] = df['Adj Close'].ewm(span=26, min_periods=26).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['signal'] = df['macd'].ewm(span=9, min_periods=9).mean()
    df.dropna(inplace=True)
    return df

def calcBollinger(data, size):
    df = data.copy()
    df["sma"] = df['Adj Close'].rolling(size).mean()
    df["bolu"] = df["sma"] + df['Adj Close'].rolling(size).std(ddof=0)
    df["bold"] = df["sma"] - df['Adj Close'].rolling(size).std(ddof=0)
    df["width"] = df["bolu"] - df["bold"]
    df.dropna(inplace=True)
    return df

def calc_rsi(data, window):
    df = data.copy()
    delta = df['Adj Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df.dropna(inplace=True)
    return df

def calc_atr(data, window):
    df = data.copy()
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Adj Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Adj Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=window).mean()
    df.dropna(inplace=True)
    return df

# New Technical Analysis Functions

def calc_stochastic(data, k_period=14, d_period=3):
    df = data.copy()
    df['L14'] = df['Low'].rolling(window=k_period).min()
    df['H14'] = df['High'].rolling(window=k_period).max()
    df['%K'] = 100 * ((df['Adj Close'] - df['L14']) / (df['H14'] - df['L14']))
    df['%D'] = df['%K'].rolling(window=d_period).mean()
    df.dropna(inplace=True)
    return df

def calc_cci(data, window=20):
    df = data.copy()
    df['TP'] = (df['High'] + df['Low'] + df['Adj Close']) / 3
    df['sma'] = df['TP'].rolling(window=window).mean()
    df['mad'] = df['TP'].rolling(window=window).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    df['CCI'] = (df['TP'] - df['sma']) / (0.015 * df['mad'])
    df.dropna(inplace=True)
    return df

def calc_obv(data):
    df = data.copy()
    df['OBV'] = 0
    df.loc[df['Adj Close'] > df['Adj Close'].shift(1), 'OBV'] = df['Volume']
    df.loc[df['Adj Close'] < df['Adj Close'].shift(1), 'OBV'] = -df['Volume']
    df['OBV'] = df['OBV'].cumsum()
    df.dropna(inplace=True)
    return df

def calc_williams_r(data, window=14):
    df = data.copy()
    df['highest_high'] = df['High'].rolling(window=window).max()
    df['lowest_low'] = df['Low'].rolling(window=window).min()
    df['Williams %R'] = -100 * ((df['highest_high'] - df['Adj Close']) / (df['highest_high'] - df['lowest_low']))
    df.dropna(inplace=True)
    return df

def calc_ichimoku(data):
    df = data.copy()
    df['tenkan_sen'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
    df['kijun_sen'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
    df['senkou_span_b'] = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)
    df['chikou_span'] = df['Adj Close'].shift(-26)
    df.dropna(inplace=True)
    return df

def display_financials(ticker):
    stock = yf.Ticker(ticker)
    st.subheader(':rainbow[Báo cáo tài chính]')

    # Income statement
    st.markdown(':rainbow[### Báo cáo thu nhập]')
    income_statement = stock.financials.T
    st.write(income_statement)

    # Balance sheet
    st.markdown(':rainbow[### Bảng cân đối kế toán]')
    balance_sheet = stock.balance_sheet.T
    st.write(balance_sheet)

    # Cashflow statement
    st.markdown(':rainbow[### Báo cáo lưu chuyển tiền tệ]')
    cashflow = stock.cashflow.T
    st.write(cashflow)

def get_financial_ratios(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info

    financial_ratios = {
        ':rainbow[Tỷ lệ Nợ/Vốn chủ sở hữu]': info.get('debtToEquity', 'N/A'),
        ':rainbow[Tỷ lệ Lợi nhuận gộp (%)]': info.get('grossMargins', 'N/A'),
        ':rainbow[Tỷ lệ Lợi nhuận ròng (%)]': info.get('profitMargins', 'N/A'),
        ':rainbow[Tỷ lệ Thanh khoản nhanh]': info.get('quickRatio', 'N/A'),
        ':rainbow[Tỷ lệ Thanh khoản hiện hành]': info.get('currentRatio', 'N/A'),
        ':rainbow[Tỷ suất Sinh lời trên Tài sản (ROA)]': info.get('returnOnAssets', 'N/A'),
        ':rainbow[Tỷ suất Sinh lời trên Vốn chủ sở hữu (ROE)]': info.get('returnOnEquity', 'N/A')
    }

    financial_ratios_df = pd.DataFrame(financial_ratios, index=['Giá trị'])
    st.subheader(':rainbow[Chỉ số tài chính]')
    st.table(financial_ratios_df.T)

def get_valuation_ratios(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info

    valuation_ratios = {
        ':rainbow[EV/EBITDA]': info.get('enterpriseToEbitda', 'N/A'),
        ':rainbow[EV/Revenue]': info.get('enterpriseToRevenue', 'N/A'),
        ':rainbow[P/E so với ngành]': info.get('trailingPE', 'N/A')
    }

    valuation_ratios_df = pd.DataFrame(valuation_ratios, index=['Giá trị'])
    st.subheader(':rainbow[Chỉ số định giá]')
    st.table(valuation_ratios_df.T)

if infoType == 'Phân tích cơ bản':
    stock = yf.Ticker(ticker)
    info = stock.info

    st.subheader(':rainbow[Thông tin công ty]')
    st.markdown(':rainbow[**Tên công ty**]: ' + info.get('longName', 'N/A'))
    st.markdown(':rainbow[**Lĩnh vực**]: ' + info.get('sector', 'N/A'))
    st.markdown(':rainbow[**Ngành**]: ' + info.get('industry', 'N/A'))
    st.markdown(':rainbow[**Liên hệ**]: ' + info.get('phone', 'N/A'))
    st.markdown(':rainbow[**Địa chỉ**]: ' + ', '.join(filter(None, [info.get('address1', ''), info.get('city', ''), info.get('zip', ''), info.get('country', '')])))
    st.markdown(':rainbow[**Website**]: ' + info.get('website', 'N/A'))

    st.subheader(':rainbow[Tóm tắt về công ty]')
    response = call_claude_sonet_stream(f"Giới thiệu công ty: {info.get('longBusinessSummary', '')}")
    st.write(response)

    st.subheader(':rainbow[Thông tin cơ bản]')
    fund_info = {
        ':rainbow[Giá trị doanh nghiệp (USD)]': info.get('enterpriseValue', 'N/A'),
        ':rainbow[Tỷ lệ giá trị doanh nghiệp trên doanh thu]': info.get('enterpriseToRevenue', 'N/A'),
        ':rainbow[Tỷ lệ giá trị doanh nghiệp trên EBITDA]': info.get('enterpriseToEbitda', 'N/A'),
        ':rainbow[Thu nhập ròng (USD)]': info.get('netIncomeToCommon', 'N/A'),
        ':rainbow[Tỷ suất lợi nhuận]': info.get('profitMargins', 'N/A'),
        ':rainbow[Tỷ số P/E dự phóng]': info.get('forwardPE', 'N/A'),
        ':rainbow[Tỷ số PEG]': info.get('pegRatio', 'N/A'),
        ':rainbow[Tỷ số giá trên giá trị sổ sách]': info.get('priceToBook', 'N/A'),
        ':rainbow[EPS dự phóng (USD)]': info.get('forwardEps', 'N/A'),
        ':rainbow[Hệ số Beta]': info.get('beta', 'N/A'),
        ':rainbow[Giá trị sổ sách (USD)]': info.get('bookValue', 'N/A'),
        ':rainbow[Tỷ lệ cổ tức (%)]': info.get('dividendRate', 'N/A'),
        ':rainbow[Lợi suất cổ tức (%)]': info.get('dividendYield', 'N/A'),
        ':rainbow[Lợi suất cổ tức trung bình 5 năm (%)]': info.get('fiveYearAvgDividendYield', 'N/A'),
        ':rainbow[Tỷ lệ chi trả cổ tức]': info.get('payoutRatio', 'N/A')
    }
    fund_df = pd.DataFrame(fund_info, index=['Giá trị'])
    st.table(fund_df.T)

    st.subheader(':rainbow[Thông tin chung về cổ phiếu]')
    st.markdown(':rainbow[**Thị trường**]: ' + info.get('market', 'N/A'))
    st.markdown(':rainbow[**Sàn giao dịch**]: ' + info.get('exchange', 'N/A'))
    st.markdown(':rainbow[**Loại báo giá**]: ' + info.get('quoteType', 'N/A'))

    start = dt.datetime.today() - dt.timedelta(365)
    end = dt.datetime.today()
    df = yf.download(ticker, start, end)
    df = df.reset_index()

    fig = go.Figure(data=go.Scatter(x=df['Date'], y=df['Adj Close'], mode='lines'))
    fig.update_layout(
        title=":rainbow[Giá trị cổ phiếu trong vòng 2 năm]",
        xaxis_title=":rainbow[Ngày]",
        yaxis_title=":rainbow[Giá ($)]",
        yaxis_tickprefix="$",
        title_x=0.5
    )
    st.plotly_chart(fig, use_container_width=True)

    market_info = {
        ":rainbow[Khối lượng giao dịch]": info.get('volume', 'N/A'),
        ":rainbow[Khối lượng giao dịch trung bình]": info.get('averageVolume', 'N/A'),
        ":rainbow[Vốn hóa thị trường]": info.get('marketCap', 'N/A'),
        ":rainbow[Số cổ phiếu lưu hành tự do]": info.get('floatShares', 'N/A'),
        ":rainbow[Giá thị trường thông thường (USD)]": info.get('regularMarketPrice', 'N/A'),
        ':rainbow[Khối lượng đặt mua]': info.get('bidSize', 'N/A'),
        ':rainbow[Khối lượng đặt bán]': info.get('askSize', 'N/A'),
        ":rainbow[Số cổ phiếu bán khống]": info.get('sharesShort', 'N/A'),
        ':rainbow[Tỷ lệ bán khống]': info.get('shortRatio', 'N/A'),
        ':rainbow[Tổng số cổ phiếu đang lưu hành]': info.get('sharesOutstanding', 'N/A')
    }
    market_df = pd.DataFrame(market_info, index=['Giá trị'])
    st.table(market_df.T)

    # Thêm báo cáo tài chính
    display_financials(ticker)

    # Thêm chỉ số tài chính
    get_financial_ratios(ticker)

    # Thêm chỉ số định giá
    get_valuation_ratios(ticker)

    st.subheader(':rainbow[Dự đoán theo chỉ số cơ bản]')

    # Dự đoán Tỷ lệ Nợ/Vốn chủ sở hữu
    with st.expander(':rainbow[Dự đoán với Tỷ lệ Nợ/Vốn chủ sở hữu]'):
        debt_to_equity = info.get('debtToEquity', 'N/A')
        response_debt_to_equity = forecast_price(
            question=f"Tỷ lệ Nợ/Vốn chủ sở hữu hiện tại là {debt_to_equity}. Dựa vào chỉ số này, dự đoán tình hình tài chính của công ty trong tương lai và khuyến nghị về đầu tư.",
            docs=debt_to_equity
        )
        st.write(response_debt_to_equity)

    # Dự đoán Tỷ lệ Lợi nhuận gộp (%)
    with st.expander(':rainbow[Dự đoán với Tỷ lệ Lợi nhuận gộp (%)]'):
        gross_margins = info.get('grossMargins', 'N/A')
        response_gross_margins = forecast_price(
            question=f"Tỷ lệ Lợi nhuận gộp hiện tại là {gross_margins}. Dựa vào chỉ số này, dự đoán khả năng sinh lời của công ty trong tương lai và khuyến nghị về đầu tư.",
            docs=gross_margins
        )
        st.write(response_gross_margins)

    # Dự đoán Tỷ lệ Lợi nhuận ròng (%)
    with st.expander(':rainbow[Dự đoán với Tỷ lệ Lợi nhuận ròng (%)]'):
        profit_margins = info.get('profitMargins', 'N/A')
        response_profit_margins = forecast_price(
            question=f"Tỷ lệ Lợi nhuận ròng hiện tại là {profit_margins}. Dựa vào chỉ số này, dự đoán khả năng sinh lời của công ty trong tương lai và khuyến nghị về đầu tư.",
            docs=profit_margins
        )
        st.write(response_profit_margins)

    # Dự đoán Tỷ lệ Thanh khoản nhanh
    with st.expander(':rainbow[Dự đoán với Tỷ lệ Thanh khoản nhanh]'):
        quick_ratio = info.get('quickRatio', 'N/A')
        response_quick_ratio = forecast_price(
            question=f"Tỷ lệ Thanh khoản nhanh hiện tại là {quick_ratio}. Dựa vào chỉ số này, dự đoán khả năng thanh khoản của công ty trong tương lai và khuyến nghị về đầu tư.",
            docs=quick_ratio
        )
        st.write(response_quick_ratio)

    # Dự đoán Tỷ lệ Thanh khoản hiện hành
    with st.expander(':rainbow[Dự đoán với Tỷ lệ Thanh khoản hiện hành]'):
        current_ratio = info.get('currentRatio', 'N/A')
        response_current_ratio = forecast_price(
            question=f"Tỷ lệ Thanh khoản hiện hành hiện tại là {current_ratio}. Dựa vào chỉ số này, dự đoán khả năng thanh khoản của công ty trong tương lai và khuyến nghị về đầu tư.",
            docs=current_ratio
        )
        st.write(response_current_ratio)

    # Dự đoán Tỷ suất Sinh lời trên Tài sản (ROA)
    with st.expander(':rainbow[Dự đoán với Tỷ suất Sinh lời trên Tài sản (ROA)]'):
        return_on_assets = info.get('returnOnAssets', 'N/A')
        response_return_on_assets = forecast_price(
            question=f"Tỷ suất Sinh lời trên Tài sản (ROA) hiện tại là {return_on_assets}. Dựa vào chỉ số này, dự đoán khả năng sinh lời của công ty trong tương lai và khuyến nghị về đầu tư.",
            docs=return_on_assets
        )
        st.write(response_return_on_assets)

    # Dự đoán Tỷ suất Sinh lời trên Vốn chủ sở hữu (ROE)
    with st.expander(':rainbow[Dự đoán với Tỷ suất Sinh lời trên Vốn chủ sở hữu (ROE)]'):
        return_on_equity = info.get('returnOnEquity', 'N/A')
        response_return_on_equity = forecast_price(
            question=f"Tỷ suất Sinh lời trên Vốn chủ sở hữu (ROE) hiện tại là {return_on_equity}. Dựa vào chỉ số này, dự đoán khả năng sinh lời của công ty trong tương lai và khuyến nghị về đầu tư.",
            docs=return_on_equity
        )
        st.write(response_return_on_equity)

    # Dự đoán EV/EBITDA
    with st.expander(':rainbow[Dự đoán với EV/EBITDA]'):
        ev_to_ebitda = info.get('enterpriseToEbitda', 'N/A')
        response_ev_to_ebitda = forecast_price(
            question=f"EV/EBITDA hiện tại là {ev_to_ebitda}. Dựa vào chỉ số này, dự đoán khả năng định giá của công ty trong tương lai và khuyến nghị về đầu tư.",
            docs=ev_to_ebitda
        )
        st.write(response_ev_to_ebitda)

    # Dự đoán EV/Revenue
    with st.expander(':rainbow[Dự đoán với EV/Revenue]'):
        ev_to_revenue = info.get('enterpriseToRevenue', 'N/A')
        response_ev_to_revenue = forecast_price(
            question=f"EV/Revenue hiện tại là {ev_to_revenue}. Dựa vào chỉ số này, dự đoán khả năng định giá của công ty trong tương lai và khuyến nghị về đầu tư.",
            docs=ev_to_revenue
        )
        st.write(response_ev_to_revenue)

    # Dự đoán P/E so với ngành
    with st.expander(':rainbow[Dự đoán với P/E so với ngành]'):
        pe_ratio = info.get('trailingPE', 'N/A')
        response_pe_ratio = forecast_price(
            question=f"Tỷ số P/E hiện tại so với ngành là {pe_ratio}. Dựa vào chỉ số này, dự đoán khả năng định giá của công ty trong tương lai và khuyến nghị về đầu tư.",
            docs=pe_ratio
        )
        st.write(response_pe_ratio)

else:
    st.subheader(':rainbow[Phân tích kỹ thuật]')
    with st.expander(':rainbow[Moving Average]'):
        coMA1, coMA2 = st.columns(2)

        with coMA1:
            numYearMA = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key=0)

        with coMA2:
            windowSizeMA = st.number_input('Window Size (Day): ', min_value=5, max_value=500, value=20, key=1)

        start = dt.datetime.today() - dt.timedelta(numYearMA * 365)
        end = dt.datetime.today()
        dataMA = yf.download(ticker, start, end)
        df_ma = calcMovingAverage(dataMA, windowSizeMA)
        df_ma = df_ma.reset_index()

        figMA = go.Figure()

        figMA.add_trace(
            go.Scatter(
                x=df_ma['Date'],
                y=df_ma['Adj Close'],
                name=":rainbow[Prices Over Last " + str(numYearMA) + " Year(s)]"
            )
        )

        figMA.add_trace(
            go.Scatter(
                x=df_ma['Date'],
                y=df_ma['sma'],
                name=":rainbow[SMA" + str(windowSizeMA) + " Over Last " + str(numYearMA) + " Year(s)]"
            )
        )

        figMA.add_trace(
            go.Scatter(
                x=df_ma['Date'],
                y=df_ma['ema'],
                name=":rainbow[EMA" + str(windowSizeMA) + " Over Last " + str(numYearMA) + " Year(s)]"
            )
        )

        figMA.update_layout(
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            hovermode="x"
        )

        figMA.update_yaxes(tickprefix="$")

        st.plotly_chart(figMA, use_container_width=True)

    with st.expander(':rainbow[Relative Strength Index (RSI)]'):
        rsi_window = st.number_input('RSI Window Size (Day): ', min_value=5, max_value=500, value=14, key=8)
        df_rsi = calc_rsi(dataMA, rsi_window)
        df_rsi = df_rsi.reset_index()

        figRSI = go.Figure()
        figRSI.add_trace(
            go.Scatter(
                x=df_rsi['Date'],
                y=df_rsi['RSI'],
                name=":rainbow[RSI Over Last " + str(numYearMA) + " Year(s)]"
            )
        )
        figRSI.update_layout(
            yaxis=dict(range=[0, 100]),
            yaxis_title=":rainbow[RSI]",
            title=":rainbow[Relative Strength Index (RSI)]",
        )
        st.plotly_chart(figRSI, use_container_width=True)

    with st.expander(':rainbow[Moving Average Convergence Divergence (MACD)]'):
        numYearMACD = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key=2)

        startMACD = dt.datetime.today() - dt.timedelta(numYearMACD * 365)
        endMACD = dt.datetime.today()
        dataMACD = yf.download(ticker, startMACD, endMACD)
        df_macd = calc_macd(dataMACD)
        df_macd = df_macd.reset_index()

        figMACD = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01)

        figMACD.add_trace(
            go.Scatter(
                x=df_macd['Date'],
                y=df_macd['Adj Close'],
                name=":rainbow[Prices Over Last " + str(numYearMACD) + " Year(s)]"
            ),
            row=1, col=1
        )

        figMACD.add_trace(
            go.Scatter(
                x=df_macd['Date'],
                y=df_macd['ema12'],
                name=":rainbow[EMA 12 Over Last " + str(numYearMACD) + " Year(s)]"
            ),
            row=1, col=1
        )

        figMACD.add_trace(
            go.Scatter(
                x=df_macd['Date'],
                y=df_macd['ema26'],
                name=":rainbow[EMA 26 Over Last " + str(numYearMACD) + " Year(s)]"
            ),
            row=1, col=1
        )

        figMACD.add_trace(
            go.Scatter(
                x=df_macd['Date'],
                y=df_macd['macd'],
                name=":rainbow[MACD Line]"
            ),
            row=2, col=1
        )

        figMACD.add_trace(
            go.Scatter(
                x=df_macd['Date'],
                y=df_macd['signal'],
                name=":rainbow[Signal Line]"
            ),
            row=2, col=1
        )

        figMACD.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1,
            xanchor="left",
            x=0
        ))

        figMACD.update_yaxes(tickprefix="$")
        st.plotly_chart(figMACD, use_container_width=True)

    with st.expander(':rainbow[Bollinger Band]'):
        coBoll1, coBoll2 = st.columns(2)
        with coBoll1:
            numYearBoll = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key=6)

        with coBoll2:
            windowSizeBoll = st.number_input('Window Size (Day): ', min_value=5, max_value=500, value=20, key=7)

        startBoll = dt.datetime.today() - dt.timedelta(numYearBoll * 365)
        endBoll = dt.datetime.today()
        dataBoll = yf.download(ticker, startBoll, endBoll)
        df_boll = calcBollinger(dataBoll, windowSizeBoll)
        df_boll = df_boll.reset_index()
        figBoll = go.Figure()
        figBoll.add_trace(
            go.Scatter(
                x=df_boll['Date'],
                y=df_boll['bolu'],
                name=":rainbow[Upper Band]"
            )
        )

        figBoll.add_trace(
            go.Scatter(
                x=df_boll['Date'],
                y=df_boll['sma'],
                name=":rainbow[SMA" + str(windowSizeBoll) + " Over Last " + str(numYearBoll) + " Year(s)]"
            )
        )

        figBoll.add_trace(
            go.Scatter(
                x=df_boll['Date'],
                y=df_boll['bold'],
                name=":rainbow[Lower Band]"
            )
        )

        figBoll.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1,
            xanchor="left",
            x=0
        ))

        figBoll.update_yaxes(tickprefix="$")
        st.plotly_chart(figBoll, use_container_width=True)

    with st.expander(':rainbow[Average True Range (ATR)]'):
        atr_window = st.number_input('ATR Window Size (Day): ', min_value=5, max_value=500, value=14, key=9)
        df_atr = calc_atr(dataMA, atr_window)
        df_atr = df_atr.reset_index()

        figATR = go.Figure()
        figATR.add_trace(
            go.Scatter(
                x=df_atr['Date'],
                y=df_atr['ATR'],
                name=":rainbow[ATR Over Last " + str(numYearMA) + " Year(s)]"
            )
        )
        figATR.update_layout(
            yaxis_title=":rainbow[ATR]",
            title=":rainbow[Average True Range (ATR)]",
        )
        st.plotly_chart(figATR, use_container_width=True)

    with st.expander(':rainbow[Stochastic Oscillator (SO)]'):
        so_k_period = st.number_input('K Period (Day): ', min_value=5, max_value=500, value=14, key=10)
        so_d_period = st.number_input('D Period (Day): ', min_value=3, max_value=500, value=3, key=11)
        df_so = calc_stochastic(dataMA, so_k_period, so_d_period)
        df_so = df_so.reset_index()

        figSO = go.Figure()
        figSO.add_trace(
            go.Scatter(
                x=df_so['Date'],
                y=df_so['%K'],
                name=":rainbow[%K Over Last " + str(numYearMA) + " Year(s)]"
            )
        )
        figSO.add_trace(
            go.Scatter(
                x=df_so['Date'],
                y=df_so['%D'],
                name=":rainbow[%D Over Last " + str(numYearMA) + " Year(s)]"
            )
        )
        figSO.update_layout(
            yaxis_title=":rainbow[Stochastic Oscillator]",
            title=":rainbow[Stochastic Oscillator (SO)]",
        )
        st.plotly_chart(figSO, use_container_width=True)

    with st.expander(':rainbow[Commodity Channel Index (CCI)]'):
        cci_window = st.number_input('CCI Window Size (Day): ', min_value=5, max_value=500, value=20, key=12)
        df_cci = calc_cci(dataMA, cci_window)
        df_cci = df_cci.reset_index()

        figCCI = go.Figure()
        figCCI.add_trace(
            go.Scatter(
                x=df_cci['Date'],
                y=df_cci['CCI'],
                name=":rainbow[CCI Over Last " + str(numYearMA) + " Year(s)]"
            )
        )
        figCCI.update_layout(
            yaxis_title=":rainbow[CCI]",
            title=":rainbow[Commodity Channel Index (CCI)]",
        )
        st.plotly_chart(figCCI, use_container_width=True)

    with st.expander(':rainbow[On-Balance Volume (OBV)]'):
        df_obv = calc_obv(dataMA)
        df_obv = df_obv.reset_index()

        figOBV = go.Figure()
        figOBV.add_trace(
            go.Scatter(
                x=df_obv['Date'],
                y=df_obv['OBV'],
                name=":rainbow[OBV Over Last " + str(numYearMA) + " Year(s)]"
            )
        )
        figOBV.update_layout(
            yaxis_title=":rainbow[OBV]",
            title=":rainbow[On-Balance Volume (OBV)]",
        )
        st.plotly_chart(figOBV, use_container_width=True)

    with st.expander(':rainbow[Williams %R]'):
        williams_r_window = st.number_input('Williams %R Window Size (Day): ', min_value=5, max_value=500, value=14, key=13)
        df_williams_r = calc_williams_r(dataMA, williams_r_window)
        df_williams_r = df_williams_r.reset_index()

        figWilliamsR = go.Figure()
        figWilliamsR.add_trace(
            go.Scatter(
                x=df_williams_r['Date'],
                y=df_williams_r['Williams %R'],
                name=":rainbow[Williams %R Over Last " + str(numYearMA) + " Year(s)]"
            )
        )
        figWilliamsR.update_layout(
            yaxis_title=":rainbow[Williams %R]",
            title=":rainbow[Williams %R]",
        )
        st.plotly_chart(figWilliamsR, use_container_width=True)

    with st.expander(':rainbow[Ichimoku Cloud]'):
        df_ichimoku = calc_ichimoku(dataMA)
        df_ichimoku = df_ichimoku.reset_index()

        figIchimoku = go.Figure()
        figIchimoku.add_trace(
            go.Scatter(
                x=df_ichimoku['Date'],
                y=df_ichimoku['tenkan_sen'],
                name=":rainbow[Tenkan-sen]"
            )
        )
        figIchimoku.add_trace(
            go.Scatter(
                x=df_ichimoku['Date'],
                y=df_ichimoku['kijun_sen'],
                name=":rainbow[Kijun-sen]"
            )
        )
        figIchimoku.add_trace(
            go.Scatter(
                x=df_ichimoku['Date'],
                y=df_ichimoku['senkou_span_a'],
                name=":rainbow[Senkou Span A]"
            )
        )
        figIchimoku.add_trace(
            go.Scatter(
                x=df_ichimoku['Date'],
                y=df_ichimoku['senkou_span_b'],
                name=":rainbow[Senkou Span B]"
            )
        )
        figIchimoku.add_trace(
            go.Scatter(
                x=df_ichimoku['Date'],
                y=df_ichimoku['chikou_span'],
                name=":rainbow[Chikou Span]"
            )
        )
        figIchimoku.update_layout(
            yaxis_title=":rainbow[Price]",
            title=":rainbow[Ichimoku Cloud]",
        )
        st.plotly_chart(figIchimoku, use_container_width=True)

st.subheader(':rainbow[Dự đoán theo chỉ số kỹ thuật]')

# Dự đoán với chỉ số MACD
with st.expander(':rainbow[Dự đoán với chỉ số MACD]'):
    response_macd = forecast_price(
        question="Dựa vào các chỉ số MACD trên, đưa ra phân tích giá chứng khoán trong thời gian tới, thời điểm mua và bán cụ thể.",
        docs=df_macd
    )
    st.write(df_macd)
    st.write(response_macd)

# Dự đoán với chỉ số Bollinger Bands
with st.expander(':rainbow[Dự đoán với chỉ số BOLL]'):
    response_boll = forecast_price(
        question="Dựa vào các chỉ số Bollinger Bands trên, đưa ra phân tích giá chứng khoán trong thời gian tới, thời điểm mua và bán cụ thể.",
        docs=df_boll
    )
    st.write(df_boll)
    st.write(response_boll)

# Dự đoán với chỉ số EMA/SMA
with st.expander(':rainbow[Dự đoán với chỉ số EMA]'):
    response_ema = forecast_price(
        question="Dựa vào các chỉ số EMA/SMA trên, đưa ra phân tích giá chứng khoán trong thời gian tới, thời điểm mua và bán cụ thể.",
        docs=df_ma
    )
    st.write(df_ma)
    st.write(response_ema)

# Dự đoán với chỉ số RSI
with st.expander(':rainbow[Dự đoán với chỉ số RSI]'):
    response_rsi = forecast_price(
        question="Dựa vào các chỉ số RSI trên, đưa ra phân tích giá chứng khoán trong thời gian tới, thời điểm mua và bán cụ thể.",
        docs=df_rsi
    )
    st.write(df_rsi)
    st.write(response_rsi)

# Dự đoán với chỉ số ATR
with st.expander(':rainbow[Dự đoán với chỉ số ATR]'):
    response_atr = forecast_price(
        question="Dựa vào các chỉ số ATR trên, đưa ra phân tích giá chứng khoán trong thời gian tới, thời điểm mua và bán cụ thể.",
        docs=df_atr
    )
    st.write(df_atr)
    st.write(response_atr)

# Dự đoán với chỉ số Stochastic Oscillator
with st.expander(':rainbow[Dự đoán với chỉ số Stochastic Oscillator]'):
    response_so = forecast_price(
        question="Dựa vào các chỉ số Stochastic Oscillator trên, đưa ra phân tích giá chứng khoán trong thời gian tới, thời điểm mua và bán cụ thể.",
        docs=df_so
    )
    st.write(df_so)
    st.write(response_so)

# Dự đoán với chỉ số CCI
with st.expander(':rainbow[Dự đoán với chỉ số CCI]'):
    response_cci = forecast_price(
        question="Dựa vào các chỉ số CCI trên, đưa ra phân tích giá chứng khoán trong thời gian tới, thời điểm mua và bán cụ thể.",
        docs=df_cci
    )
    st.write(df_cci)
    st.write(response_cci)

# Dự đoán với chỉ số OBV
with st.expander(':rainbow[Dự đoán với chỉ số OBV]'):
    response_obv = forecast_price(
        question="Dựa vào các chỉ số OBV trên, đưa ra phân tích giá chứng khoán trong thời gian tới, thời điểm mua và bán cụ thể.",
        docs=df_obv
    )
    st.write(df_obv)
    st.write(response_obv)

# Dự đoán với chỉ số Williams %R
with st.expander(':rainbow[Dự đoán với chỉ số Williams %R]'):
    response_williams_r = forecast_price(
        question="Dựa vào các chỉ số Williams %R trên, đưa ra phân tích giá chứng khoán trong thời gian tới, thời điểm mua và bán cụ thể.",
        docs=df_williams_r
    )
    st.write(df_williams_r)
    st.write(response_williams_r)

# Dự đoán với chỉ số Ichimoku Cloud
with st.expander(':rainbow[Dự đoán với chỉ số Ichimoku Cloud]'):
    response_ichimoku = forecast_price(
        question="Dựa vào các chỉ số Ichimoku Cloud trên, đưa ra phân tích giá chứng khoán trong thời gian tới, thời điểm mua và bán cụ thể.",
        docs=df_ichimoku
    )
    st.write(df_ichimoku)
    st.write(response_ichimoku)
