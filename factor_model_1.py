import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant
from scipy.stats import linregress, zscore
import matplotlib.pyplot as plt
from tqdm import tqdm

# === Helper Functions ===
def get_index_members(index_name):
    if index_name == "Nifty 50":
        df = pd.read_html("https://en.wikipedia.org/wiki/NIFTY_50", header=0)[1]
    elif index_name == "Nifty 500":
        df = pd.read_html("https://en.wikipedia.org/wiki/NIFTY_500", header=0)[4]
    else:
        return pd.DataFrame()

    return df

def calculate_mcap(price_df, shares_dict):
    mcap = pd.DataFrame()
    for ticker in price_df.columns:
        shares_out = shares_dict.get(ticker, np.nan)
        if not np.isnan(shares_out):
            mcap[ticker] = price_df[ticker] * shares_out
    return mcap

# === Streamlit UI ===
st.set_page_config(layout="wide")
st.title("Factor Model Dashboard")

option = st.sidebar.radio("Choose Analysis Type:", ["Factor Investment Model for Indices", "Factor Risk of Portfolio"])

START_DATE = "2005-12-31"
END_DATE = "2025-07-31"
REBALANCE_MONTHS = [3, 6, 9, 12]  # Quarterly

if option == "Factor Investment Model for Indices (Long only)":
    st.subheader("Factor Investment for Indian indices")
    index_name = st.selectbox("Select Index", ["Nifty 50", "Nifty 500"])
    df_members = get_index_members(index_name)

    symbols = df_members["Symbol"].dropna().tolist()[:50]  # Limit for speed


    # Step 2: Download adjusted close daily price history
    data = {}
    shares_outstanding = {}
    for symbol in tqdm(symbols):
        try:
            df = yf.download(symbol + ".NS", start=START_DATE, end=END_DATE, interval="1d", progress=False, auto_adjust=True)
            if df.empty or 'Close' not in df.columns:
                continue
            df = df[['Close']].rename(columns={'Close': symbol})
            data[symbol] = df
            ticker = yf.Ticker(symbol + ".NS")
            info = ticker.info
            shares_outstanding[symbol] = info.get("sharesOutstanding", np.nan)
        except:
            continue

    if not data:
        st.error("No data downloaded from yfinance. Check your internet connection or ticker symbols.")
        st.stop()

    valid_data = {k: v for k, v in data.items() if not v.empty}
    prices = pd.concat(valid_data.values(), axis=1)
    prices.columns = list(valid_data.keys())

    prices.index = pd.to_datetime(prices.index)
    prices = prices.dropna(axis=1, how='all')

    # === Calculate Barra-style factors ===
    log_returns = np.log(prices / prices.shift(1))
    monthly_prices = prices.resample('ME').last()

    st.dataframe(monthly_prices)


