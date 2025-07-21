import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant
from scipy.stats import linregress
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

option = st.sidebar.radio("Choose Analysis Type:", ["Factor Models for Indices", "Factor Risk of Portfolio"])

START_DATE = "2005-12-31"
END_DATE = "2025-07-31"
REBALANCE_MONTHS = [3, 6, 9, 12]  # Quarterly

if option == "Factor Risk of Portfolio":
    index_name = st.selectbox("Select Benchmark Index", ["Nifty 50", "Nifty 500"])
    df_members = get_index_members(index_name)
    tickers = df_members["Symbol"].dropna().tolist()

    selected_stocks = st.multiselect("Select up to 20 stocks for your portfolio", options=tickers, max_selections=20)

    if selected_stocks:
        prices = {}
        shares_outstanding = {}
        for symbol in selected_stocks:
            try:
                df = yf.download(symbol + ".NS", start=START_DATE, end=END_DATE, interval="1d", progress=False, auto_adjust=True)
                if df.empty or 'Close' not in df.columns:
                    continue
                df = df[['Close']].rename(columns={'Close': symbol})
                prices[symbol] = df
                ticker = yf.Ticker(symbol + ".NS")
                info = ticker.info
                shares_outstanding[symbol] = info.get("sharesOutstanding", np.nan)
            except:
                continue

        if not prices:
            st.error("No data downloaded for selected stocks.")
            st.stop()

        prices_df = pd.concat(prices.values(), axis=1)
        prices_df.columns = prices.keys()
        prices_df.index = pd.to_datetime(prices_df.index)
        prices_df = prices_df.dropna(axis=1, how='all')

        monthly_prices = prices_df.resample('ME').last()
        log_returns = np.log(prices_df / prices_df.shift(1))

        tilt_table = []
        for date in monthly_prices.index[12:]:
            try:
                current_prices = monthly_prices.loc[date]
                past_prices = monthly_prices.shift(12).loc[date]
                returns_12m = (current_prices / past_prices) - 1

                shares_series = pd.Series(shares_outstanding)
                mcap = current_prices * shares_series
                size = np.log(mcap)
                momentum = returns_12m
                volatility = log_returns.loc[:date].rolling(252).std().loc[date]

                # Dummy values for Value and Growth
                value = pd.Series(np.random.normal(0, 1, len(current_prices)), index=current_prices.index)
                growth = pd.Series(np.random.normal(0, 1, len(current_prices)), index=current_prices.index)

                factors = pd.DataFrame({
                    'Size': size,
                    'Momentum': momentum,
                    'Volatility': volatility,
                    'Value': value,
                    'Growth': growth
                }).dropna()

                weights = mcap[factors.index] / mcap[factors.index].sum()
                tilt = (factors.T @ weights).rename(date)
                tilt_table.append(tilt)
            except:
                continue

        if tilt_table:
            tilt_df = pd.DataFrame(tilt_table)
            st.subheader("Factor Tilt Evolution (Time Series)")
            st.line_chart(tilt_df)

            st.subheader("Latest Factor Tilt (Bar Plot)")
            latest_tilt = tilt_df.iloc[-1]
            fig, ax = plt.subplots()
            latest_tilt.plot(kind='bar', ax=ax)
            ax.set_ylabel("Factor Tilt")
            st.pyplot(fig)
        else:
            st.warning("Could not calculate tilt due to missing data.")
