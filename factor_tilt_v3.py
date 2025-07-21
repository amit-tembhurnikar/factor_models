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
        prices_df.columns = list(prices.keys())[:prices_df.shape[1]]  # match column length
        prices_df.index = pd.to_datetime(prices_df.index)
        prices_df = prices_df.dropna(axis=1, how='all')

        monthly_prices = prices_df.resample('ME').last()
        log_returns = np.log(prices_df / prices_df.shift(1))

        tilt_table = []
        benchmark_tilt_table = []

        # Get benchmark tickers and data
        benchmark_prices = {}
        benchmark_shares = {}
        for symbol in tickers[:50]:
            try:
                df = yf.download(symbol + ".NS", start=START_DATE, end=END_DATE, interval="1d", progress=False, auto_adjust=True)
                if df.empty or 'Close' not in df.columns:
                    continue
                df = df[['Close']].rename(columns={'Close': symbol})
                benchmark_prices[symbol] = df
                ticker_obj = yf.Ticker(symbol + ".NS")
                info = ticker_obj.info
                benchmark_shares[symbol] = info.get("sharesOutstanding", np.nan)
            except:
                continue

        benchmark_prices_df = pd.concat(benchmark_prices.values(), axis=1)
        benchmark_prices_df.columns = list(benchmark_prices.keys())[:benchmark_prices_df.shape[1]]
        benchmark_prices_df.index = pd.to_datetime(benchmark_prices_df.index)
        benchmark_prices_df = benchmark_prices_df.dropna(axis=1, how='all')
        benchmark_monthly_prices = benchmark_prices_df.resample('ME').last()
        benchmark_log_returns = np.log(benchmark_prices_df / benchmark_prices_df.shift(1))

        for date in monthly_prices.index[12:]:
            try:
                # Portfolio
                current_prices = monthly_prices.loc[date]
                past_prices = monthly_prices.shift(12).loc[date]
                returns_12m = (current_prices / past_prices) - 1
                shares_series = pd.Series(shares_outstanding)
                mcap = current_prices * shares_series
                size = np.log(mcap)
                momentum = returns_12m
                volatility = log_returns.loc[:date].rolling(252).std().loc[date]
                value = pd.Series(np.random.normal(0, 1, len(current_prices)), index=current_prices.index)
                growth = pd.Series(np.random.normal(0, 1, len(current_prices)), index=current_prices.index)

                factors = pd.DataFrame({
                    'Size': size,
                    'Momentum': momentum,
                    'Volatility': volatility,
                    'Value': value,
                    'Growth': growth
                }).dropna()

                # Normalize factors
                norm_factors = factors.apply(zscore)
                weights = mcap[factors.index] / mcap[factors.index].sum()
                tilt = (norm_factors.T @ weights).rename(date)
                tilt_table.append(tilt)

                # Benchmark
                bm_current = benchmark_monthly_prices.loc[date]
                bm_past = benchmark_monthly_prices.shift(12).loc[date]
                bm_returns_12m = (bm_current / bm_past) - 1
                bm_mcap = bm_current * pd.Series(benchmark_shares)
                bm_size = np.log(bm_mcap)
                bm_momentum = bm_returns_12m
                bm_vol = benchmark_log_returns.loc[:date].rolling(252).std().loc[date]
                bm_value = pd.Series(np.random.normal(0, 1, len(bm_current)), index=bm_current.index)
                bm_growth = pd.Series(np.random.normal(0, 1, len(bm_current)), index=bm_current.index)

                bm_factors = pd.DataFrame({
                    'Size': bm_size,
                    'Momentum': bm_momentum,
                    'Volatility': bm_vol,
                    'Value': bm_value,
                    'Growth': bm_growth
                }).dropna()

                bm_norm = bm_factors.apply(zscore)
                bm_weights = bm_mcap[bm_factors.index] / bm_mcap[bm_factors.index].sum()
                bm_tilt = (bm_norm.T @ bm_weights).rename(date)
                benchmark_tilt_table.append(bm_tilt)
            except:
                continue

        if tilt_table and benchmark_tilt_table:
            tilt_df = pd.DataFrame(tilt_table)
            bm_tilt_df = pd.DataFrame(benchmark_tilt_table)

            st.subheader("Factor Tilt Evolution (Portfolio vs Benchmark)")
            for factor in tilt_df.columns:
                st.subheader(f"{factor} Tilt vs {index_name}")
                st.line_chart(pd.DataFrame({"Portfolio": tilt_df[factor], "Benchmark": bm_tilt_df[factor]}))

            st.subheader("Latest Factor Tilt Comparison (Bar Plot)")
            latest_port = tilt_df.iloc[-1]
            latest_bm = bm_tilt_df.iloc[-1]
            fig, ax = plt.subplots()
            df_plot = pd.DataFrame({"Portfolio": latest_port, "Benchmark": latest_bm})
            df_plot.plot(kind='bar', ax=ax)
            ax.set_ylabel("Factor Tilt (Z-score)")
            st.pyplot(fig)
        else:
            st.warning("Could not calculate tilt due to missing data.")
