import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant
from scipy.stats import linregress
import matplotlib.pyplot as plt

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

def calculate_exposure(returns, factor_scores):
    exposures = {}
    returns.index = returns.index.tz_localize(None)
    factor_scores.index = factor_scores.index.tz_localize(None)

    for stock in returns.columns:
        y = returns[stock].dropna()
        X = factor_scores.loc[y.index]
        df = pd.concat([y, X], axis=1).dropna()

        if df.shape[0] < len(factor_scores.columns) + 1:
            continue

        y_clean = df.iloc[:, 0].astype(float)
        X_clean = df.iloc[:, 1:].astype(float)
        X_const = add_constant(X_clean)
        model = OLS(y_clean, X_const).fit()
        exposures[stock] = model.params.drop("const")

    exposure_df = pd.DataFrame(exposures).T
    exposure_df.index.name = "Ticker"
    exposure_df.columns.name = "Factor"
    return exposure_df

# === Streamlit UI ===
st.set_page_config(layout="wide")
st.title("📊 Nifty Factor Model Dashboard")

option = st.sidebar.radio("Choose Analysis Type:", ["Factor Models for Indices", "Factor Risk of Portfolio"])

start_date = "2020-01-01"
end_date = "2024-12-31"

if option == "Factor Models for Indices":
    index_name = st.selectbox("Select Index", ["Nifty 50", "Nifty 500", "Nifty Midcap 50", "Nifty Smallcap 50"])
    df_members = get_index_members(index_name)
    tickers = df_members["Symbol"].dropna().tolist()[:50]  # Limit for speed

    st.write(f"Running Factor Model for {index_name}...")
    price_data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]
    shares_out_data = {ticker: yf.Ticker(ticker).info.get("sharesOutstanding", np.nan) for ticker in tickers}

    mcap_df = calculate_mcap(price_data, shares_out_data)
    monthly_prices = price_data.resample("M").last()
    monthly_returns = monthly_prices.pct_change().dropna()
    size_factor = -np.log(mcap_df).resample("M").last().dropna()

    perf_df = pd.DataFrame()
    perf_df["Return"] = monthly_returns.mean(axis=1)
    perf_df["Cumulative"] = (1 + perf_df["Return"]).cumprod()

    sharpe = perf_df["Return"].mean() / perf_df["Return"].std() * np.sqrt(4)
    x = np.arange(len(perf_df))
    y = np.log(perf_df["Cumulative"])
    slope, _, _, _, _ = linregress(x, y)
    alpha = slope * 4

    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    st.metric("Annualized Alpha", f"{alpha:.2%}")

    st.line_chart(perf_df["Cumulative"], use_container_width=True)

    st.subheader("Factor Exposure")
    exposure_df = calculate_exposure(monthly_returns, size_factor)
    st.dataframe(exposure_df)

elif option == "Factor Risk of Portfolio":
    index_name = st.selectbox("Select Benchmark Index", ["Nifty 50", "Nifty 500", "Nifty Midcap 50", "Nifty Smallcap 50"])
    df_members = get_index_members(index_name)
    tickers = df_members["Symbol"].dropna().tolist()

    selected_stocks = st.multiselect("Select up to 20 stocks for your portfolio", options=tickers, max_selections=20)

    if selected_stocks:
        price_data = yf.download(selected_stocks, start=start_date, end=end_date)["Adj Close"]
        shares_out_data = {ticker: yf.Ticker(ticker).info.get("sharesOutstanding", np.nan) for ticker in selected_stocks}

        mcap_df = calculate_mcap(price_data, shares_out_data)
        monthly_prices = price_data.resample("M").last()
        monthly_returns = monthly_prices.pct_change().dropna()
        size_factor = -np.log(mcap_df).resample("M").last().dropna()

        st.subheader("Portfolio Factor Exposure")
        exposure_df = calculate_exposure(monthly_returns, size_factor)
        st.dataframe(exposure_df.loc[selected_stocks])

        perf_df = pd.DataFrame()
        perf_df["Return"] = monthly_returns.mean(axis=1)
        perf_df["Cumulative"] = (1 + perf_df["Return"]).cumprod()

        sharpe = perf_df["Return"].mean() / perf_df["Return"].std() * np.sqrt(4)
        x = np.arange(len(perf_df))
        y = np.log(perf_df["Cumulative"])
        slope, _, _, _, _ = linregress(x, y)
        alpha = slope * 4

        st.metric("Portfolio Sharpe Ratio", f"{sharpe:.2f}")
        st.metric("Portfolio Annualized Alpha", f"{alpha:.2%}")
        st.line_chart(perf_df["Cumulative"], use_container_width=True)
