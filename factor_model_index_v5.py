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

if option == "Factor Models for Indices":
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

    prices = pd.concat(data.values(), axis=1)
    prices.columns = data.keys()

    prices.index = pd.to_datetime(prices.index)
    prices = prices.dropna(axis=1, how='all')

    # === Calculate Barra-style factors ===
    log_returns = np.log(prices / prices.shift(1))
    monthly_prices = prices.resample('ME').last()

    factor_data = {}
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

            # Placeholder for value/growth (replace with real fundamentals later)
            value = pd.Series(np.random.normal(0, 1, len(current_prices)), index=current_prices.index)
            growth = pd.Series(np.random.normal(0, 1, len(current_prices)), index=current_prices.index)

            factors = pd.DataFrame({
                'Size': size,
                'Momentum': momentum,
                'Volatility': volatility,
                'Value': value,
                'Growth': growth
            })
            factor_data[date] = factors
        except KeyError:
            continue

    # === Backtest each factor ===
    def backtest_factor(factor_name):
        factor_returns = []
        for i, date in enumerate(factor_data.keys()):
            if date.month not in REBALANCE_MONTHS:
                continue
            df = factor_data[date].dropna()
            if len(df) < 10:
                continue

            top = df[factor_name].nlargest(int(len(df)*0.1)).index
            bottom = df[factor_name].nsmallest(int(len(df)*0.1)).index

            try:
                next_date = list(factor_data.keys())[i+1]
            except IndexError:
                break

            try:
                future_prices = monthly_prices.loc[next_date]
                curr_prices = monthly_prices.loc[date]

                top_ret = (future_prices[top] / curr_prices[top] - 1).mean()
                bottom_ret = (future_prices[bottom] / curr_prices[bottom] - 1).mean()
                factor_returns.append((next_date, top_ret, bottom_ret))
            except:
                continue

        df = pd.DataFrame(factor_returns, columns=["Date", "Top", "Bottom"]).set_index("Date")
        df.index = pd.to_datetime(df.index)
        df["Top_10%_companies"] = (1 + df["Top"]).cumprod()
        df["Bottom_10%_companies"] = (1 + df["Bottom"]).cumprod()
        df["Return"] = df["Top"] - df["Bottom"]
        df["Cumulative"] = (1 + df["Return"]).cumprod()
        return df

    # === Run backtests ===
    factors = ['Size', 'Momentum', 'Volatility', 'Value', 'Growth']
    results = {factor: backtest_factor(factor) for factor in factors}

    # === Benchmark ===
    benchmark = yf.download("^NSEI", start=START_DATE, end=END_DATE, interval="1mo")['Close'].resample('ME').last()
    benchmark = benchmark.rename(columns={"^NSEI": "Nifty Index"})
    benchmark_ret = benchmark.pct_change().dropna()
    benchmark_cum = (1 + benchmark_ret).cumprod()

    # === Plot Streamlit Outputs ===
    st.title("Factor Strategy Backtests (Strategy only for theoretical representation not a investment advise)")

    st.markdown(f"""
    **Back-test analysis for Nifty 50 Index**

    - Current index members are used for historical analysis.
    - I understand this does not reflect actual index history and is only for model representation.
    - Portfolios are rebalanced quarterly.
    """)

    for factor in factors:
        df = results[factor]
        st.subheader(f"{factor} Factor")
        if factor == 'Size':
            st.subheader("Methodology")
            st.markdown("""
            Have taken historic prices and multiplied by current outstanding shares to get historic Market Cap series.

            - Largest 10% of the companies are considered as Top 10%.
            - Smallest 10% of the companies are considered as bottom 10%.
            """)
        if factor == 'Momentum':
            st.subheader("Methodology")
            st.markdown("""
            Have taken historic price series and calculated 12 month returns as on quarter end.

            - Top 10% and bottom 10% of the companies are considered on the basis of 12 month returns.
            """)
        if factor == 'Volatility':
            st.subheader("Methodology")
            st.markdown("""
            Have taken historic prices and volatility is calculated as standard deviation of 12 month daily log change.

            - Companies with highest volatility are considered as Top 10%
            - 10% companies with lowest volatility are bottom 10% companies.
            """)
        if factor == 'Value':
            st.subheader("Methodology")
            st.markdown("""
            As historic fundamental data is not available publicly (API-based database), random data is used.

            - Costliest 10% of the companies are considered as Top 10%.
            - Cheapest 10% of the companies are considered as bottom 10%.
            - Ideally, cheapest companies are considered as top value companies.
            """)
        if factor == 'Growth':
            st.subheader("Methodology")
            st.markdown("""
            As historic fundamental data is not available publicly (API-based database), random data is used.

            - 10% of the companies with highest growth are considered as Top 10%.
            - 10% of the companies with lowest growth are considered as bottom 10%.
            """)

        benchmark_cum_renamed = benchmark_cum.copy()
        benchmark_cum_renamed.name = "Nifty 50"
        combined = df[["Top_10%_companies", "Bottom_10%_companies"]].join(benchmark_cum_renamed, how="inner")
        st.line_chart(combined)

    # === Factor Risk Profile ===
    st.header(f"Factor Backtesk Results for Nifty 50")
    risk_profile = pd.DataFrame(index=factors, columns=['Volatility', 'Max Drawdown', 'Sharpe Ratio', 'Alpha'])

    for factor in factors:
        df = results[factor]

        if "Return" not in df.columns or df["Return"].dropna().empty:
            continue

        returns = df["Return"].dropna()
        cumulative = df["Cumulative"]

        volatility = returns.std() * np.sqrt(4)  # Quarterly
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()

        sharpe = returns.mean() / returns.std() * np.sqrt(4)

        x = np.arange(len(returns))
        y = np.log(cumulative.dropna())
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        alpha = slope * 4  # Quarterly to annual

        risk_profile.loc[factor] = [volatility, max_dd, sharpe, alpha]

    st.dataframe(risk_profile.astype(float).style.format("{:.2f}"))

elif option == "Factor Risk of Portfolio":
    index_name = st.selectbox("Select Benchmark Index", ["Nifty 50", "Nifty 500"])
    df_members = get_index_members(index_name)
    tickers = df_members["Symbol"].dropna().tolist()

    selected_stocks = st.multiselect("Select up to 20 stocks for your portfolio", options=tickers, max_selections=20)
