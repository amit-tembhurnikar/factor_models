import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import linregress
from statsmodels.api import OLS, add_constant
import streamlit as st

# === Parameters ===
START_DATE = "2005-12-31"
END_DATE = "2025-07-31"
REBALANCE_MONTHS = [3, 6, 9, 12]  # Quarterly

# Step 1: Load NSE500 list
nse500 = pd.read_html("https://en.wikipedia.org/wiki/NIFTY_50", header=0)[1]
symbols = nse500['Symbol'].tolist()

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
    df["Top_Cumulative"] = (1 + df["Top"]).cumprod()
    df["Bottom_Cumulative"] = (1 + df["Bottom"]).cumprod()
    return df


# === Run backtests ===
factors = ['Size', 'Momentum', 'Volatility', 'Value', 'Growth']
results = {factor: backtest_factor(factor) for factor in factors}


# === Benchmark ===
benchmark = yf.download("^NSEI", start=START_DATE, end=END_DATE, interval="1mo")["Close"].resample('ME').last()
benchmark_ret = benchmark.pct_change().dropna()
benchmark_cum = (1 + benchmark_ret).cumprod()


# === Plot Streamlit Outputs ===
st.title("Factor Strategy Backtests")
for factor in factors:
    df = results[factor]
    st.subheader(f"{factor} Factor")
    benchmark_cum_renamed = benchmark_cum.copy()
    benchmark_cum_renamed.name = "Nifty 50"
    combined = df[["Top_Cumulative", "Bottom_Cumulative"]].join(benchmark_cum_renamed, how="inner")
    st.line_chart(combined)

# === Factor Risk Profile ===
st.header("Factor Risk Profile")
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

