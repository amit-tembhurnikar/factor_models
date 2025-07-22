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

    #valid_data = {k: v for k, v in data.items() if not v.empty}
    prices = pd.concat(data, axis=1)
    #prices.columns = list(valid_data.keys())

    #prices.index = pd.to_datetime(prices.index)
    #prices = prices.dropna(axis=1, how='all')

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
    st.title("Factor Strategy Backtests (Strategy only for theoretical representation not an investment advise)")

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
    st.subheader("Factor Risk of Portfolio")
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

        prices_df = pd.concat(prices, axis=1)
        #prices_df.columns = list(prices.keys())[:prices_df.shape[1]]  # match column length
        #prices_df.index = pd.to_datetime(prices_df.index)
        #prices_df = prices_df.dropna(axis=1, how='all')

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

        benchmark_prices_df = pd.concat(benchmark_prices, axis=1)
        #benchmark_prices_df.columns = list(benchmark_prices.keys())[:benchmark_prices_df.shape[1]]
        #benchmark_prices_df.index = pd.to_datetime(benchmark_prices_df.index)
        #benchmark_prices_df = benchmark_prices_df.dropna(axis=1, how='all')
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
