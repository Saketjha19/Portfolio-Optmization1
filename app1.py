import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

st.set_page_config(page_title="Portfolio Optimization", page_icon="📊", layout="wide")
st.title("📈 Portfolio Optimization App")

st.markdown("""
This app allows you to **optimize a portfolio** using historical stock data.
You can enter any list of stock tickers, choose an optimization type, and visualize results.
""")

# ---- Sidebar Inputs ----
st.sidebar.header("⚙️ Input Parameters")
tickers_input = st.sidebar.text_input("Enter Stock Tickers (comma-separated)", "AAPL,MSFT,GOOG,AMZN")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

period = st.sidebar.selectbox("Select Historical Period", ["1y", "3y", "5y"], index=2)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (annual)", value=0.04, step=0.01)
opt_goal = st.sidebar.radio("Optimization Goal", ["Maximize Sharpe Ratio", "Minimize Volatility"])
run_optimization = st.sidebar.button("Run Optimization")

# ---- Data Fetching ----
@st.cache_data
def get_data(tickers, period):
    return yf.download(tickers, period=period)["Adj Close"].dropna()

if run_optimization:
    data = get_data(tickers, period)
    log_returns = np.log(data / data.shift(1)).dropna()

    # ---- Optimization ----
    def portfolio_performance(weights):
        ret = np.sum(log_returns.mean() * weights) * 252
        vol = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 252, weights)))
        sharpe = (ret - risk_free_rate) / vol
        return ret, vol, sharpe

    def neg_sharpe(weights):
        return -portfolio_performance(weights)[2]

    def portfolio_vol(weights):
        return portfolio_performance(weights)[1]

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(len(tickers)))
    init_guess = np.repeat(1 / len(tickers), len(tickers))

    if opt_goal == "Maximize Sharpe Ratio":
        result = minimize(neg_sharpe, init_guess, bounds=bounds, constraints=cons)
    else:
        result = minimize(portfolio_vol, init_guess, bounds=bounds, constraints=cons)

    opt_weights = result.x
    ret, vol, sharpe = portfolio_performance(opt_weights)

    # ---- Display Results ----
    st.subheader("✅ Optimization Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Annual Return", f"{ret:.2%}")
    col2.metric("Annual Volatility", f"{vol:.2%}")
    col3.metric("Sharpe Ratio", f"{sharpe:.2f}")

    weights_df = pd.DataFrame({
        "Ticker": tickers,
        "Weight": np.round(opt_weights, 4)
    })
    st.write("### Portfolio Weights")
    st.dataframe(weights_df, use_container_width=True)
    st.bar_chart(weights_df.set_index("Ticker"))

    # ---- Portfolio Performance ----
    portfolio_returns = (log_returns @ opt_weights)
    cumulative = (1 + portfolio_returns).cumprod()
    st.subheader("📈 Cumulative Portfolio Performance")
    st.line_chart(cumulative)

else:
    st.info("👈 Enter tickers and click **Run Optimization** to begin.")
