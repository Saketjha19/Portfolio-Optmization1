{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOEjOPUFHMUqYhzatCxO+gY",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Saketjha19/Portfolio-Optmization1/blob/main/app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "P6gbO9CIvpa2"
      },
      "outputs": [],
      "source": [
        "import streamlit as st\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import yfinance as yf\n",
        "from scipy.optimize import minimize\n",
        "\n",
        "st.set_page_config(page_title=\"Portfolio Optimization\", page_icon=\"📊\", layout=\"wide\")\n",
        "st.title(\"📈 Portfolio Optimization App\")\n",
        "\n",
        "st.markdown(\"\"\"\n",
        "This app allows you to **optimize a portfolio** using historical stock data.\n",
        "You can enter any list of stock tickers, choose an optimization type, and visualize results.\n",
        "\"\"\")\n",
        "\n",
        "# ---- Sidebar Inputs ----\n",
        "st.sidebar.header(\"⚙️ Input Parameters\")\n",
        "tickers_input = st.sidebar.text_input(\"Enter Stock Tickers (comma-separated)\", \"AAPL,MSFT,GOOG,AMZN\")\n",
        "tickers = [t.strip().upper() for t in tickers_input.split(\",\") if t.strip()]\n",
        "\n",
        "period = st.sidebar.selectbox(\"Select Historical Period\", [\"1y\", \"3y\", \"5y\"], index=2)\n",
        "risk_free_rate = st.sidebar.number_input(\"Risk-Free Rate (annual)\", value=0.04, step=0.01)\n",
        "opt_goal = st.sidebar.radio(\"Optimization Goal\", [\"Maximize Sharpe Ratio\", \"Minimize Volatility\"])\n",
        "run_optimization = st.sidebar.button(\"Run Optimization\")\n",
        "\n",
        "# ---- Data Fetching ----\n",
        "@st.cache_data\n",
        "def get_data(tickers, period):\n",
        "    return yf.download(tickers, period=period)[\"Adj Close\"].dropna()\n",
        "\n",
        "if run_optimization:\n",
        "    data = get_data(tickers, period)\n",
        "    log_returns = np.log(data / data.shift(1)).dropna()\n",
        "\n",
        "    # ---- Optimization ----\n",
        "    def portfolio_performance(weights):\n",
        "        ret = np.sum(log_returns.mean() * weights) * 252\n",
        "        vol = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 252, weights)))\n",
        "        sharpe = (ret - risk_free_rate) / vol\n",
        "        return ret, vol, sharpe\n",
        "\n",
        "    def neg_sharpe(weights):\n",
        "        return -portfolio_performance(weights)[2]\n",
        "\n",
        "    def portfolio_vol(weights):\n",
        "        return portfolio_performance(weights)[1]\n",
        "\n",
        "    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})\n",
        "    bounds = tuple((0, 1) for _ in range(len(tickers)))\n",
        "    init_guess = np.repeat(1 / len(tickers), len(tickers))\n",
        "\n",
        "    if opt_goal == \"Maximize Sharpe Ratio\":\n",
        "        result = minimize(neg_sharpe, init_guess, bounds=bounds, constraints=cons)\n",
        "    else:\n",
        "        result = minimize(portfolio_vol, init_guess, bounds=bounds, constraints=cons)\n",
        "\n",
        "    opt_weights = result.x\n",
        "    ret, vol, sharpe = portfolio_performance(opt_weights)\n",
        "\n",
        "    # ---- Display Results ----\n",
        "    st.subheader(\"✅ Optimization Results\")\n",
        "    col1, col2, col3 = st.columns(3)\n",
        "    col1.metric(\"Expected Annual Return\", f\"{ret:.2%}\")\n",
        "    col2.metric(\"Annual Volatility\", f\"{vol:.2%}\")\n",
        "    col3.metric(\"Sharpe Ratio\", f\"{sharpe:.2f}\")\n",
        "\n",
        "    weights_df = pd.DataFrame({\n",
        "        \"Ticker\": tickers,\n",
        "        \"Weight\": np.round(opt_weights, 4)\n",
        "    })\n",
        "    st.write(\"### Portfolio Weights\")\n",
        "    st.dataframe(weights_df, use_container_width=True)\n",
        "    st.bar_chart(weights_df.set_index(\"Ticker\"))\n",
        "\n",
        "    # ---- Portfolio Performance ----\n",
        "    portfolio_returns = (log_returns @ opt_weights)\n",
        "    cumulative = (1 + portfolio_returns).cumprod()\n",
        "    st.subheader(\"📈 Cumulative Portfolio Performance\")\n",
        "    st.line_chart(cumulative)\n",
        "\n",
        "else:\n",
        "    st.info(\"👈 Enter tickers and click **Run Optimization** to begin.\")\n"
      ]
    }
  ]
}