# Nifty 500 Systematic Momentum Portfolio

A quantitative, rules-based algorithmic stock screener and historical backtesting engine designed for the Indian Nifty 500 universe.

This project automatically scans daily market data across multiple timeframes (Daily, Weekly, Monthly) to identify statistically significant momentum trends confirmed by structural volume support and volatility expansion. It includes a comprehensive backtesting suite that simulates exactly how this strategy would have performed over the last 4 years with an initial capital of INR 5 Lakhs, and an interactive Streamlit UI for visual analysis.

---

## 🚀 Features

- **Automated Universe Fetching**: Dynamically reads the current Nifty 50 constituents directly from the NSE archives.
- **Multitimeframe Momentum Scoring**: Computes simple moving average (SMA) alignments:
  - Daily Close > 50-period SMA
  - Weekly Close > 10-period SMA
  - Monthly Close > 10-period SMA
- **Cross-Sectional Factor Z-Scoring**: Ranks the eligible universe using a blended composite score of:
  - 1-Month Return (40% weight)
  - Relative Volume Breakout (30% weight: 5-Day Vol / 50-Day Vol)
  - Volatility Expansion (30% weight: Current ATR / 50-Day Avg ATR)
- **Historical Backtester (`backtester.py`)**: Precisely simulates 4.5+ years of daily data to generate a realistic 4-year monthly-rebalanced equity curve, Max Drawdown metrics, Strategy Win Rate, and full trade logs.
- **Monte Carlo Analysis**: Projects 100 randomly sampled 1-year forward equity paths to establish expected medians and 90% confidence intervals.
- **Interactive UI (`gui_screener.py`)**: A Streamlit heatmap WebApp that visualizes your currently suggested model portfolio, the daily screener results, and your complete simulated performance metrics.

---

## 🛠️ Setup & Installation

### 1. Prerequisites
You need python 3.8+ installed on your system.

Clone the repository and install the dependencies into your environment:
```bash
git clone <your-github-repo-url>
cd xalgo_quant_portfolio_manager

# (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows

# Install required packages
pip install -r requirements.txt
```

### 2. Generate Scan Results & Run Backtest
Before starting the user interface, you must generate the underlying `.csv` local database caches by running the backtesting data miner.

```bash
# This downloads 4.5 years of data for all 500 Nifty constituents
# Calculates all indicators, scores, simulates exact trades, and builds the UI caches
python backtester.py
```
*Note: Due to the volume of data fetched from Yahoo Finance, this might take 2-4 minutes on the first run.*

### 3. Launch the Application
Start the interactive Streamlit user interface:

```bash
streamlit run gui_screener.py
```

It will automatically open a local web server (usually at `http://localhost:8501`) displaying your Momentum Heatmap, your suggested INR 5 Lakh Next-Rebalance Portfolio, and the 4-year backtested statistics.

---

## 📝 Strategy Logic & Rules

### Rebalance Rules
1. **Frequency**: Rebalancing occurs strictly on the last valid trading day of every month.
2. **Sizing**: Equal-weighting an allocated pool of capital (INR 5,00,000 baseline) across the highest-scoring 15 positions.
3. **Execution**: Assumes trades are executed the following morning or very close to the market end.
4. **Sell Rules**: If a previously held stock drops out of the absolute highest score rankings, or its multitimeframe Momentum alignment definitively breaks, the algorithms sells the entire position and re-allocates the capital towards new top-ranked candidates.

### Disclaimer
*This repository contains educational quantitative research code as analyzed and prepared by Santoo Chakraborty. The generated model portfolios and historical backtests do not account for exact trading friction, slippage, taxation, or exchange outages. **Always exercise strict risk management**—standalone momentum strategies can experience >30% drawdowns during larger absolute market regimes.*
