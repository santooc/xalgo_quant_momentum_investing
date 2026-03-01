import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def get_historical_data_for_backtest(symbols, years=4):
    """
    Downloads historical data required for a backtest.
    Needs `years` + 2 years (for 50-day and monthly indicators)
    """
    end_date = datetime.today()
    start_date = end_date - timedelta(days=int((years + 1) * 365))
    
    print(f"Downloading historical data for {len(symbols)} symbols from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    
    # Download data in one go (this might take a minute, but it's required for backtesting)
    # Using group_by='ticker' for easier parsing
    data = yf.download(symbols, start=start_date, end=end_date, group_by='ticker', progress=False)
    
    # Store daily closes and volumes
    df_close = pd.DataFrame()
    df_volume = pd.DataFrame()
    df_high = pd.DataFrame()
    df_low = pd.DataFrame()
    
    for ticker in symbols:
        try:
            if len(symbols) == 1:
                ticker_data = data
            else:
                ticker_data = data[ticker]
                
            if ticker_data.empty:
                continue
                
            df_close[ticker] = ticker_data['Close']
            df_volume[ticker] = ticker_data['Volume']
            df_high[ticker] = ticker_data['High']
            df_low[ticker] = ticker_data['Low']
        except KeyError:
            continue
            
    df_close.dropna(how='all', inplace=True)
    df_volume.dropna(how='all', inplace=True)
    df_high.dropna(how='all', inplace=True)
    df_low.dropna(how='all', inplace=True)
    
    return df_close, df_volume, df_high, df_low

def calculate_monthly_scores(df_close, df_volume, df_high, df_low):
    """
    Simulates the scanner logic historically across all months.
    """
    # 1. Structural Indicators
    # Daily 50 SMA
    sma_50_d = df_close.rolling(window=50).mean()
    
    # Daily Volume Ratio
    vol_5 = df_volume.rolling(window=5).mean()
    vol_50 = df_volume.rolling(window=50).mean()
    vol_ratio = vol_5 / vol_50.replace(0, np.nan)
    
    # ATR Expansion
    high_low = df_high - df_low
    high_close = np.abs(df_high - df_close.shift(1))
    low_close = np.abs(df_low - df_close.shift(1))
    
    tr_frames = [high_low, high_close, low_close]
    true_range = pd.concat([df.stack() for df in tr_frames], axis=1).max(axis=1).unstack()
    
    atr_14 = true_range.rolling(window=14).mean()
    atr_50_avg = atr_14.rolling(window=50).mean()
    atr_ratio = atr_14 / atr_50_avg.replace(0, np.nan)
    
    # 1-Month Return (approx 21 days)
    return_1m = df_close.pct_change(periods=21)
    
    # Weekly 10 SMA Alignment
    # Resample to weekly
    weekly_close = df_close.resample('W').last()
    sma_10_w = weekly_close.rolling(window=10).mean()
    
    # Monthly 10 SMA Alignment
    monthly_close = df_close.resample('ME').last()
    sma_10_m = monthly_close.rolling(window=10).mean()
    
    # Reindex weekly/monthly back to daily to compare
    sma_10_w_daily = sma_10_w.reindex(df_close.index, method='ffill')
    sma_10_m_daily = sma_10_m.reindex(df_close.index, method='ffill')
    
    # Alignment masks
    aligned_d = df_close > sma_50_d
    aligned_w = df_close > sma_10_w_daily
    aligned_m = df_close > sma_10_m_daily
    
    aligned_momentum = aligned_d & aligned_w & aligned_m
    
    # 2. Get End of Month Dates for Rebalancing
    # We rebalance on the last available trading day of each month in our dataset
    # ensuring we capture the most recent days like Feb 28th
    valid_dates = df_close.groupby([df_close.index.year, df_close.index.month]).apply(lambda x: x.index[-1]).values
    valid_dates = pd.to_datetime(valid_dates)
    
    rebalance_data = {}
    
    for date in valid_dates:
        # Cross-sectional Scoring for this date
        try:
            date_idx = df_close.index.get_loc(date)
        except KeyError:
            continue
            
        ret_row = return_1m.iloc[date_idx]
        vol_r_row = vol_ratio.iloc[date_idx]
        atr_r_row = atr_ratio.iloc[date_idx]
        aligned_row = aligned_momentum.iloc[date_idx]
        
        # Valid universe: must not be NaN in returns
        valid_mask = ret_row.notna() & aligned_row.notna()
        
        if valid_mask.sum() < 10:
            continue # Skip if not enough data yet
            
        ret_val = ret_row[valid_mask]
        vol_val = vol_r_row[valid_mask].fillna(0)
        atr_val = atr_r_row[valid_mask].fillna(0)
        aligned_val = aligned_row[valid_mask]
        
        def zscore(s):
            if len(s) == 0 or s.std() == 0: return s * 0
            clipped = s.clip(lower=s.quantile(0.01), upper=s.quantile(0.99))
            return (clipped - clipped.mean()) / clipped.std()
            
        z_ret = zscore(ret_val)
        z_vol = zscore(vol_val)
        z_atr = zscore(atr_val)
        
        score = (z_ret * 0.4) + (z_vol * 0.3) + (z_atr * 0.3)
        
        # Combine into DataFrame
        day_df = pd.DataFrame({
            'Score': score,
            'Aligned': aligned_val,
            'Close': df_close.iloc[date_idx][valid_mask]
        })
        
        rebalance_data[date] = day_df
        
    return rebalance_data, df_close

def run_backtest(rebalance_data, df_close, initial_capital=500000, max_positions=15, max_dd_target=0.10):
    """
    Simulates monthly rebalancing on the following day open (we approximate using the current day close, 
    since we are scanning at end-of-day). Assumes equal weight.
    """
    portfolio_value = [initial_capital]
    dates = list(rebalance_data.keys())
    
    if not dates:
        print("No rebalance data available.")
        return None, None
        
    # Start on the first valid rebalance date
    start_idx = 0
    # Find subset of daily dates from first rebalance to end
    daily_dates = df_close.index[df_close.index >= dates[start_idx]]
    
    portfolio_equity = pd.Series(index=daily_dates, dtype=float)
    portfolio_equity.iloc[0] = initial_capital
    
    current_holdings = {} # ticker -> shares
    holding_costs = {} # ticker -> entry price
    cash = initial_capital
    trade_logs = []
    cash = initial_capital
    
    print(f"Running backtest from {dates[0].strftime('%Y-%m-%d')} to {daily_dates[-1].strftime('%Y-%m-%d')}")
    
    rebalance_idx = 0
    
    for i, current_date in enumerate(daily_dates):
        # 1. Update Portfolio Value based on current day's close
        current_value = cash
        for ticker, shares in list(current_holdings.items()):
            if pd.notna(df_close.loc[current_date, ticker]):
                current_value += shares * df_close.loc[current_date, ticker]
            else:
                # If delisted or no data, assume previous known value
                # (Simplified for this model)
                pass 
                
        portfolio_equity.loc[current_date] = current_value
        
        # 2. Check if today is a rebalance day
        if rebalance_idx < len(dates) and current_date == dates[rebalance_idx]:
            # Perform Rebalance at current close
            day_data = rebalance_data[current_date]
            
            # Sell everything (simplified turnover, assuming 0 friction for now)
            for ticker, shares in current_holdings.items():
                sell_price = df_close.loc[current_date, ticker]
                buy_price = holding_costs.get(ticker, sell_price)
                if buy_price > 0:
                    ret_pct = (sell_price / buy_price) - 1
                else:
                    ret_pct = 0
                
                trade_logs.append({
                    'Date': current_date.strftime('%Y-%m-%d'),
                    'Ticker': ticker,
                    'Action': 'SELL',
                    'Price': sell_price,
                    'Shares': shares,
                    'Return_Pct': ret_pct
                })
                
            cash = current_value
            current_holdings = {}
            holding_costs = {}
            
            # Filter strictly aligned and get top scores
            eligible = day_data[day_data['Aligned'] == True].sort_values('Score', ascending=False)
            top_picks = eligible.head(max_positions)
            
            if not top_picks.empty:
                # Equal weight
                capital_per_stock = cash / len(top_picks)
                for ticker, row in top_picks.iterrows():
                    price = row['Close']
                    if price > 0:
                        shares = int(capital_per_stock / price)
                        current_holdings[ticker] = shares
                        holding_costs[ticker] = price
                        cash -= (shares * price)
                        
                        trade_logs.append({
                            'Date': current_date.strftime('%Y-%m-%d'),
                            'Ticker': ticker,
                            'Action': 'BUY',
                            'Price': price,
                            'Shares': shares,
                            'Return_Pct': 0.0
                        })
                        
            rebalance_idx += 1
            
    # Calculate Metrics
    returns = portfolio_equity.pct_change().dropna()
    total_return = (portfolio_equity.iloc[-1] / portfolio_equity.iloc[0]) - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    
    # Sharpe Ratio (assuming 0 risk-free rate for simplicity, or 6% for INR)
    rf_daily = 0.06 / 252
    excess_returns = returns - rf_daily
    if excess_returns.std() > 0:
        sharpe = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
    else:
        sharpe = 0
        
    # Drawdown
    running_max = portfolio_equity.cummax()
    drawdown = (portfolio_equity - running_max) / running_max
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin().strftime('%Y-%m-%d') if pd.notna(drawdown.idxmin()) else "N/A"
    
    # Win / Loss
    closed_trades = [t['Return_Pct'] for t in trade_logs if t['Action'] == 'SELL']
    winners = len([x for x in closed_trades if x > 0])
    total_closed = len(closed_trades)
    win_rate = (winners / total_closed) if total_closed > 0 else 0
    
    # Yearly Breakdown
    yearly_df = portfolio_equity.resample('YE').last()
    yearly_returns = yearly_df.pct_change()
    yearly_returns.iloc[0] = (yearly_df.iloc[0] / initial_capital) - 1
    yearly_returns.index = yearly_returns.index.year
    
    stats = {
        'Total Return': f"{total_return*100:.2f}%",
        'Annualized Return': f"{annual_return*100:.2f}%",
        'Sharpe Ratio': f"{sharpe:.2f}",
        'Max Drawdown': f"{max_dd*100:.2f}%",
        'Max DD Date': max_dd_date,
        'Win Rate': f"{win_rate*100:.2f}% ({winners}/{total_closed} trades)",
        'Final Equity': f"INR {portfolio_equity.iloc[-1]:,.2f}"
    }
    
    return portfolio_equity, stats, drawdown, trade_logs, yearly_returns

def run_monte_carlo(returns, num_simulations=100, horizon_days=252, initial_capital=500000):
    """
    Runs a simple non-parametric Monte Carlo simulation by resampling historical daily returns.
    """
    simulations = np.zeros((horizon_days, num_simulations))
    
    # Convert series to numpy array for speed
    ret_array = returns.values
    
    for i in range(num_simulations):
        # Randomly sample from historical returns with replacement
        sim_returns = np.random.choice(ret_array, size=horizon_days, replace=True)
        # Calculate cumulative returns
        cum_returns = np.cumprod(1 + sim_returns)
        simulations[:, i] = initial_capital * cum_returns
        
    return simulations

if __name__ == "__main__":
    from universe import get_nifty_500_symbols
    
    symbols = get_nifty_500_symbols()
    # To test locally quickly, we can limit symbols, or run full
    # symbols = symbols[:50] 
    
    df_c, df_v, df_h, df_l = get_historical_data_for_backtest(symbols, years=4)
    print("Calculating historically...")
    reb_data, clean_c = calculate_monthly_scores(df_c, df_v, df_h, df_l)
    
    print("Running backtest...")
    equity, stats, dd, trade_logs, yearly_returns = run_backtest(reb_data, clean_c, initial_capital=500000, max_positions=15)
    
    print("--- Backtest Results ---")
    for k, v in stats.items():
        print(f"{k}: {v}")
        
    print("\n--- Yearly Returns ---")
    print(yearly_returns)
        
    # Save the equity curve for Streamlit
    if equity is not None:
        equity_df = pd.DataFrame({'Equity': equity, 'Drawdown': dd})
        equity_df.to_csv("backtest_equity.csv")
        
        pd.DataFrame(trade_logs).to_csv("trade_logs.csv", index=False)
        yearly_returns.to_csv("yearly_returns.csv", header=['Return'])
        print("Saved detailed metrics to CSVs.")
        
        # Save last date's picks for the UI
        last_date = list(reb_data.keys())[-1]
        last_data = reb_data[last_date]
        last_eligible = last_data[last_data['Aligned'] == True].sort_values('Score', ascending=False).head(15)
        
        # Reset index to make Ticker a column
        last_eligible = last_eligible.reset_index()
        if 'index' in last_eligible.columns:
            last_eligible.rename(columns={'index': 'Ticker'}, inplace=True)
            
        # Add Entry_Date
        last_eligible['Entry_Date'] = last_date.strftime('%Y-%m-%d')
        
        last_eligible.to_csv("model_portfolio_suggested.csv", index=False)
        print("Saved suggested model portfolio to model_portfolio_suggested.csv")
