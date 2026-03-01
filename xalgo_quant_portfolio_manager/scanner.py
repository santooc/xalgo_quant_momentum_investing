import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from universe import get_nifty_500_symbols

def calc_atr(df, period=14):
    """Calculates True Range and Average True Range (ATR)."""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    
    return true_range.rolling(window=period).mean()

def run_scan(symbols):
    """
    Downloads historical data for symbols and calculates indicators.
    """
    print(f"Fetching data for {len(symbols)} symbols...")
    
    # Download 2 years of daily data to ensure enough history for 50-day and monthly MAs
    end_date = datetime.today()
    start_date = end_date - timedelta(days=730)
    
    # yf.download returns a multi-index DataFrame if multiple tickers are passed
    # It's faster to download in bulk
    df_dict = {}
    
    # We will process in batches to avoid overwhelming the connection or memory
    batch_size = 50
    results = []
    
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1} ({len(batch)} symbols)")
        
        try:
            # Group by ticker to get dict-like access
            data = yf.download(batch, start=start_date, end=end_date, group_by='ticker', progress=False)
        except Exception as e:
            print(f"Error downloading batch: {e}")
            continue
            
        for ticker in batch:
            try:
                # If only 1 ticker in batch, yf.download format is different
                if len(batch) == 1:
                    ticker_df = data.copy()
                else:
                    ticker_df = data[ticker].copy()
                    
                if ticker_df.empty or len(ticker_df) < 100:
                    continue
                    
                # Drop rows with all NaNs which happens across different exchange holidays sometimes
                ticker_df.dropna(how='all', inplace=True)
                
                # --- Daily Indicators ---
                # Daily 50 SMA
                ticker_df['SMA_50_D'] = ticker_df['Close'].rolling(window=50).mean()
                
                # Volume Average (5 and 50 days)
                ticker_df['Vol_5_D'] = ticker_df['Volume'].rolling(window=5).mean()
                ticker_df['Vol_50_D'] = ticker_df['Volume'].rolling(window=50).mean()
                ticker_df['Vol_Ratio'] = ticker_df['Vol_5_D'] / ticker_df['Vol_50_D'].replace(0, np.nan)
                
                # ATR Expansion
                ticker_df['ATR_14_D'] = calc_atr(ticker_df, 14)
                ticker_df['ATR_50_D_Avg'] = ticker_df['ATR_14_D'].rolling(window=50).mean()
                ticker_df['ATR_Ratio'] = ticker_df['ATR_14_D'] / ticker_df['ATR_50_D_Avg'].replace(0, np.nan)
                
                # 1-Month Return (approx 21 trading days)
                ticker_df['Return_1M'] = ticker_df['Close'].pct_change(periods=21)
                
                # Inval Level: 20-day lowest low
                ticker_df['Inval_Level'] = ticker_df['Low'].rolling(window=20).min()
                
                # --- Weekly Indicators ---
                # Resample to Weekly taking the last Close and sum of Volume, etc.
                weekly_df = ticker_df.resample('W').agg({'Close': 'last'})
                weekly_df['SMA_10_W'] = weekly_df['Close'].rolling(window=10).mean()
                
                # --- Monthly Indicators ---
                # Resample to Monthly
                monthly_df = ticker_df.resample('ME').agg({'Close': 'last'})
                monthly_df['SMA_10_M'] = monthly_df['Close'].rolling(window=10).mean()
                
                # --- Get Latest Values ---
                latest_d = ticker_df.iloc[-1]
                
                # For weekly and monthly, we might be mid-week/mid-month, so we use the latest available
                latest_w = weekly_df.iloc[-1]
                latest_m = monthly_df.iloc[-1]
                
                # Alignment Check
                daily_aligned = latest_d['Close'] > latest_d['SMA_50_D']
                weekly_aligned = latest_w['Close'] > latest_w['SMA_10_W']
                monthly_aligned = latest_m['Close'] > latest_m['SMA_10_M']
                aligned_momentum = daily_aligned and weekly_aligned and monthly_aligned
                
                # Output Dictionary
                results.append({
                    'Ticker': ticker,
                    'Close': latest_d['Close'],
                    'Aligned_Momentum': aligned_momentum,
                    'Vol_Ratio': latest_d['Vol_Ratio'],
                    'ATR_Ratio': latest_d['ATR_Ratio'],
                    'Return_1M': latest_d['Return_1M'],
                    'Inval_Level': latest_d['Inval_Level']
                })
                
            except Exception as e:
                # print(f"Error processing {ticker}: {e}")
                continue
                
    # Convert results to DataFrame
    res_df = pd.DataFrame(results)
    
    if res_df.empty:
        print("No data fetched.")
        return res_df
        
    # --- Scoring ---
    # Cross-sectional Z-scores for scoring
    # We clip outliers to avoid skewed Z-scores
    
    def calc_zscore(series):
        # Clip at 1st and 99th percentile
        clipped = series.clip(lower=series.quantile(0.01), upper=series.quantile(0.99))
        return (clipped - clipped.mean()) / clipped.std()
        
    res_df['Z_Return_1M'] = calc_zscore(res_df['Return_1M'])
    res_df['Z_Vol_Ratio'] = calc_zscore(res_df['Vol_Ratio'])
    res_df['Z_ATR_Ratio'] = calc_zscore(res_df['ATR_Ratio'])
    
    # Composite Score: 40% Momentum (return), 30% Volume burst, 30% Volatility Expansion
    # We fillna with 0 for Z-scores so missing data doesn't break the score
    res_df.fillna({'Z_Return_1M': 0, 'Z_Vol_Ratio': 0, 'Z_ATR_Ratio': 0}, inplace=True)
    res_df['Composite_Score'] = (res_df['Z_Return_1M'] * 0.4) + (res_df['Z_Vol_Ratio'] * 0.3) + (res_df['Z_ATR_Ratio'] * 0.3)
    
    # Sort by score descending
    res_df = res_df.sort_values('Composite_Score', ascending=False).reset_index(drop=True)
    
    # Define boolean filters based on raw rules (for the UI to show visually)
    res_df['Volume_Confirmed'] = res_df['Vol_Ratio'] > 1.5
    res_df['Vol_Expansion'] = res_df['ATR_Ratio'] > 1.2
    
    return res_df

if __name__ == "__main__":
    symbols = get_nifty_500_symbols()
    
    # For a quick test, just use top 20
    # symbols = symbols[:20] 
    
    print(f"Universe size: {len(symbols)}")
    scan_results = run_scan(symbols)
    
    if not scan_results.empty:
        scan_results.to_csv("scan_results.csv", index=False)
        print("Scan complete. Results saved to scan_results.csv")
        print("\nTop 5 Results:")
        print(scan_results.head(5)[['Ticker', 'Close', 'Aligned_Momentum', 'Volume_Confirmed', 'Vol_Expansion', 'Composite_Score']])
