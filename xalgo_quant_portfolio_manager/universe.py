import pandas as pd
import requests
import io

def get_nifty_500_symbols():
    """
    Fetches the Nifty 500 symbols from NSE India website.
    Returns a list of Yahoo Finance compatible tickers (e.g., 'RELIANCE.NS').
    """
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        
        # Extract the 'Symbol' column and append '.NS' for Yahoo Finance
        symbols = df['Symbol'].tolist()
        yf_symbols = [f"{sym}.NS" for sym in symbols]
        
        return yf_symbols
    except Exception as e:
        print(f"Error fetching Nifty 500 list from NSE: {e}")
        print("Falling back to a smaller sample list.")
        # Fallback to a small list for testing if the NSE site blocks the request
        return [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", 
            "INFY.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS",
            "BAJFINANCE.NS", "LART.NS", "HINDUNILVR.NS", "AXISBANK.NS",
            "KOTAKBANK.NS", "LT.NS", "TATAMOTORS.NS"
        ]

if __name__ == "__main__":
    symbols = get_nifty_500_symbols()
    print(f"Fetched {len(symbols)} symbols. First 5: {symbols[:5]}")
