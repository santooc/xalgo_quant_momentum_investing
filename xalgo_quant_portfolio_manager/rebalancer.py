import pandas as pd
import os
from datetime import datetime

class Rebalancer:
    """
    Simulates monthly rebalancing or generates execution logic based on the 
    scanner CSV output.
    """
    def __init__(self, max_holdings=20, sell_threshold_rank=50):
        self.max_holdings = max_holdings
        # If a stock drops below this rank in the composite score, we sell it
        self.sell_threshold_rank = sell_threshold_rank
        
    def generate_orders(self, current_portfolio_csv="portfolio.csv", scanner_results_csv="scan_results.csv"):
        print(f"--- Monthly Rebalance Run: {datetime.now().strftime('%Y-%m-%d')} ---")
        
        # 1. Load Scanner Results
        if not os.path.exists(scanner_results_csv):
            print(f"Error: {scanner_results_csv} not found. Please run the scanner first.")
            return
            
        df_scan = pd.read_csv(scanner_results_csv)
        # Assuming the CSV is already sorted by Composite_Score descending
        
        # 2. Load Current Portfolio (if exists)
        current_holdings = []
        if os.path.exists(current_portfolio_csv):
            df_port = pd.read_csv(current_portfolio_csv)
            current_holdings = df_port['Ticker'].tolist()
        else:
            print(f"No existing portfolio ({current_portfolio_csv}) found. Starting fresh.")
            df_port = pd.DataFrame(columns=['Ticker', 'Entry_Price', 'Inval_Level'])
            
        print(f"Current Holdings Count: {len(current_holdings)}")
        
        orders = []
        
        # 3. Sell Logic
        # We sell if: 
        #   a) The stock is no longer in the scanner top N (e.g., top 50)
        #   b) Momentum is no longer aligned
        #   c) (If we were doing daily tracking, we'd check if Price < Inval_Level)
        
        top_n_tickers = df_scan.head(self.sell_threshold_rank)['Ticker'].tolist()
        
        keepers = []
        for ticker in current_holdings:
            # Check if it still passes rules
            scan_row = df_scan[df_scan['Ticker'] == ticker]
            
            if scan_row.empty:
                print(f"[SELL] {ticker}: Fell out of the universe.")
                orders.append({'Action': 'SELL', 'Ticker': ticker, 'Reason': 'Fell out of universe'})
            else:
                aligned = scan_row.iloc[0]['Aligned_Momentum']
                rank_passes = ticker in top_n_tickers
                
                if not aligned:
                    print(f"[SELL] {ticker}: Momentum Alignment Broken.")
                    orders.append({'Action': 'SELL', 'Ticker': ticker, 'Reason': 'Momentum Broken'})
                elif not rank_passes:
                    print(f"[SELL] {ticker}: Score rank dropped below {self.sell_threshold_rank}.")
                    orders.append({'Action': 'SELL', 'Ticker': ticker, 'Reason': 'Score Dropped'})
                else:
                    keepers.append(ticker)
                    print(f"[HOLD] {ticker}: Remains in top {self.sell_threshold_rank} and aligned.")
                    
        # 4. Buy Logic
        # We want to buy the top-scoring valid stocks up to self.max_holdings
        capital_available_slots = self.max_holdings - len(keepers)
        
        new_buys = []
        
        if capital_available_slots > 0:
            # Filter candidates: Must have Aligned_Momentum = True
            # For strict buying, we could also require Volume_Confirmed or Vol_Expansion
            candidates = df_scan[(df_scan['Aligned_Momentum'] == True) & 
                                 (~df_scan['Ticker'].isin(keepers))]
            
            for _, row in candidates.head(capital_available_slots).iterrows():
                ticker = row['Ticker']
                new_buys.append({
                    'Ticker': ticker,
                    'Entry_Price': row['Close'],
                    'Inval_Level': row['Inval_Level'],
                    'Score': row['Composite_Score']
                })
                orders.append({'Action': 'BUY', 'Ticker': ticker, 'Reason': f"Top Score {row['Composite_Score']:.2f}"})
                print(f"[BUY]  {ticker}: Top ranking candidate.")
        else:
            print("Portfolio full. No new buys.")
            
        # 5. Save the new State
        new_portfolio_data = []
        
        # Add keepers back
        for ticker in keepers:
            # Get old data if we want to preserve entry price, or just update inval
            old_row = df_port[df_port['Ticker'] == ticker].iloc[0]
            new_scan_row = df_scan[df_scan['Ticker'] == ticker].iloc[0]
            
            new_portfolio_data.append({
                'Ticker': ticker,
                'Entry_Price': old_row['Entry_Price'],
                'Inval_Level': new_scan_row['Inval_Level'] # Update trail stop
            })
            
        # Add new buys
        for buy in new_buys:
            new_portfolio_data.append({
                'Ticker': buy['Ticker'],
                'Entry_Price': buy['Entry_Price'],
                'Inval_Level': buy['Inval_Level']
            })
            
        new_df = pd.DataFrame(new_portfolio_data)
        new_df.to_csv(current_portfolio_csv, index=False)
        print(f"\nRebalance complete. Orders generated: {len(orders)}.")
        print(f"New Portfolio saved to {current_portfolio_csv} with {len(new_df)} positions.")

if __name__ == "__main__":
    rebalancer = Rebalancer(max_holdings=5, sell_threshold_rank=10) # Small for testing
    rebalancer.generate_orders()
