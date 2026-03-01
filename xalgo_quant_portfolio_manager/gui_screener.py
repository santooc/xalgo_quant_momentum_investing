import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Nifty 500 Scanner & Backtester", layout="wide")

st.title("🚀 Nifty 500 Systematic Momentum Portfolio")
st.markdown("A quantitative, monthly-rebalanced momentum scanner and backtester.")

# Tabs
tab1, tab2, tab3 = st.tabs(["📈 Backtest Performance (4-Year)", "🎯 Suggested Model Portfolio", "🔥 Daily Heatmap Screener"])

# Helper function to load data safely
@st.cache_data
def load_data(filename):
    if os.path.exists(filename):
        # Read the first column as index/dates for equity
        if "equity" in filename:
            df = pd.read_csv(filename)
            df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            return df
        return pd.read_csv(filename)
    return pd.DataFrame()

# ----------------- TAB 1: BACKTEST PERFORMANCE -----------------
with tab1:
    st.header("Historical Portfolio Simulation")
    st.markdown("INR 5 Lakhs Initial Capital | Monthly Rebalance | Equal Weight Top 15")
    
    equity_df = load_data('backtest_equity.csv')
    
    if equity_df.empty:
        st.warning("`backtest_equity.csv` not found. Please run `python backtester.py` first.")
    else:
        # Calculate summary stats directly from equity curve
        initial_cap = equity_df['Equity'].iloc[0]
        final_cap = equity_df['Equity'].iloc[-1]
        total_ret = (final_cap / initial_cap) - 1
        
        daily_ret = equity_df['Equity'].pct_change().dropna()
        ann_ret = (1 + total_ret) ** (252 / len(daily_ret)) - 1
        
        # Risk-free approx 6%
        rf = 0.06 / 252 
        sharpe = np.sqrt(252) * ( (daily_ret.mean() - rf) / daily_ret.std() ) if daily_ret.std() > 0 else 0
        
        max_dd = equity_df['Drawdown'].min()
        
        # Calculate additional metrics
        win_rate_str = "N/A"
        trade_logs = load_data('trade_logs.csv')
        yearly_df = load_data('yearly_returns.csv')
        
        if not trade_logs.empty and 'Action' in trade_logs.columns:
            closed = trade_logs[trade_logs['Action'] == 'SELL']
            if not closed.empty and 'Return_Pct' in closed.columns:
                winners = len(closed[closed['Return_Pct'] > 0])
                win_rate_str = f"{(winners / len(closed)) * 100:.1f}%"
                
        # Max DD date
        running_max = equity_df['Equity'].cummax()
        dd_series = (equity_df['Equity'] - running_max) / running_max
        max_dd_date = dd_series.idxmin().strftime('%Y-%m-%d') if pd.notna(dd_series.idxmin()) else "N/A"
        
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Final Capital", f"₹ {final_cap:,.0f}")
        c2.metric("Ann. Return", f"{ann_ret*100:.2f} %")
        c3.metric("Max Drawdown", f"{max_dd*100:.2f} %")
        c4.metric("Sharpe Ratio", f"{sharpe:.2f}")
        c5.metric("Win Rate", win_rate_str)
        c6.metric("Max DD Date", max_dd_date)
        
        st.divider()
        
        if not yearly_df.empty:
            # Safely plot yearly breakdown
            yr_col = yearly_df.columns[0]
            ret_col = yearly_df.columns[1]
            fig_yr = go.Figure(data=[
                go.Bar(x=yearly_df[yr_col].astype(str), y=yearly_df[ret_col]*100,
                       text=(yearly_df[ret_col]*100).round(1).astype(str) + '%',
                       textposition='auto',
                       marker_color=np.where(yearly_df[ret_col] > 0, 'green', 'indianred'))
            ])
            fig_yr.update_layout(height=350, title="Year-by-Year Breakdown (%)", xaxis_type='category', 
                                 margin=dict(t=40, b=0), yaxis_title="Return (%)")
            st.plotly_chart(fig_yr, use_container_width=True)
        
        # Plotly Equity & Drawdown Chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, row_heights=[0.7, 0.3])
                            
        fig.add_trace(go.Scatter(x=equity_df.index, y=equity_df['Equity'],
                                 name="Portfolio Value (₹)", line=dict(color='green', width=2)),
                      row=1, col=1)
                      
        fig.add_trace(go.Scatter(x=equity_df.index, y=equity_df['Drawdown'] * 100,
                                 name="Drawdown (%)", fill='tozeroy', line=dict(color='red', width=1)),
                      row=2, col=1)
                      
        fig.update_layout(height=600, title_text="Equity Curve & Drawdowns", showlegend=False)
        fig.update_yaxes(title_text="Portfolio Value (INR)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)
        
        
        if not trade_logs.empty:
            with st.expander("📜 View Detailed Trade History & Portfolio Rebalancing Logs"):
                # format dataframe for better viewing
                fmt_df = trade_logs.copy()
                if 'Return_Pct' in fmt_df.columns:
                    fmt_df['Return_Pct'] = (fmt_df['Return_Pct'] * 100).round(2).astype(str) + ' %'
                if 'Price' in fmt_df.columns:
                    fmt_df['Price'] = fmt_df['Price'].round(2)
                st.dataframe(fmt_df, use_container_width=True)
                
        # --- MONTE CARLO ---
        st.subheader("Monte Carlo Simulation (1-Year Forward Horizon)")
        st.markdown("Simulating 100 random paths of future 252 trading days based on historical daily return distribution.")
        
        if st.button("Run Monte Carlo Analysis"):
            ret_vals = daily_ret.values
            num_paths = 100
            days = 252
            
            sim_paths = np.zeros((days, num_paths))
            for i in range(num_paths):
                sim_returns = np.random.choice(ret_vals, size=days, replace=True)
                cum_ret = np.cumprod(1 + sim_returns)
                sim_paths[:, i] = final_cap * cum_ret
                
            x_axis = list(range(days))
            fig_mc = go.Figure()
            # Plot roughly 50 paths to avoid overwhelming browser
            for i in range(50):
                fig_mc.add_trace(go.Scatter(x=x_axis, y=sim_paths[:, i].tolist(), mode='lines', 
                                            line=dict(width=1, color='rgba(0,100,250,0.1)'),
                                            showlegend=False))
            
            # Plot the median path
            median_path = np.median(sim_paths, axis=1)
            fig_mc.add_trace(go.Scatter(x=x_axis, y=median_path.tolist(), mode='lines', 
                                        line=dict(width=3, color='orange'), name='Median Path'))
                                        
            fig_mc.update_layout(height=500, title_text="Simulated Forward Equity Paths",
                                 xaxis_title="Trading Days Forward", yaxis_title="Portfolio Value (INR)")
            st.plotly_chart(fig_mc, use_container_width=True)
            
            # MC Metrics
            p5 = np.percentile(sim_paths[-1, :], 5)
            p95 = np.percentile(sim_paths[-1, :], 95)
            st.info(f"**1-Year Expected Outcome (Median):** ₹ {median_path[-1]:,.0f}")
            st.write(f"**90% Confidence Interval:** ₹ {p5:,.0f} to ₹ {p95:,.0f}")


# ----------------- TAB 2: MODEL PORTFOLIO -----------------
with tab2:
    st.header("Suggested Model Portfolio (Next Rebalance)")
    st.markdown("The algorithm's top selections computed as of the latest market close.")
    
    port_df = load_data('model_portfolio_suggested.csv')
    
    if port_df.empty:
        st.warning("`model_portfolio_suggested.csv` not found. Please run backtester.")
    else:
        # Assuming we allocate 500000 across these
        alloc_per_stock = 500000 / len(port_df)
        
        display_port = port_df.copy()
        
        # Calculate suggested shares safely
        # Ensure Close column exists
        if 'Close' in display_port.columns:
            # Handle potential zeros or NaNs in Close
            display_port['Close'] = display_port['Close'].fillna(1e-9).replace(0, 1e-9)
            display_port['Suggested_Shares'] = (alloc_per_stock / display_port['Close']).astype(int)
            display_port['Allocated_Capital'] = display_port['Suggested_Shares'] * display_port['Close']
            
            # Format display
            cols_to_show = ['Ticker', 'Entry_Date', 'Close', 'Score', 'Suggested_Shares', 'Allocated_Capital']
            # Reorder columns safely: only include columns that exist
            cols_to_show = [c for c in cols_to_show if c in display_port.columns]
            
            st.dataframe(display_port[cols_to_show].style.format({
                'Close': '₹ {:.2f}',
                'Score': '{:.2f}',
                'Allocated_Capital': '₹ {:,.0f}'
            }), use_container_width=True)
            
            st.success(f"**Total Capital Allocated:** ₹ {display_port['Allocated_Capital'].sum():,.0f} (out of ₹ 500,000)")
        else:
            st.dataframe(display_port)

# ----------------- TAB 3: DAILY HEATMAP -----------------
with tab3:
    st.header("Daily Nifty 500 Screener")
    
    df = load_data('scan_results.csv')
    
    if df.empty:
        st.warning("`scan_results.csv` not found.")
    else:
        st.sidebar.markdown("---")
        st.sidebar.header("Heatmap Filters")
        require_momentum = st.sidebar.checkbox("Require Aligned Momentum", value=True)
        require_volume = st.sidebar.checkbox("Require Volume Verification (> 1.5x)", value=False)
        require_volatility = st.sidebar.checkbox("Require Volatility Expansion (> 1.2x ATR)", value=False)
        
        filtered_df = df.copy()
        
        if require_momentum:
            filtered_df = filtered_df[filtered_df['Aligned_Momentum'] == True]
        if require_volume:
            filtered_df = filtered_df[filtered_df['Volume_Confirmed'] == True]
        if require_volatility:
            filtered_df = filtered_df[filtered_df['Vol_Expansion'] == True]
            
        st.write(f"Showing **{len(filtered_df)}** stocks matching criteria.")
        
        def style_dataframe(df_to_style):
            def highlight_bool(val):
                if val is True:
                    return 'background-color: darkgreen; color: white'
                elif val is False:
                    return 'background-color: darkred; color: white'
                return ''
                
            return df_to_style.style.applymap(highlight_bool, subset=['Aligned_Momentum', 'Volume_Confirmed', 'Vol_Expansion'])\
                                .background_gradient(cmap='RdYlGn', subset=['Composite_Score'])\
                                .format(precision=2)

        display_cols = ['Ticker', 'Close', 'Inval_Level', 'Aligned_Momentum', 'Volume_Confirmed', 'Vol_Expansion', 'Composite_Score']
        display_cols = [c for c in display_cols if c in filtered_df.columns]
        
        st.dataframe(style_dataframe(filtered_df[display_cols]), use_container_width=True, height=600)
