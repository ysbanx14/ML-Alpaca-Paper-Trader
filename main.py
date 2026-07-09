import os
import traceback
import datetime
from dotenv import load_dotenv
import threading
from streamlit.runtime.scriptrunner import add_script_run_ctx

import streamlit as st
import time
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Load environment variables
load_dotenv()

from data_pipeline import DataPipeline
from ml_model import MLModelPipeline
from backtester import Backtester
from paper_trader import PaperTrader
from portfolio_manager import PortfolioManager

pm = PortfolioManager()

# Page Configuration
st.set_page_config(page_title="ML Trading System", layout="wide", page_icon="🤖")

if 'execution_status' not in st.session_state:
    st.session_state['execution_status'] = {}

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main {
        background-color: #0e1117;
    }
    h1, h2, h3 {
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background: linear-gradient(90deg, #1CB5E0 0%, #000851 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        border: 1px solid #1CB5E0;
    }
    .disclaimer {
        background-color: rgba(255, 75, 75, 0.1);
        border-left: 5px solid #ff4b4b;
        padding: 15px;
        border-radius: 5px;
        margin-top: 20px;
        margin-bottom: 20px;
        font-weight: bold;
        color: #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Quantitative ML Trading System")

# -------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------
st.sidebar.header("Navigation")
app_mode = st.sidebar.radio("Go to", ["Research & Deploy", "Live Portfolio Tracking", "Auto-Pilot Logs"])

st.sidebar.header("Configuration")
env_api_key = os.getenv("APCA_API_KEY_ID")
env_secret_key = os.getenv("APCA_API_SECRET_KEY")

if env_api_key and env_secret_key:
    st.sidebar.success("✅ API Keys loaded securely from .env")
    api_key = env_api_key
    secret_key = env_secret_key
else:
    api_key = st.sidebar.text_input("Alpaca API Key ID", type="password")
    secret_key = st.sidebar.text_input("Alpaca Secret Key", type="password")

# Tracked Tickers & Daily Execution (Global Action)
st.sidebar.markdown("---")
st.sidebar.header("Deployed Models")
tracked = pm.get_tracked_tickers()
active_models = len(tracked)

if api_key and secret_key:
    pt_ui = PaperTrader(api_key, secret_key)
    portfolio_data = pt_ui.get_portfolio_capital()
    allocated_capital = portfolio_data["allocated_capital"]
    total_equity = portfolio_data["total_equity"]
    
    st.sidebar.caption(f"**Capital Allocated:** ${allocated_capital:,.2f} / ${total_equity:,.2f}")
else:
    st.sidebar.caption("API Keys required to fetch allocation data.")

if tracked:
    st.sidebar.write("**Tracked:** " + ", ".join(tracked))
else:
    st.sidebar.write("No models deployed yet.")

st.sidebar.markdown("---")
st.sidebar.header("Mass Deploy")
st.sidebar.write("Deploy strategy to top liquid assets instantly:")
if st.sidebar.button("🚀 Deploy to Top 50 Liquid Stocks"):
    top_50 = ["AAPL", "NVDA", "TSLA", "AMD", "MSFT", "META", "GOOGL", "AMZN", "NFLX", "SPY", "QQQ", "BA", "DIS", "JPM", "V", "MA", "WMT", "JNJ", "PG", "HD", "UNH", "XOM", "CVX", "ABBV", "PFE", "KO", "PEP", "MRK", "TMO", "COST", "CSCO", "MCD", "NKE", "CRM", "ADBE", "TXN", "AVGO", "QCOM", "INTC", "IBM", "HON", "AMGN", "CAT", "GE", "MMM", "GS", "MS", "BLK", "UBER"]
    added_count = 0
    for t_mass in top_50:
        if pm.add_ticker(t_mass):
            added_count += 1
    st.sidebar.success(f"Successfully deployed {added_count} new tickers to the Live Portfolio!")
def run_portfolio_execution(board_placeholder=None):
    if not api_key or not secret_key:
        st.sidebar.error("API Keys missing.")
        return
    if not tracked:
        st.sidebar.warning("No tickers deployed.")
        return
    dp = DataPipeline(api_key, secret_key)
    ml = MLModelPipeline()
    pt = PaperTrader(api_key, secret_key)
    
    tracked_sorted = sorted(tracked)
    
    for t in tracked_sorted:
        if t not in st.session_state['execution_status']:
            st.session_state['execution_status'][t] = {"indicator": "🔴", "message": "-"}
        else:
            st.session_state['execution_status'][t]["indicator"] = "🔴"
            
    if board_placeholder is not None:
        status_df = pd.DataFrame.from_dict(st.session_state['execution_status'], orient='index').reset_index()
        status_df.columns = ["Ticker", "Status", "Latest Action"]
        board_placeholder.dataframe(status_df, use_container_width=True, hide_index=True)
            
    for t in tracked_sorted:
        try:
            time.sleep(0.5)  # Rate Limit Protection
            df = dp.fetch_data(t, years=5)
            if df.empty:
                st.session_state['execution_status'][t]["indicator"] = "🟢"
                st.session_state['execution_status'][t]["message"] = "ERROR - No data"
                if board_placeholder is not None:
                    status_df = pd.DataFrame.from_dict(st.session_state['execution_status'], orient='index').reset_index()
                    status_df.columns = ["Ticker", "Status", "Latest Action"]
                    board_placeholder.dataframe(status_df, use_container_width=True, hide_index=True)
                continue
            df_features = dp.engineer_features(df)
            split_idx = int(len(df_features) * 0.8)
            train_df = df_features.iloc[:split_idx]
            ml.train(train_df)
            
            latest_df = dp.get_latest_data_for_prediction(t)
            if not latest_df.empty:
                today_signal, _ = ml.predict_today_signal(latest_df)
                res = pt.execute_trade(t, today_signal, active_models=active_models)
                st.session_state['execution_status'][t]["indicator"] = "🟢"
                st.session_state['execution_status'][t]["message"] = res
        except Exception as e:
            st.session_state['execution_status'][t]["indicator"] = "🟢"
            st.session_state['execution_status'][t]["message"] = f"ERROR - Failed to trade: {e}"
            
        if board_placeholder is not None:
            status_df = pd.DataFrame.from_dict(st.session_state['execution_status'], orient='index').reset_index()
            status_df.columns = ["Ticker", "Status", "Latest Action"]
            board_placeholder.dataframe(status_df, use_container_width=True, hide_index=True)

st.sidebar.markdown("---")
st.sidebar.header("🤖 Auto-Pilot Configuration")
auto_pilot_interval = st.sidebar.slider("Auto-Pilot Check Interval (Minutes)", min_value=1, max_value=60, value=5)
auto_pilot = st.sidebar.toggle("🤖 Enable Auto-Pilot Trading", value=False)

if auto_pilot:
    st.sidebar.warning(f"Auto-Pilot is ACTIVE: Scanning and trading every {auto_pilot_interval} minutes.")

def get_plotly_layout():
    return dict(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

if app_mode == "Research & Deploy":
    ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper().strip()
    
    if st.sidebar.button("🚀 Run Pipeline"):
        st.session_state['run_pipeline_for'] = ticker
        
    if st.session_state.get('run_pipeline_for') == ticker:
        if not api_key or not secret_key:
            st.error("Please enter both Alpaca API Key and Secret Key in the sidebar.")
            st.stop()
            
        with st.spinner("Fetching Data and Training Model..."):
            try:
                dp = DataPipeline(api_key, secret_key)
                raw_df = dp.fetch_data(ticker, years=5)
                if raw_df.empty:
                    st.error(f"No data returned for {ticker}.")
                    st.stop()
                    
                df_features = dp.engineer_features(raw_df)
                split_idx = int(len(df_features) * 0.8)
                train_df = df_features.iloc[:split_idx]
                test_df = df_features.iloc[split_idx:]
                
                ml = MLModelPipeline()
                ml.train(train_df)
                
                signals = ml.generate_signals(df_features)
                test_signals = ml.generate_signals(test_df)
                
                bt = Backtester(initial_capital=100000.0)
                _, _, bt_df = bt.run(df_features, signals)
                oos_strat_metrics, oos_bh_metrics, _ = bt.run(test_df, test_signals)
                
                # SECTION 1: Model Diagnostics
                st.markdown("---")
                st.header("🧠 1. Model Diagnostics (PCA & Features)")
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.info("**Gradient Boosting Classifier**")
                    st.metric("PCA Components Kept (>= 80% Var)", ml.n_components_kept)
                    st.metric("Total Original Features", len(ml._get_feature_cols(df_features)))
                with col2:
                    fig_pca = go.Figure()
                    fig_pca.add_trace(go.Scatter(y=ml.cumulative_variance_, mode='lines+markers', name='Cumulative Variance', line=dict(color='#00ff99', width=2)))
                    fig_pca.add_hline(y=0.80, line_dash="dash", line_color="red", annotation_text="80% Threshold")
                    fig_pca.update_layout(title="PCA Cumulative Explained Variance", xaxis_title="Number of Components", yaxis_title="Cumulative Variance Explained", **get_plotly_layout())
                    st.plotly_chart(fig_pca, use_container_width=True)
                    
                # SECTION 2: Backtest Results
                st.markdown("---")
                st.header("📊 2. Backtest Results")
                st.subheader("Out-Of-Sample Performance (Last 20% of data)")
                metrics_df = pd.DataFrame([oos_strat_metrics, oos_bh_metrics], index=["ML Strategy (OOS)", "Buy & Hold (OOS)"]).T
                st.dataframe(metrics_df.style.highlight_max(axis=1, color='#2e4034'), use_container_width=True)
                
                fig_equity = go.Figure()
                fig_equity.add_trace(go.Scatter(x=bt_df.index, y=bt_df['strategy_equity'], mode='lines', name='ML Strategy', line=dict(color='#1CB5E0', width=2)))
                fig_equity.add_trace(go.Scatter(x=bt_df.index, y=bt_df['buy_hold_equity'], mode='lines', name='Buy & Hold', line=dict(color='#ff9900', width=2, dash='dot')))
                fig_equity.update_layout(title=f"Full Equity Curve Comparison ({ticker})", xaxis_title="Date", yaxis_title="Portfolio Value ($)", legend=dict(x=0.01, y=0.99), **get_plotly_layout())
                fig_equity.update_yaxes(type="log", title_text="Portfolio Value (Log Scale, $)")
                
                split_date = test_df.index[0]
                fig_equity.add_vline(x=split_date, line_width=2, line_dash="dash", line_color="red")
                fig_equity.add_annotation(x=split_date, y=1.05, yref="paper", text="← In-Sample | Out-Of-Sample →", showarrow=False, xanchor="left", font=dict(color="red"))
                st.plotly_chart(fig_equity, use_container_width=True)
                
                st.subheader("Out-Of-Sample Normalized Comparison")
                oos_df = bt_df.loc[split_date:].copy()
                start_strat = oos_df['strategy_equity'].iloc[0]
                start_bh = oos_df['buy_hold_equity'].iloc[0]
                oos_strat_rebased = (oos_df['strategy_equity'] / start_strat) * 100000.0
                oos_bh_rebased = (oos_df['buy_hold_equity'] / start_bh) * 100000.0
                fig_oos = go.Figure()
                fig_oos.add_trace(go.Scatter(x=oos_df.index, y=oos_strat_rebased, mode='lines', name='ML Strategy (OOS)', line=dict(color='#1CB5E0', width=2)))
                fig_oos.add_trace(go.Scatter(x=oos_df.index, y=oos_bh_rebased, mode='lines', name='Buy & Hold (OOS)', line=dict(color='#ff9900', width=2, dash='dot')))
                fig_oos.update_layout(title=f"Out-Of-Sample Equity Curve ({ticker})", xaxis_title="Date", yaxis_title="Portfolio Value (Linear Scale, $)", legend=dict(x=0.01, y=0.99), **get_plotly_layout())
                st.plotly_chart(fig_oos, use_container_width=True)
                
                # SECTION 3: Live Paper Trading -> Deploy Strategy
                st.markdown("---")
                st.header("⚡ 3. Deploy Strategy")
                
                latest_df = dp.get_latest_data_for_prediction(ticker)
                if not latest_df.empty:
                    today_signal, today_prob = ml.predict_today_signal(latest_df)
                    signal_text = "🟢 LONG (Buy)" if today_signal == 1 else "⚪ FLAT (Sell/Cash)"
                    prob_pct = today_prob * 100
                    st.markdown(f"### Current Signal for {ticker}: **{signal_text}**")
                    st.progress(today_prob, text=f"Probability of Upward Move: {prob_pct:.2f}%")
                    st.markdown('<div class="disclaimer">Deploying this model will save it to the tracked tickers list for live portfolio execution.</div>', unsafe_allow_html=True)
                    
                    if st.button(f"🟢 Deploy ML Strategy for {ticker}"):
                        if pm.add_ticker(ticker):
                            st.success(f"{ticker} has been successfully deployed to the Live Portfolio!")
                        else:
                            st.info(f"{ticker} is already deployed.")
                            
                    with st.expander("🔧 Troubleshooting & Manual Override"):
                        if st.button("🔴 Force Test Buy Order (Bypass ML)", type="primary"):
                            with st.spinner("Forcing API Trade Execution..."):
                                pt = PaperTrader(api_key, secret_key)
                                # Force a Long signal (1) regardless of the model
                                result = pt.execute_trade(ticker, 1, active_models=active_models) 
                                if "SUCCESS" in result:
                                    st.success(result)
                                else:
                                    st.error(result)
                else:
                    st.warning("Could not fetch latest data for real-time prediction.")
            except Exception as e:
                st.error(f"An error occurred during pipeline execution: {str(e)}")
                st.code(traceback.format_exc(), language="text")

elif app_mode == "Live Portfolio Tracking":
    st.header("📈 Live Portfolio Tracking")
    
    if not tracked:
        st.warning("You have not deployed any ML strategies yet. Go to 'Research & Deploy' to add tickers.")
    else:
        selected_ticker = st.selectbox("Select Deployed Ticker", tracked)
        
        if selected_ticker:
            pt = PaperTrader(api_key, secret_key)
            dp = DataPipeline(api_key, secret_key)
            ml = MLModelPipeline()
            
            with st.spinner(f"Loading live data for {selected_ticker}..."):
                st.subheader(f"Current Position: {selected_ticker}")
                pos = pt.get_position(selected_ticker)
                if pos:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Shares Held", pos['shares'])
                    col2.metric("Avg Entry Price", f"${pos['avg_entry_price']:.2f}")
                    col3.metric("Market Value", f"${pos['market_value']:.2f}")
                    col4.metric("Unrealized P&L", f"${pos['unrealized_pl']:.2f}", f"{pos['unrealized_plpc']*100:.2f}%")
                else:
                    st.info(f"No active position held for {selected_ticker}.")
                
                st.markdown("---")
                
                st.subheader("Today's ML Signal")
                try:
                    df = dp.fetch_data(selected_ticker, years=5)
                    if not df.empty:
                        df_features = dp.engineer_features(df)
                        split_idx = int(len(df_features) * 0.8)
                        ml.train(df_features.iloc[:split_idx])
                        latest_df = dp.get_latest_data_for_prediction(selected_ticker)
                        today_signal, today_prob = ml.predict_today_signal(latest_df)
                        signal_text = "🟢 LONG (Buy)" if today_signal == 1 else "⚪ FLAT (Sell/Cash)"
                        st.write(f"**Action:** {signal_text} (Probability: {today_prob*100:.2f}%)")
                except Exception as e:
                    st.error(f"Error calculating signal: {e}")
                
                st.markdown("---")
                
                st.subheader("Recent Alpaca Trade Logs")
                logs = pt.get_trade_logs(selected_ticker)
                if logs:
                    log_df = pd.DataFrame(logs)
                    st.dataframe(log_df, use_container_width=True)
                else:
                    st.info("No historical trades found for this ticker.")
                
                st.markdown("---")
                
                with st.expander("🔧 Manual Trade Controls (Troubleshooting)"):
                    man_col1, man_col2 = st.columns(2)
                    with man_col1:
                        if st.button("🟢 Force Market Buy", key=f"buy_{selected_ticker}"):
                            with st.spinner("Forcing API Trade Execution..."):
                                res = pt.execute_trade(selected_ticker, 1, active_models=active_models)
                                if "SUCCESS" in res:
                                    st.success(res)
                                else:
                                    st.error(res)
                    with man_col2:
                        if st.button("🔴 Force Market Sell", key=f"sell_{selected_ticker}"):
                            with st.spinner("Forcing API Trade Execution..."):
                                res = pt.execute_trade(selected_ticker, 0, active_models=active_models)
                                if "SUCCESS" in res:
                                    st.success(res)
                                else:
                                    st.error(res)
                
                if st.button(f"🛑 Stop Tracking {selected_ticker}"):
                    res = pt.execute_trade(selected_ticker, 0, active_models=active_models)
                    pm.remove_ticker(selected_ticker)
                    st.success(f"{selected_ticker} removed. Liquidating position: {res}")
                    try:
                        st.rerun()
                    except:
                        try:
                            st.experimental_rerun()
                        except:
                            st.info("Please refresh the page to update the UI.")

elif app_mode == "Auto-Pilot Logs":
    st.header("🖨️ Auto-Pilot Execution Board")
    
    if st.button("🗑️ Clear Board"):
        st.session_state['execution_status'] = {}
        try:
            st.rerun()
        except:
            st.experimental_rerun()
            
    st.markdown("---")
    
    if not st.session_state.get('execution_status'):
        st.info("No executions recorded yet. Deploy models to begin tracking.")
    else:
        board_placeholder = st.empty()
        status_df = pd.DataFrame.from_dict(st.session_state['execution_status'], orient='index')
        status_df.reset_index(inplace=True)
        status_df.columns = ["Ticker", "Status", "Latest Action"]
        board_placeholder.dataframe(status_df, use_container_width=True, hide_index=True)

if auto_pilot:
    current_tick = st_autorefresh(interval=auto_pilot_interval * 60 * 1000, key="auto_pilot_refresh")
    if 'last_tick' not in st.session_state:
        st.session_state['last_tick'] = 0
    
    if current_tick > st.session_state['last_tick']:
        st.session_state['last_tick'] = current_tick
        board_ph = board_placeholder if 'board_placeholder' in locals() else None
        t = threading.Thread(target=run_portfolio_execution, args=(board_ph,))
        add_script_run_ctx(t)
        t.start()
