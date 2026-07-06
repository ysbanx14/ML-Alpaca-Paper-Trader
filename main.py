import os
from dotenv import load_dotenv

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Load environment variables
load_dotenv()

from data_pipeline import DataPipeline
from ml_model import MLModelPipeline
from backtester import Backtester
from paper_trader import PaperTrader

# Page Configuration
st.set_page_config(page_title="ML Trading System", layout="wide", page_icon="🤖")

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
st.sidebar.header("Configuration")

# Check for environment variables
env_api_key = os.getenv("APCA_API_KEY_ID")
env_secret_key = os.getenv("APCA_API_SECRET_KEY")

if env_api_key and env_secret_key:
    st.sidebar.success("✅ API Keys loaded securely from .env")
    api_key = env_api_key
    secret_key = env_secret_key
else:
    # Masked inputs for API keys fallback
    api_key = st.sidebar.text_input("Alpaca API Key ID", type="password")
    secret_key = st.sidebar.text_input("Alpaca Secret Key", type="password")

ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper().strip()

run_pipeline = st.sidebar.button("🚀 Run Pipeline")

# Helper function to generate Plotly theme
def get_plotly_layout():
    return dict(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

if run_pipeline:
    if not api_key or not secret_key:
        st.error("Please enter both Alpaca API Key and Secret Key in the sidebar.")
        st.stop()
        
    if not ticker:
        st.error("Please enter a valid Ticker Symbol.")
        st.stop()
        
    with st.spinner("Fetching Data and Training Model..."):
        try:
            # 1. Initialize Pipeline
            dp = DataPipeline(api_key, secret_key)
            
            # 2. Fetch Historical Data
            raw_df = dp.fetch_data(ticker, years=5)
            if raw_df.empty:
                st.error(f"No data returned for {ticker}.")
                st.stop()
                
            # 3. Engineer Features
            df_features = dp.engineer_features(raw_df)
            
            # --- NEW: Train/Test Split (80/20 Chronological) ---
            split_idx = int(len(df_features) * 0.8)
            train_df = df_features.iloc[:split_idx]
            test_df = df_features.iloc[split_idx:]
            
            # 4. Train Model ONLY on train_df to avoid look-ahead bias
            ml = MLModelPipeline()
            ml.train(train_df)
            
            # Generate historical signals for the entire dataset (to plot the full curve)
            signals = ml.generate_signals(df_features)
            
            # Generate signals just for the test set (to calculate pure OOS metrics)
            test_signals = ml.generate_signals(test_df)
            
            # 5. Run Backtest
            bt = Backtester(initial_capital=100000.0)
            
            # Full curve backtest (for visualization only)
            _, _, bt_df = bt.run(df_features, signals)
            
            # Out-Of-Sample backtest (for accurate metrics)
            oos_strat_metrics, oos_bh_metrics, _ = bt.run(test_df, test_signals)
            
            # -------------------------------------------------------------
            # SECTION 1: Model Diagnostics
            # -------------------------------------------------------------
            st.markdown("---")
            st.header("🧠 1. Model Diagnostics (PCA & Features)")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.info(f"**Random Forest Classifier**")
                st.metric("PCA Components Kept (>= 80% Var)", ml.n_components_kept)
                st.metric("Total Original Features", len(ml._get_feature_cols(df_features)))
            
            with col2:
                # Plotly Chart for Cumulative Variance
                fig_pca = go.Figure()
                fig_pca.add_trace(go.Scatter(
                    y=ml.cumulative_variance_,
                    mode='lines+markers',
                    name='Cumulative Variance',
                    line=dict(color='#00ff99', width=2)
                ))
                # Add horizontal line at 80%
                fig_pca.add_hline(y=0.80, line_dash="dash", line_color="red", annotation_text="80% Threshold")
                
                fig_pca.update_layout(
                    title="PCA Cumulative Explained Variance",
                    xaxis_title="Number of Components",
                    yaxis_title="Cumulative Variance Explained",
                    **get_plotly_layout()
                )
                st.plotly_chart(fig_pca, use_container_width=True)
                
            # -------------------------------------------------------------
            # SECTION 2: Backtest Results
            # -------------------------------------------------------------
            st.markdown("---")
            st.header("📊 2. Backtest Results")
            
            # Metrics Table
            st.subheader("Out-Of-Sample Performance (Last 20% of data)")
            metrics_df = pd.DataFrame([oos_strat_metrics, oos_bh_metrics], index=["ML Strategy (OOS)", "Buy & Hold (OOS)"]).T
            st.dataframe(metrics_df.style.highlight_max(axis=1, color='#2e4034'), use_container_width=True)
            
            # Interactive Equity Curve
            fig_equity = go.Figure()
            fig_equity.add_trace(go.Scatter(
                x=bt_df.index, y=bt_df['strategy_equity'],
                mode='lines', name='ML Strategy', line=dict(color='#1CB5E0', width=2)
            ))
            fig_equity.add_trace(go.Scatter(
                x=bt_df.index, y=bt_df['buy_hold_equity'],
                mode='lines', name='Buy & Hold', line=dict(color='#ff9900', width=2, dash='dot')
            ))
            fig_equity.update_layout(
                title=f"Full Equity Curve Comparison ({ticker})",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                legend=dict(x=0.01, y=0.99),
                **get_plotly_layout()
            )
            
            # Add vertical line for Train/Test Split
            split_date = test_df.index[0]
            fig_equity.add_vline(
                x=split_date, line_width=2, line_dash="dash", line_color="red",
                annotation_text="← In-Sample | Out-Of-Sample →", annotation_position="top left"
            )
            
            st.plotly_chart(fig_equity, use_container_width=True)
            
            # -------------------------------------------------------------
            # SECTION 3: Live Paper Trading
            # -------------------------------------------------------------
            st.markdown("---")
            st.header("⚡ 3. Live Paper Trading")
            
            # Fetch latest data to predict today's signal
            latest_df = dp.get_latest_data_for_prediction(ticker)
            if not latest_df.empty:
                today_signal, today_prob = ml.predict_today_signal(latest_df)
                
                signal_text = "🟢 LONG (Buy)" if today_signal == 1 else "⚪ FLAT (Sell/Cash)"
                prob_pct = today_prob * 100
                
                st.markdown(f"### Current Signal for {ticker}: **{signal_text}**")
                st.progress(today_prob, text=f"Probability of Upward Move: {prob_pct:.2f}%")
                
                st.markdown('<div class="disclaimer">⚠️ PAPER TRADING ONLY - NO REAL MONEY. This will execute an API call to Alpaca Paper Trading.</div>', unsafe_allow_html=True)
                
                if st.button("Submit Paper Trade"):
                    with st.spinner("Executing Trade..."):
                        pt = PaperTrader(api_key, secret_key)
                        result = pt.execute_trade(ticker, today_signal)
                        if "SUCCESS" in result:
                            st.success(result)
                        elif "SKIPPED" in result:
                            st.info(result)
                        else:
                            st.error(result)
            else:
                st.warning("Could not fetch latest data for real-time prediction.")
                
        except Exception as e:
            st.error(f"An error occurred during pipeline execution: {str(e)}")
