# Quantitative ML Trading System

> An interactive, stateful machine learning execution terminal built for sophisticated quantitative portfolio management.

This application is a complete, end-to-end algorithmic trading platform. It seamlessly bridges the gap between rigorous quantitative research and active portfolio management, allowing users to rigorously backtest machine learning strategies and dynamically deploy them into a stateful, live paper-trading environment via the Alpaca API.

---

## Assignment Compliance Check

This system has been engineered to strictly satisfy all project requirements:

| Requirement | Implementation & Compliance Details |
| :--- | :--- |
| **5 Years of Historical Data** | Fetches exactly 5 years of daily, **split-adjusted** IEX data via Alpaca to prevent artificial drawdowns on corporate actions. |
| **Ticker Selection** | Users can dynamically input any US equity ticker via the interactive Streamlit sidebar. |
| **6 Technical Indicators** | Engineered using pure native Pandas math (avoiding library deprecation issues). Covers Momentum (RSI, MACD), Volatility (Bollinger Bands), Volume (OBV), and Trend (SMA_50, SMA_200). |
| **PCA Dimensionality Reduction** | Implements `StandardScaler` followed by PCA, dynamically keeping only the components necessary to explain **>= 80%** of the cumulative variance. |
| **Trained Classifier** | Utilizes a strictly regularized `GradientBoostingClassifier` (restricted `max_depth`, `n_estimators`, `subsample`) to combat overfitting while preserving OOS predictive confidence. |
| **0.6 Probability Barrier** | Strict threshold enforcement: Signals trigger a Long (1) only if `P(Upward Move) > 0.6`, else Flat (0). |
| **Strict 1% Portfolio Sizing** | Trade execution strictly limits equity allocation to exactly 1% of the total account equity per position. |
| **$100k Initial Capital Backtest** | Vectorized OOS backtesting engine evaluates the ML strategy against a Buy & Hold baseline starting with exactly $100,000. |
| **7 Core Performance Metrics** | Reports Total Return, Annualized Return, Max Drawdown, Sharpe Ratio, Sortino Ratio, Win Rate, and Total Trades. |
| **Live Paper Execution** | Connects to Alpaca's Paper API to execute dynamic fractional (notional) and whole-integer (qty) fallback orders. |
| **Mandatory Warning String** | UI explicitly states: *"PAPER TRADING ONLY - NO REAL MONEY. This will execute an API call to Alpaca Paper Trading."* |

---

## System Architecture & Modular File Overview

The codebase is strictly modularized into distinct operational layers:

1. **`main_3.py` (The Execution Terminal)**
   - The Streamlit front-end featuring a dynamic, multi-view layout: **"Research & Deploy"** for backtesting, **"Live Portfolio Tracking"** for monitoring active deployments, and **"Auto-Pilot Logs"** for real-time execution status tracking.
2. **`data_pipeline_2.py` (Data Engineering)**
   - Interfaces with the Alpaca API for fetching 5-year split-adjusted bars. Handles all native Pandas technical indicator calculations and target variable alignment (predicting $T+1$ returns).
3. **`ml_model_2.py` (The Quant Brain)**
   - Manages the entire Scikit-Learn pipeline. Scales features, captures >80% variance via PCA, and trains the robust `GradientBoostingClassifier`.
4. **`backtester_3.py` (Performance Evaluation)**
   - A highly optimized, vectorized backtesting engine that simulates historical equity curves and computes the 7 core quantitative metrics.
5. **`paper_trader_2.py` (Order Execution Engine)**
   - Interfaces with Alpaca's Paper Trading client. Features robust fallback logic (reverting to integer share quantities if fractional notional buying is rejected) and cleanly captures out-of-hours API responses.
6. **`portfolio_manager.py` (State Preservation Layer)**
   - Utilizes a local JSON ledger (`tracked_tickers.json`) to persist an unlimited number of deployed models across browser refreshes and app restarts.

---

## Installation & Environment Configuration

### 1. Install Dependencies
Ensure you have Python 3.9+ installed, then run:
```bash
pip install streamlit pandas numpy scikit-learn plotly alpaca-py python-dotenv streamlit-autorefresh
```

### 2. Configure API Keys
The system requires an Alpaca Trading API account (Paper). In the root directory of the project, create a file named `.env` and populate it with your keys:

```env
# .env
APCA_API_KEY_ID=your_alpaca_paper_key_here
APCA_API_SECRET_KEY=your_alpaca_paper_secret_here
```
*(Alternatively, keys can be securely entered directly via the Streamlit sidebar upon launch).*

---

## Workflow & Operational Manual

Launch the application terminal by running:
```bash
streamlit run main_3.py
```

### Phase 1: Research & Deploy
1. Open the **Navigation** sidebar and select **Research & Deploy**.
2. Input a target ticker (e.g., `AMZN`) and click **Run Pipeline**.
3. Review the Model Diagnostics (PCA variance capture) and the Backtest Results (Log-scaled Equity Curves and the 7 Core Metrics).
4. If the Out-Of-Sample performance is satisfactory, scroll to Section 3 and click **Deploy ML Strategy** to permanently save the ticker to your live portfolio state. (You may deploy an unlimited number of tickers).

### Phase 2: Live Portfolio Tracking
1. Switch the sidebar navigation to **Live Portfolio Tracking**.
2. Select any deployed ticker from the dropdown to instantly view its live Alpaca position metrics (Shares, Avg Entry, Unrealized P&L), today's actionable ML probability, and historical trade execution logs.
3. To liquidate the position and remove it from the ledger, click **Stop Tracking**.

### Phase 3: The Autopilot Engine & Execution Board
The application features a non-blocking, infinite background threading execution engine.
1. In the sidebar under **Auto-Pilot Configuration**, adjust the polling interval and turn the **Enable Auto-Pilot Trading** toggle to True.
2. The system will now continuously evaluate and trade all deployed models seamlessly in a dedicated background thread without interrupting the user interface.
3. Switch the sidebar navigation to **Auto-Pilot Logs** to monitor the live Execution Status Board. This Pandas DataFrame natively visualizes the background thread's progress, displaying row-by-row updates of the Alpaca API responses.

### Manual Testing & Override (For Video Demos)
If you need to guarantee an API execution for a demonstration (or if you wish to override the ML prediction):
1. Navigate to the **Live Portfolio Tracking** tab.
2. Open the **Manual Trade Controls (Troubleshooting)** expander.
3. Use the **Force Market Buy** or **Force Market Sell** buttons to bypass the ML logic and explicitly force an order through the Alpaca API.
