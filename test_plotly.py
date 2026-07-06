import os
from dotenv import load_dotenv
import pandas as pd
import plotly.graph_objects as go
from data_pipeline import DataPipeline
from ml_model import MLModelPipeline
from backtester import Backtester

load_dotenv()
api_key = os.getenv("APCA_API_KEY_ID")
secret_key = os.getenv("APCA_API_SECRET_KEY")

dp = DataPipeline(api_key, secret_key)
raw_df = dp.fetch_data("TSLA", years=5)
df_features = dp.engineer_features(raw_df)

split_idx = int(len(df_features) * 0.8)
train_df = df_features.iloc[:split_idx]
test_df = df_features.iloc[split_idx:]

ml = MLModelPipeline()
ml.train(train_df)
signals = ml.generate_signals(df_features)

bt = Backtester(initial_capital=100000.0)
_, _, bt_df = bt.run(df_features, signals)

fig_equity = go.Figure()
fig_equity.add_trace(go.Scatter(
    x=bt_df.index, y=bt_df['strategy_equity'],
    mode='lines', name='ML Strategy'
))
fig_equity.add_trace(go.Scatter(
    x=bt_df.index, y=bt_df['buy_hold_equity'],
    mode='lines', name='Buy & Hold'
))

split_date = test_df.index[0]
fig_equity.add_vline(
    x=split_date, line_width=2, line_dash="dash", line_color="red"
)

# Convert to dict to trigger any rendering logic that might crash
fig_equity.to_dict()
print("Plotly code ran successfully!")
