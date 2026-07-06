import os
from dotenv import load_dotenv
load_dotenv()
from data_pipeline import DataPipeline
from ml_model import MLModelPipeline
from backtester import Backtester

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
test_signals = ml.generate_signals(test_df)

bt = Backtester(initial_capital=100000.0)
_, _, bt_df = bt.run(df_features, signals)
oos_strat, oos_bh, _ = bt.run(test_df, test_signals)
print("SUCCESS!")
