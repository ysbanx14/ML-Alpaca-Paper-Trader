import pandas as pd
import numpy as np
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed, Adjustment
from datetime import datetime

class DataPipeline:
    def __init__(self, api_key: str, secret_key: str):
        # We use the StockHistoricalDataClient for basic market data
        self.client = StockHistoricalDataClient(api_key, secret_key)
        
    def fetch_data(self, symbol: str, years: int = 5) -> pd.DataFrame:
        """Fetches historical daily data from Alpaca."""
        end_date = datetime.now()
        start_date = end_date - pd.Timedelta(days=365 * years)
        
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date,
            feed=DataFeed.IEX,
            adjustment=Adjustment.ALL
        )
        
        bars = self.client.get_stock_bars(request_params)
        df = bars.df
        
        # Alpaca returns a MultiIndex (symbol, timestamp). Let's drop the symbol level.
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index(level=0, drop=True)
            
        # Ensure timestamp is timezone-naive
        df.index = df.index.tz_convert(None)
        
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies technical indicators using pure native pandas math."""
        if df is None or df.empty:
            return df
            
        df = df.copy()
        
        # 1. Trend: SMA (50-day)
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        
        # 2. Trend: MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
            
        # 3. Momentum: RSI (14-day)
        delta = df['close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = pd.Series(gain, index=df.index).rolling(window=14).mean()
        avg_loss = pd.Series(loss, index=df.index).rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # 4. Volatility: Bollinger Bands
        sma_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        df['BB_Upper'] = sma_20 + (std_20 * 2)
        df['BB_Lower'] = sma_20 - (std_20 * 2)
            
        # 5. Volume: OBV
        direction = np.sign(df['close'].diff())
        direction = direction.fillna(1) # Handle the first row NaN
        df['OBV'] = (direction * df['volume']).cumsum()
        
        # 6. Additional: Log Returns, 20-day Rolling Mean, 20-day Rolling Std
        df['Log_Return'] = np.log(df['close'] / df['close'].shift(1))
        df['Roll_Mean_20'] = df['close'].rolling(window=20).mean()
        df['Roll_Std_20'] = df['close'].rolling(window=20).std()
        
        # Target Variable: 1 if next day's return > 0, else 0
        df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Clean Data (Drop NaNs resulting from indicator lookbacks)
        df = df.dropna()
        
        # Explicitly drop the last row since its "future" target cannot be known
        df = df.iloc[:-1]
        
        return df

    def get_latest_data_for_prediction(self, symbol: str) -> pd.DataFrame:
        """Fetches just enough historical data to generate today's signal."""
        # Need at least 50 days for SMA_50, let's fetch 150 days to be safe
        end_date = datetime.now()
        start_date = end_date - pd.Timedelta(days=150)
        
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date,
            feed=DataFeed.IEX,
            adjustment=Adjustment.ALL
        )
        
        bars = self.client.get_stock_bars(request_params)
        df = bars.df
        
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index(level=0, drop=True)
            
        df.index = df.index.tz_convert(None)
        
        # Engineer features without dropping the last row
        df_featured = self.engineer_features_predict(df)
            
        return df_featured

    def engineer_features_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Similar to engineer_features, but does NOT drop the last row."""
        if df is None or df.empty:
            return df
            
        df = df.copy()
        
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
            
        delta = df['close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain, index=df.index).rolling(window=14).mean()
        avg_loss = pd.Series(loss, index=df.index).rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        sma_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        df['BB_Upper'] = sma_20 + (std_20 * 2)
        df['BB_Lower'] = sma_20 - (std_20 * 2)
            
        direction = np.sign(df['close'].diff()).fillna(1)
        df['OBV'] = (direction * df['volume']).cumsum()
        
        df['Log_Return'] = np.log(df['close'] / df['close'].shift(1))
        df['Roll_Mean_20'] = df['close'].rolling(window=20).mean()
        df['Roll_Std_20'] = df['close'].rolling(window=20).std()
        
        # No Target computed here.
        df = df.dropna()
        
        return df
