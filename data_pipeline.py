import pandas as pd
import pandas_ta as ta
import numpy as np
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
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
            feed=DataFeed.IEX
        )
        
        bars = self.client.get_stock_bars(request_params)
        df = bars.df
        
        # Alpaca returns a MultiIndex (symbol, timestamp). Let's drop the symbol level.
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index(level=0, drop=True)
            
        # Ensure timestamp is timezone-naive or properly handled for pandas_ta
        df.index = df.index.tz_convert(None)
        
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies technical indicators using pandas_ta and manual calculations."""
        if df is None or df.empty:
            return df
            
        df = df.copy()
        
        # 1. Trend: SMA (50-day)
        df['SMA_50'] = ta.sma(df['close'], length=50)
        
        # 2. Trend: MACD
        macd = ta.macd(df['close'])
        # macd returns multiple columns: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        if macd is not None and not macd.empty:
            df = pd.concat([df, macd.iloc[:, 0].rename('MACD')], axis=1)
        else:
            df['MACD'] = np.nan
            
        # 3. Momentum: RSI (14-day)
        df['RSI_14'] = ta.rsi(df['close'], length=14)
        
        # 4. Volatility: Bollinger Bands
        bb = ta.bbands(df['close'], length=20, std=2)
        if bb is not None and not bb.empty:
            # Typically returns BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
            # Let's extract Upper and Lower bands
            df['BB_Lower'] = bb.iloc[:, 0]
            df['BB_Upper'] = bb.iloc[:, 2]
        else:
            df['BB_Lower'] = np.nan
            df['BB_Upper'] = np.nan
            
        # 5. Volume: OBV
        df['OBV'] = ta.obv(df['close'], df['volume'])
        
        # 6. Additional: Log Returns, 20-day Rolling Mean, 20-day Rolling Std
        df['Log_Return'] = np.log(df['close'] / df['close'].shift(1))
        df['Roll_Mean_20'] = df['close'].rolling(window=20).mean()
        df['Roll_Std_20'] = df['close'].rolling(window=20).std()
        
        # Target Variable: 1 if next day's return > 0, else 0
        df['Next_Day_Return'] = df['close'].pct_change().shift(-1)
        df['Target'] = (df['Next_Day_Return'] > 0).astype(int)
        
        # Clean Data (Drop NaNs resulting from indicator lookbacks and shifting)
        df = df.dropna()
        
        return df

    def get_latest_data_for_prediction(self, symbol: str) -> pd.DataFrame:
        """Fetches just enough historical data to generate today's signal."""
        # Need at least 50 days for SMA_50, let's fetch 100 days to be safe
        end_date = datetime.now()
        start_date = end_date - pd.Timedelta(days=150)
        
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date,
            feed=DataFeed.IEX
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
        """Similar to engineer_features, but does NOT drop the last row due to unknown future."""
        if df is None or df.empty:
            return df
            
        df = df.copy()
        df['SMA_50'] = ta.sma(df['close'], length=50)
        macd = ta.macd(df['close'])
        df['MACD'] = macd.iloc[:, 0] if macd is not None and not macd.empty else np.nan
        df['RSI_14'] = ta.rsi(df['close'], length=14)
        bb = ta.bbands(df['close'], length=20, std=2)
        
        if bb is not None and not bb.empty:
            df['BB_Lower'] = bb.iloc[:, 0]
            df['BB_Upper'] = bb.iloc[:, 2]
        else:
            df['BB_Lower'] = np.nan
            df['BB_Upper'] = np.nan
            
        df['OBV'] = ta.obv(df['close'], df['volume'])
        df['Log_Return'] = np.log(df['close'] / df['close'].shift(1))
        df['Roll_Mean_20'] = df['close'].rolling(window=20).mean()
        df['Roll_Std_20'] = df['close'].rolling(window=20).std()
        
        # No Next_Day_Return or Target computed here
        df = df.dropna()
        return df
