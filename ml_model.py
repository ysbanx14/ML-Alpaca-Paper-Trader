import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier

class MLModelPipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        # Initialize PCA without fixed n_components. We'll set it during fit.
        self.pca = PCA()
        self.model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            min_samples_leaf=2,
            learning_rate=0.1,
            subsample=0.9,
            max_features='sqrt',
            random_state=42
        )
        
        self.n_components_kept = 0
        self.explained_variance_ratio_ = None
        self.cumulative_variance_ = None

    def _get_feature_cols(self, df: pd.DataFrame) -> list:
        # Exclude non-feature columns
        exclude = ['Next_Day_Return', 'Target', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'trade_count']
        return [col for col in df.columns if col not in exclude]

    def train(self, df: pd.DataFrame):
        """Trains the ML pipeline (Scaler -> PCA -> RF)"""
        features = self._get_feature_cols(df)
        X = df[features].values
        y = df['Target'].values

        # 1. Scaling
        X_scaled = self.scaler.fit_transform(X)

        # 2. PCA
        # Fit PCA on scaled data to determine how many components explain >= 80% variance
        self.pca.fit(X_scaled)
        cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
        
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.cumulative_variance_ = cumulative_variance
        
        # Find index where cumulative variance crosses 80% (0.80)
        num_components = np.argmax(cumulative_variance >= 0.80) + 1
        self.n_components_kept = num_components
        
        # Re-initialize PCA with the selected number of components
        self.pca = PCA(n_components=self.n_components_kept)
        X_pca = self.pca.fit_transform(X_scaled)

        # 3. Train Model
        self.model.fit(X_pca, y)
        
    def predict_probabilities(self, df: pd.DataFrame) -> np.ndarray:
        """Returns the probability of the Long (1) class."""
        features = self._get_feature_cols(df)
        X = df[features].values
        
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        
        probs = self.model.predict_proba(X_pca)
        return probs[:, 1]  # Return probability of class 1

    def generate_signals(self, df: pd.DataFrame, threshold: float = 0.6) -> pd.Series:
        """Generates binary signals based on probability threshold."""
        probs = self.predict_probabilities(df)
        signals = (probs > threshold).astype(int)
        return pd.Series(signals, index=df.index, name='Signal')

    def predict_today_signal(self, df: pd.DataFrame, threshold: float = 0.6) -> tuple:
        """Predicts signal for the very last row (today)."""
        probs = self.predict_probabilities(df)
        today_prob = probs[-1]
        signal = 1 if today_prob > threshold else 0
        return signal, today_prob
