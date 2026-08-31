"""Simple baseline that keeps today's AQI as the forecast."""
from __future__ import annotations

import numpy as np
import pandas as pd


class PersistenceBaseline:
    """No-op estimator: prediction = today's AQI, regardless of horizon."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PersistenceBaseline":
        return self  # nothing to learn

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["aqi"].to_numpy()
