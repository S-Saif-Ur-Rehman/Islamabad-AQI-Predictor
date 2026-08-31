"""Local feature store that saves tables as parquet files."""
from __future__ import annotations

import pandas as pd
import re

from config import config
from src.feature_store.base import FeatureStore


class LocalFeatureStore(FeatureStore):
    def __init__(self):
        config.LOCAL_STORE_DIR.mkdir(parents=True, exist_ok=True)

    def _path(self, feature_group_name: str):
        if not re.fullmatch(r"aqi_[a-z0-9_]+", feature_group_name):
            raise ValueError("Invalid feature group name")
        return config.LOCAL_STORE_DIR / f"{feature_group_name}.parquet"

    def insert(self, feature_group_name: str, df: pd.DataFrame, primary_key: list[str]) -> None:
        path = self._path(feature_group_name)
        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=primary_key, keep="last")
        else:
            combined = df.drop_duplicates(subset=primary_key, keep="last")

        sort_col = "timestamp" if "timestamp" in combined.columns else (
            "date" if "date" in combined.columns else None
        )
        if sort_col:
            combined = combined.sort_values(sort_col)

        combined.to_parquet(path, index=False)

    def read(self, feature_group_name: str) -> pd.DataFrame:
        path = self._path(feature_group_name)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def get_training_data(self, feature_group_name: str) -> pd.DataFrame:
        return self.read(feature_group_name)
