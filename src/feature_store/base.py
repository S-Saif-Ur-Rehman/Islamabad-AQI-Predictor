"""Feature store interface for local and Hopsworks backends."""
from __future__ import annotations

import abc

import pandas as pd


class FeatureStore(abc.ABC):
    @abc.abstractmethod
    def insert(self, feature_group_name: str, df: pd.DataFrame, primary_key: list[str]) -> None:
        """Upsert rows into a feature group, deduplicating on primary_key."""

    @abc.abstractmethod
    def read(self, feature_group_name: str) -> pd.DataFrame:
        """Read the full contents of a feature group as a DataFrame."""

    @abc.abstractmethod
    def get_training_data(self, feature_group_name: str) -> pd.DataFrame:
        """Read training dataset (features + labels) via Feature View / store."""


def get_feature_store() -> "FeatureStore":
    from config import config

    if config.USE_HOPSWORKS:
        from src.feature_store.hopsworks_store import HopsworksFeatureStore

        return HopsworksFeatureStore()
    from src.feature_store.local_store import LocalFeatureStore

    return LocalFeatureStore()
