"""Build the daily feature table from raw hourly data."""
from __future__ import annotations

import logging

from config import config
from src.features.feature_engineering import add_forecast_targets, build_daily_features
from src.feature_store.base import get_feature_store
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def run():
    store = get_feature_store()
    hourly_df = store.read(config.RAW_HOURLY_FG)
    if hourly_df.empty:
        logger.warning("No raw hourly data yet — run the feature or backfill pipeline first.")
        return hourly_df

    daily_df = build_daily_features(hourly_df)
    daily_df = add_forecast_targets(daily_df, config.FORECAST_HORIZONS_DAYS)

    store.insert(config.DAILY_FEATURES_FG, daily_df, primary_key=["date"])
    logger.info("Rebuilt daily feature table: %d days", len(daily_df))
    return daily_df


if __name__ == "__main__":
    setup_logging()
    run()
