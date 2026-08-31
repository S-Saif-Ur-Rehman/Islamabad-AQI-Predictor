"""Backfill historical AQI and weather data for model training."""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from config import config
from src.clients.open_meteo_client import get_historical_weather
from src.clients.openweather_client import OpenWeatherClient
from src.features.aqi_calculator import estimate_overall_aqi
from src.feature_store.base import get_feature_store
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

# Split large date ranges into smaller requests.
CHUNK_DAYS = 30


def _chunk_ranges(start: datetime, end: datetime, chunk_days: int):
    cur = start
    while cur < end:
        chunk_end = min(cur + timedelta(days=chunk_days), end)
        yield cur, chunk_end
        cur = chunk_end


def backfill(days: int) -> pd.DataFrame:
    if not 1 <= days <= config.MAX_BACKFILL_DAYS:
        raise ValueError(f"days must be between 1 and {config.MAX_BACKFILL_DAYS}")
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)

    openweather = OpenWeatherClient()

    pollution_rows: list[dict] = []
    for chunk_start, chunk_end in _chunk_ranges(start, end, CHUNK_DAYS):
        logger.info("Fetching pollution history %s -> %s", chunk_start.date(), chunk_end.date())
        pollution_rows.extend(openweather.get_air_pollution_history(chunk_start, chunk_end))

    if not pollution_rows:
        logger.warning("No historical pollution data returned — nothing to backfill.")
        return pd.DataFrame()

    pollution_df = pd.DataFrame(pollution_rows)

    weather_df = get_historical_weather(start.date(), end.date())

    merged = pd.merge_asof(
        pollution_df.sort_values("timestamp"),
        weather_df.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("1h"),
    )

    merged["aqi"] = merged.apply(lambda r: estimate_overall_aqi(r.get("pm2_5"), r.get("pm10")), axis=1)
    merged = merged.rename(columns={"pm2_5": "pm25"})
    merged["city"] = config.CITY_NAME
    merged = merged.dropna(subset=["aqi"])

    keep_cols = [
        "timestamp", "city", "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
        "temp_c", "humidity_pct", "pressure_hpa", "wind_speed_ms",
    ]
    merged = merged[[c for c in keep_cols if c in merged.columns]]

    store = get_feature_store()
    store.insert(config.RAW_HOURLY_FG, merged, primary_key=["timestamp"])
    logger.info("Backfilled %d hourly rows (%d days) into %s", len(merged), days, config.RAW_HOURLY_FG)
    return merged


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Backfill historical AQI + weather features.")
    parser.add_argument(
        "--days", type=int, default=730,
        help="How many days of history to backfill (default 730 = 2 years). "
             "OpenWeather's pollution history goes back to 2020-11-27, so you "
             "can safely go much higher, e.g. --days 1800.",
    )
    args = parser.parse_args()
    backfill(args.days)
