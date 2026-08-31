"""Fetch one AQI row and save it to the hourly feature store."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from config import config
from src.clients.aqicn_client import AQICNClient
from src.clients.openweather_client import OpenWeatherClient
from src.feature_store.base import get_feature_store
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def run() -> pd.DataFrame:
    aqicn = AQICNClient()
    openweather = OpenWeatherClient()

    aqi_reading = aqicn.get_current_reading()
    weather = openweather.get_current_weather()
    pollution = openweather.get_current_air_pollution()

    # Round the timestamp to the hour so repeated hourly runs land on a
    # consistent primary key even if the job fires a few seconds late.
    ts = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    row = {
        "timestamp": ts,
        "city": config.CITY_NAME,
        "aqi": aqi_reading["aqi"],
        "pm25": aqi_reading.get("pm25") or pollution.get("pm2_5"),
        "pm10": aqi_reading.get("pm10") or pollution.get("pm10"),
        "o3": aqi_reading.get("o3") or pollution.get("o3"),
        "no2": aqi_reading.get("no2") or pollution.get("no2"),
        "so2": aqi_reading.get("so2") or pollution.get("so2"),
        "co": aqi_reading.get("co") or pollution.get("co"),
        "temp_c": weather["temp_c"],
        "humidity_pct": weather["humidity_pct"],
        "pressure_hpa": weather["pressure_hpa"],
        "wind_speed_ms": weather["wind_speed_ms"],
    }
    df = pd.DataFrame([row])

    # Force these to float64 explicitly. With only one row, pandas infers
    # each column's dtype from that single value — so whenever the API
    # happens to return a whole number (e.g. pm25=42 instead of 42.3),
    # pandas gives that column int64 instead of float64. Hopsworks' feature
    # group schema expects double for these columns, so an int64 column
    # fails the insert (and a None value would otherwise leave the column
    # as dtype 'object', which fails the same check for a different reason).
    float_cols = [
        "pm25", "pm10", "o3", "no2", "so2", "co",
        "temp_c", "pressure_hpa", "wind_speed_ms",
    ]
    df[float_cols] = df[float_cols].astype("float64")
    # The existing Hopsworks feature group deliberately stores humidity as
    # an integer percentage. It is always whole-number data from OpenWeather.
    df["humidity_pct"] = df["humidity_pct"].astype("int64")

    store = get_feature_store()
    store.insert(config.RAW_HOURLY_FG, df, primary_key=["timestamp"])
    logger.info("Inserted hourly reading for %s at %s: AQI=%s", config.CITY_NAME, ts.isoformat(), row["aqi"])
    return df


if __name__ == "__main__":
    setup_logging()
    run()
