"""Fetch older weather data from Open-Meteo."""
from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd
import requests

from config import config

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"


def get_historical_weather(start_date: date, end_date: date, lat: float = None, lon: float = None) -> pd.DataFrame:
    lat = lat if lat is not None else config.LATITUDE
    lon = lon if lon is not None else config.LONGITUDE

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
        "timezone": "UTC",
    }
    resp = requests.get(BASE_URL, params=params, timeout=config.REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    hourly = resp.json().get("hourly", {})
    if not hourly:
        return pd.DataFrame()

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(hourly["time"]).tz_localize(timezone.utc),
            "temp_c": hourly.get("temperature_2m"),
            "humidity_pct": hourly.get("relative_humidity_2m"),
            "pressure_hpa": hourly.get("surface_pressure"),
            "wind_speed_ms": hourly.get("wind_speed_10m"),
        }
    )
    return df


if __name__ == "__main__":
    from datetime import timedelta

    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=3)
    print(get_historical_weather(start, end).head())
