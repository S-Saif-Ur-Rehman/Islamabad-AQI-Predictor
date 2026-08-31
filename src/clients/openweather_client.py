"""Client for OpenWeather weather and pollution data."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import requests

from config import config
from src.utils.error_handling import sanitize_exception_message

logger = logging.getLogger(__name__)

BASE_URL = "https://api.openweathermap.org"


class OpenWeatherClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.OPENWEATHER_API_KEY
        if not self.api_key:
            raise ValueError(
                "OPENWEATHER_API_KEY is not set. Add it to your .env file "
                "(get a free key at https://home.openweathermap.org/api_keys)."
            )

    def _get(self, path: str, params: dict) -> dict:
        params = {**params, "appid": self.api_key}
        try:
            resp = requests.get(f"{BASE_URL}{path}", params=params, timeout=config.REQUEST_TIMEOUT_SECONDS)
            resp.raise_for_status()
        except requests.exceptions.RequestException:
            raise RuntimeError("OpenWeather request failed. Please try again later.") from None
        return resp.json()

    # Weather

    def get_current_weather(self, lat: float = None, lon: float = None) -> dict:
        lat = lat if lat is not None else config.LATITUDE
        lon = lon if lon is not None else config.LONGITUDE
        data = self._get("/data/2.5/weather", {"lat": lat, "lon": lon, "units": "metric"})
        return self._parse_weather_entry(data["dt"], data["main"], data.get("wind", {}))

    def get_weather_forecast(self, lat: float = None, lon: float = None) -> list[dict]:
        """Returns up to 5 days of 3-hour-step forecasts."""
        lat = lat if lat is not None else config.LATITUDE
        lon = lon if lon is not None else config.LONGITUDE
        data = self._get("/data/2.5/forecast", {"lat": lat, "lon": lon, "units": "metric"})
        return [
            self._parse_weather_entry(entry["dt"], entry["main"], entry.get("wind", {}))
            for entry in data.get("list", [])
        ]

    @staticmethod
    def _parse_weather_entry(unix_ts: int, main: dict, wind: dict) -> dict:
        return {
            "timestamp": datetime.fromtimestamp(unix_ts, tz=timezone.utc),
            "temp_c": main.get("temp"),
            "humidity_pct": main.get("humidity"),
            "pressure_hpa": main.get("pressure"),
            "wind_speed_ms": wind.get("speed"),
        }

    # Air pollution

    def get_current_air_pollution(self, lat: float = None, lon: float = None) -> dict:
        lat = lat if lat is not None else config.LATITUDE
        lon = lon if lon is not None else config.LONGITUDE
        data = self._get("/data/2.5/air_pollution", {"lat": lat, "lon": lon})
        return self._parse_pollution_entry(data["list"][0])

    def get_air_pollution_history(
        self, start: datetime, end: datetime, lat: float = None, lon: float = None
    ) -> list[dict]:
        """
        Returns hourly pollutant concentrations between `start` and `end`
        (inclusive, UTC). Used by the backfill pipeline.
        """
        lat = lat if lat is not None else config.LATITUDE
        lon = lon if lon is not None else config.LONGITUDE
        data = self._get(
            "/data/2.5/air_pollution/history",
            {
                "lat": lat,
                "lon": lon,
                "start": int(start.timestamp()),
                "end": int(end.timestamp()),
            },
        )
        return [self._parse_pollution_entry(entry) for entry in data.get("list", [])]

    @staticmethod
    def _parse_pollution_entry(entry: dict) -> dict:
        components = entry.get("components", {})
        return {
            "timestamp": datetime.fromtimestamp(entry["dt"], tz=timezone.utc),
            "ow_aqi_1to5": entry.get("main", {}).get("aqi"),  # OpenWeather's own 1-5 scale
            "co": components.get("co"),
            "no2": components.get("no2"),
            "o3": components.get("o3"),
            "so2": components.get("so2"),
            "pm2_5": components.get("pm2_5"),
            "pm10": components.get("pm10"),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = OpenWeatherClient()
    print(client.get_current_weather())
    print(client.get_current_air_pollution())
