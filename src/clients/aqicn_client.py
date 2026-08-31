"""Small client for AQICN data."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import requests

from config import config
from src.utils.error_handling import sanitize_exception_message

logger = logging.getLogger(__name__)

BASE_URL = "https://api.waqi.info"


class AQICNClient:
    def __init__(self, token: Optional[str] = None):
        self.token = token or config.AQICN_API_TOKEN
        if not self.token:
            raise ValueError(
                "AQICN_API_TOKEN is not set. Add it to your .env file "
                "(get a free token at https://aqicn.org/data-platform/token/)."
            )

    def _get(self, path: str) -> dict:
        url = f"{BASE_URL}/{path}"
        try:
            resp = requests.get(
                url,
                params={"token": self.token},
                timeout=config.REQUEST_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(
                "AQICN request failed. Please try again later."
            ) from None
        payload = resp.json()
        if payload.get("status") != "ok":
            raise RuntimeError("AQICN API returned an invalid response.")
        return payload["data"]

    def get_current_by_geo(self, lat: float, lon: float) -> dict:
        """Fetch the current reading for the station nearest to (lat, lon)."""
        return self._get(f"feed/geo:{lat};{lon}/")

    def get_current_by_city(self, city_slug: str) -> dict:
        """Fetch the current reading for a named city/station slug."""
        return self._get(f"feed/{city_slug}/")

    def get_current_reading(self) -> dict:
        """
        Fetch the current reading using config location, falling back from
        geo lookup to the named city slug if geo lookup fails.
        """
        try:
            data = self.get_current_by_geo(config.LATITUDE, config.LONGITUDE)
        except Exception:
            data = self.get_current_by_city(config.AQICN_CITY_SLUG)
        return self._parse_reading(data)

    @staticmethod
    def _parse_reading(data: dict) -> dict:
        """Flatten the AQICN response into a single-level dict of features."""
        iaqi = data.get("iaqi", {})

        def sub(key: str) -> Optional[float]:
            entry = iaqi.get(key)
            return entry.get("v") if entry else None

        # AQICN gives a timestamp string with a UTC offset baked in; fall back
        # to "now" if parsing fails so a bad timestamp never kills the pipeline.
        try:
            observed_at = datetime.fromisoformat(data["time"]["iso"]).astimezone(timezone.utc)
        except Exception:  # noqa: BLE001
            observed_at = datetime.now(timezone.utc)

        return {
            "timestamp": observed_at,
            "aqi": data.get("aqi"),
            "station_name": data.get("city", {}).get("name"),
            "pm25": sub("pm25"),
            "pm10": sub("pm10"),
            "o3": sub("o3"),
            "no2": sub("no2"),
            "so2": sub("so2"),
            "co": sub("co"),
            # AQICN also reports station-level temp/humidity/pressure/wind;
            # useful as a cross-check against OpenWeather but OpenWeather is
            # the primary weather source since it also gives us a forecast.
            "station_temp": sub("t"),
            "station_humidity": sub("h"),
            "station_pressure": sub("p"),
            "station_wind": sub("w"),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = AQICNClient()
    print(client.get_current_reading())
