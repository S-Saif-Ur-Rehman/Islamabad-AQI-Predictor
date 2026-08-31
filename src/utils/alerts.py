"""Check whether AQI values cross the alert threshold."""
from __future__ import annotations

from config import config
from src.features.aqi_calculator import aqi_category


def check_alert(current_aqi: float, forecast: dict[str, float]) -> dict:
    """Return alert details for a forecast dictionary."""
    threshold = config.AQI_ALERT_THRESHOLD
    breaches = {}
    if current_aqi is not None and current_aqi >= threshold:
        breaches["current"] = current_aqi
    for horizon, value in forecast.items():
        if value is not None and value >= threshold:
            breaches[horizon] = value

    return {
        "alert": len(breaches) > 0,
        "threshold": threshold,
        "breaches": breaches,
        "worst_category": aqi_category(max([current_aqi or 0, *[v for v in forecast.values() if v]], default=0)),
    }
