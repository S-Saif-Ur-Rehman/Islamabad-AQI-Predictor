"""AQI math for PM2.5 and PM10."""
from __future__ import annotations

from typing import Optional

# (C_low, C_high, AQI_low, AQI_high)
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]

PM10_BREAKPOINTS = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 504, 301, 400),
    (505, 604, 401, 500),
]


def _linear_scale(concentration: float, breakpoints: list[tuple]) -> Optional[float]:
    if concentration is None:
        return None
    if concentration < 0:
        return None
    # Round to the EPA breakpoint step before lookup.
    concentration = (int(concentration * 10) / 10) if breakpoints is PM25_BREAKPOINTS else int(concentration)
    for c_low, c_high, aqi_low, aqi_high in breakpoints:
        if c_low <= concentration <= c_high:
            return round(
                (aqi_high - aqi_low) / (c_high - c_low) * (concentration - c_low) + aqi_low
            )
    # Cap values at the top AQI level instead of extrapolating.
    if concentration > breakpoints[-1][1]:
        return 500
    return None


def aqi_from_pm25(pm25_ugm3: Optional[float]) -> Optional[float]:
    return _linear_scale(pm25_ugm3, PM25_BREAKPOINTS)


def aqi_from_pm10(pm10_ugm3: Optional[float]) -> Optional[float]:
    return _linear_scale(pm10_ugm3, PM10_BREAKPOINTS)


def estimate_overall_aqi(pm25_ugm3: Optional[float], pm10_ugm3: Optional[float]) -> Optional[float]:
    """Overall AQI is the max of the individual pollutant sub-indices."""
    candidates = [v for v in (aqi_from_pm25(pm25_ugm3), aqi_from_pm10(pm10_ugm3)) if v is not None]
    return max(candidates) if candidates else None


def aqi_category(aqi: Optional[float]) -> str:
    """EPA AQI category label for a given AQI value."""
    if aqi is None:
        return "Unknown"
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"
