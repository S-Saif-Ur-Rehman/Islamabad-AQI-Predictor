import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.aqi_calculator import aqi_category, aqi_from_pm10, aqi_from_pm25, estimate_overall_aqi


def test_aqi_from_pm25_good_range():
    # 0 ug/m3 -> AQI 0, top of "Good" band (12.0) -> AQI 50
    assert aqi_from_pm25(0.0) == 0
    assert aqi_from_pm25(12.0) == 50


def test_aqi_from_pm25_moderate_range():
    # Midpoint of the 12.1-35.4 band should land inside 51-100
    result = aqi_from_pm25(23.75)
    assert 51 <= result <= 100


def test_aqi_from_pm10_boundary():
    assert aqi_from_pm10(54) == 50
    assert aqi_from_pm10(0) == 0


def test_aqi_above_scale_clamps_to_500():
    assert aqi_from_pm25(1000) == 500


def test_aqi_none_input_returns_none():
    assert aqi_from_pm25(None) is None


def test_estimate_overall_aqi_takes_worst_pollutant():
    # PM2.5 well into "Unhealthy", PM10 still "Good" -> overall should follow PM2.5
    result = estimate_overall_aqi(pm25_ugm3=100, pm10_ugm3=10)
    assert result == aqi_from_pm25(100)


def test_aqi_category_labels():
    assert aqi_category(25) == "Good"
    assert aqi_category(75) == "Moderate"
    assert aqi_category(125) == "Unhealthy for Sensitive Groups"
    assert aqi_category(175) == "Unhealthy"
    assert aqi_category(250) == "Very Unhealthy"
    assert aqi_category(400) == "Hazardous"
    assert aqi_category(None) == "Unknown"
