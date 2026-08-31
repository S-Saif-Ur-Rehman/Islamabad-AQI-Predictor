import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.feature_engineering import (
    add_forecast_targets,
    add_time_features,
    build_daily_features,
    select_available_features,
)


def _make_hourly_df(n_days: int = 10) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=n_days * 24, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "aqi": rng.integers(20, 180, size=len(timestamps)),
            "pm25": rng.uniform(5, 80, size=len(timestamps)),
            "pm10": rng.uniform(10, 120, size=len(timestamps)),
            "o3": rng.uniform(5, 60, size=len(timestamps)),
            "no2": rng.uniform(5, 40, size=len(timestamps)),
            "so2": rng.uniform(1, 20, size=len(timestamps)),
            "co": rng.uniform(100, 900, size=len(timestamps)),
            "temp_c": rng.uniform(-5, 30, size=len(timestamps)),
            "humidity_pct": rng.uniform(20, 90, size=len(timestamps)),
            "pressure_hpa": rng.uniform(990, 1025, size=len(timestamps)),
            "wind_speed_ms": rng.uniform(0, 10, size=len(timestamps)),
        }
    )


def test_add_time_features_creates_expected_columns():
    df = _make_hourly_df(1)
    result = add_time_features(df)
    for col in ["hour", "day", "month", "day_of_week", "is_weekend", "hour_sin", "hour_cos"]:
        assert col in result.columns
    assert result["hour_sin"].between(-1, 1).all()


def test_build_daily_features_one_row_per_day():
    hourly = _make_hourly_df(n_days=5)
    daily = build_daily_features(hourly)
    assert len(daily) == 5
    assert "aqi" in daily.columns
    assert "aqi_rolling_mean_3d" in daily.columns
    # First day should have no lag-1 value
    assert pd.isna(daily.loc[0, "aqi_lag_1d"])
    # Second day's lag-1 should equal the first day's AQI
    assert daily.loc[1, "aqi_lag_1d"] == daily.loc[0, "aqi"]


def test_add_forecast_targets_shifts_correctly():
    hourly = _make_hourly_df(n_days=6)
    daily = build_daily_features(hourly)
    targeted = add_forecast_targets(daily, [1, 2, 3])
    # Target for day 0, horizon 1 == actual AQI on day 1
    assert targeted.loc[0, "aqi_target_1d"] == daily.loc[1, "aqi"]
    # Last `h` rows for each horizon should be NaN (no future data yet)
    assert targeted["aqi_target_3d"].iloc[-3:].isna().all()


def test_select_available_features_filters_missing_columns():
    df = pd.DataFrame({"day_of_week": [0], "is_weekend": [0], "some_other_col": [1]})
    features = select_available_features(df)
    assert "day_of_week" in features
    assert "is_weekend" in features
    assert "some_other_col" not in features
