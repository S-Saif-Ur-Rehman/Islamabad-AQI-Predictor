"""Feature engineering for hourly and daily AQI data."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Add time features and cyclical encoding for the day and month."""
    df = df.copy()
    ts = pd.to_datetime(df[timestamp_col])
    df["hour"] = ts.dt.hour
    df["day"] = ts.dt.day
    df["month"] = ts.dt.month
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Keep time values smooth across day and year boundaries.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_hourly_derived_features(df: pd.DataFrame, aqi_col: str = "aqi") -> pd.DataFrame:
    """Add short-term AQI change and rolling values for hourly data."""
    df = df.sort_values("timestamp").copy()
    df = df.set_index(pd.to_datetime(df["timestamp"]))

    df["aqi_change_rate_1h"] = df[aqi_col].diff()
    df["aqi_rolling_mean_3h"] = df[aqi_col].rolling("3h").mean()
    df["aqi_rolling_mean_24h"] = df[aqi_col].rolling("24h").mean()
    df["aqi_rolling_std_24h"] = df[aqi_col].rolling("24h").std()

    return df.reset_index(drop=True)


def build_daily_features(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Turn hourly rows into one daily row per date."""
    df = hourly_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date

    agg_map = {
        "aqi": ["mean", "max", "min"],
        "pm25": "mean",
        "pm10": "mean",
        "o3": "mean",
        "no2": "mean",
        "so2": "mean",
        "co": "mean",
        "temp_c": "mean",
        "humidity_pct": "mean",
        "pressure_hpa": "mean",
        "wind_speed_ms": "mean",
    }
    # Keep only columns that exist in the raw data.
    agg_map = {k: v for k, v in agg_map.items() if k in df.columns}

    daily = df.groupby("date").agg(agg_map)
    daily.columns = ["_".join(c) if isinstance(c, tuple) else c for c in daily.columns]
    daily = daily.rename(columns={"aqi_mean": "aqi"})  # aqi_mean is our primary daily target/feature
    daily = daily.reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    # Daily time features
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(int)
    daily["month"] = daily["date"].dt.month
    daily["month_sin"] = np.sin(2 * np.pi * daily["month"] / 12)
    daily["month_cos"] = np.cos(2 * np.pi * daily["month"] / 12)

    # Add change and rolling context.
    daily["aqi_change_rate_1d"] = daily["aqi"].diff()
    daily["aqi_rolling_mean_3d"] = daily["aqi"].rolling(3, min_periods=1).mean()
    daily["aqi_rolling_mean_7d"] = daily["aqi"].rolling(7, min_periods=1).mean()
    daily["aqi_lag_1d"] = daily["aqi"].shift(1)
    daily["aqi_lag_2d"] = daily["aqi"].shift(2)
    daily["aqi_lag_3d"] = daily["aqi"].shift(3)

    return daily


def add_forecast_targets(daily_df: pd.DataFrame, horizons_days: list[int]) -> pd.DataFrame:
    """Add target columns for each future forecast horizon."""
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    indexed = df.set_index("date")["aqi"]
    for h in horizons_days:
        # Match by date so missing future days stay missing.
        lookup_dates = df["date"] + pd.Timedelta(days=h)
        df[f"aqi_target_{h}d"] = indexed.reindex(lookup_dates).to_numpy()
    return df


FEATURE_COLUMNS = [
    "day_of_week",
    "is_weekend",
    "month_sin",
    "month_cos",
    "pm25_mean",
    "pm10_mean",
    "o3_mean",
    "no2_mean",
    "so2_mean",
    "co_mean",
    "temp_c_mean",
    "humidity_pct_mean",
    "pressure_hpa_mean",
    "wind_speed_ms_mean",
    "aqi",
    "aqi_change_rate_1d",
    "aqi_rolling_mean_3d",
    "aqi_rolling_mean_7d",
    "aqi_lag_1d",
    "aqi_lag_2d",
    "aqi_lag_3d",
]


def select_available_features(df: pd.DataFrame) -> list[str]:
    """Use only the feature columns that are available."""
    return [c for c in FEATURE_COLUMNS if c in df.columns]
