"""Project settings and environment config."""
import os
from pathlib import Path
from dotenv import load_dotenv

from src.utils.error_handling import suppress_known_runtime_warnings

# Load local env values when present.
load_dotenv()
suppress_known_runtime_warnings()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Location
CITY_NAME = os.getenv("CITY_NAME", "Islamabad")
LATITUDE = float(os.getenv("LATITUDE", "33.6844"))
LONGITUDE = float(os.getenv("LONGITUDE", "73.0479"))
# Islamabad station used by the project.
AQICN_CITY_SLUG = os.getenv("AQICN_CITY_SLUG", "pakistan/islamabad/us-embassy")

# API keys
# Accept both legacy and current env names.
AQICN_API_TOKEN = os.getenv("AQICN_API_TOKEN") or os.getenv("AQICN_API_KEY", "")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

# Feature store and model registry
USE_HOPSWORKS = os.getenv("USE_HOPSWORKS", "false").strip().lower() == "true"
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME") or os.getenv("HOPSWORKS_PROJECT", "")

# Feature group names
RAW_HOURLY_FG = "aqi_raw_hourly"
DAILY_FEATURES_FG = "aqi_daily_features"
PREDICTIONS_FG = "aqi_predictions"

# Local fallback storage
LOCAL_STORE_DIR = PROJECT_ROOT / "data" / "local_store"
LOCAL_MODEL_REGISTRY_DIR = PROJECT_ROOT / "models_registry"

# Forecast windows
FORECAST_HORIZONS_DAYS = [1, 2, 3]

# Alerts and limits
AQI_ALERT_THRESHOLD = int(os.getenv("AQI_ALERT_THRESHOLD", "150"))
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN", "")
MAX_BACKFILL_DAYS = int(os.getenv("MAX_BACKFILL_DAYS", "730"))

# Request settings
REQUEST_TIMEOUT_SECONDS = 15
