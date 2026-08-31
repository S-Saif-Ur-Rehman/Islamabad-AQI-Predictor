"""Small Flask API for forecast requests."""
from __future__ import annotations

import logging
import sys
import time
import hmac
from pathlib import Path

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import config
from src.pipelines import inference_pipeline
from src.utils.error_handling import safe_failure, sanitize_exception_message, suppress_known_runtime_warnings

suppress_known_runtime_warnings()

app = Flask(__name__)
app.config["PROPAGATE_EXCEPTIONS"] = False
app.config["JSON_SORT_KEYS"] = False
app.logger.disabled = True
logging.getLogger("werkzeug").disabled = True
logging.getLogger("werkzeug").setLevel(logging.CRITICAL)
_last_request = 0.0


def safe_json_error(message: str, status_code: int):
    """Return a clean JSON error response."""
    return {"error": message}, status_code


@app.errorhandler(HTTPException)
def handle_http_exception(exc: HTTPException):
    if exc.code == 400:
        return safe_json_error("bad request", 400)
    if exc.code == 401:
        return safe_json_error("authentication required", 401)
    if exc.code == 403:
        return safe_json_error("forbidden", 403)
    if exc.code == 404:
        return safe_json_error("resource not found", 404)
    if exc.code == 405:
        return safe_json_error("method not allowed", 405)
    if exc.code == 429:
        return safe_json_error("rate limit exceeded", 429)
    if exc.code and exc.code >= 500:
        return safe_json_error("internal server error", 500)
    return safe_json_error("request failed", exc.code or 400)


@app.errorhandler(Exception)
def handle_unexpected_error(exc: Exception):
    if isinstance(exc, HTTPException):
        return handle_http_exception(exc)
    sanitized = sanitize_exception_message(exc, "internal server error")
    return safe_json_error(sanitized if sanitized == "internal server error" else "internal server error", 500)


@app.route("/")
def index():
    return jsonify(
        {
            "status": "ok",
            "city": config.CITY_NAME,
            "message": "AQI forecast API is running.",
            "endpoints": ["/health", "/forecast", "/predict"],
        }
    )


@app.route("/favicon.ico")
def favicon():
    return safe_json_error("resource not found", 404)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "city": config.CITY_NAME})


@app.route("/forecast")
@app.route("/predict")  # alias for the forecast endpoint
@safe_failure(default_return={"error": "forecast unavailable"}, logger=logging.getLogger(__name__))
def forecast():
    """Run the forecast and return JSON."""
    global _last_request
    if config.API_AUTH_TOKEN and not hmac.compare_digest(request.headers.get("X-API-Key", ""), config.API_AUTH_TOKEN):
        return {"error": "authentication required"}, 401
    now = time.monotonic()
    if now - _last_request < 5:
        return {"error": "rate limit exceeded"}, 429
    _last_request = now
    result = inference_pipeline.run()
    return jsonify(result)


@app.route("/history")
@safe_failure(default_return=[], logger=logging.getLogger(__name__))
def history():
    """Return daily history in JSON format."""
    from src.feature_store.base import get_feature_store

    store = get_feature_store()
    df = store.read(config.DAILY_FEATURES_FG)
    if df.empty:
        return jsonify([])
    df = df.sort_values("date").reset_index(drop=True)
    # Convert dates to JSON-friendly strings.
    records = df.to_dict(orient="records")
    return jsonify(records)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
