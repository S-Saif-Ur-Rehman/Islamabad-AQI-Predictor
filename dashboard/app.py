"""Streamlit dashboard for the AQI predictor."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Allow imports from the project root when running locally.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Load secrets into env vars for the app config.
try:
    for _key, _value in st.secrets.items():
        os.environ.setdefault(str(_key), str(_value))
except Exception:
    # No Streamlit secrets file here, so the app uses env vars directly.
    pass

from config import config
from src.feature_store.base import get_feature_store
from src.utils.error_handling import sanitize_exception_message


def _run_inference() -> dict:
    # Single-tier architecture: always run inference in-process (reads Hopsworks
    # directly, loads the model, computes SHAP). No Flask API involved.
    from src.pipelines import inference_pipeline

    return inference_pipeline.run()

st.set_page_config(page_title=f"AQI Forecast — {config.CITY_NAME}", page_icon="🌫️", layout="wide")

CATEGORY_COLORS = {
    "Good": "#00e400",
    "Moderate": "#ffff00",
    "Unhealthy for Sensitive Groups": "#ff7e00",
    "Unhealthy": "#ff0000",
    "Very Unhealthy": "#8f3f97",
    "Hazardous": "#7e0023",
    "Unknown": "#9e9e9e",
}

MODEL_DISPLAY_NAMES = {
    "persistence_baseline": "Persistence baseline",
    "ridge": "Ridge Regression",
    "random_forest": "Random Forest",
    "tensorflow_mlp": "TensorFlow MLP",
}


# Data loading

@st.cache_data(ttl=600)
def load_daily_history():
    try:
        store = get_feature_store()
        df = store.read(config.DAILY_FEATURES_FG)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600)
def load_model_comparison():
    path = config.LOCAL_MODEL_REGISTRY_DIR / "comparison.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def category_badge(category: str, value: float) -> str:
    color = CATEGORY_COLORS.get(category, "#9e9e9e")
    return f"""
    <div style="background-color:{color}22;border:1px solid {color};border-radius:10px;
                padding:14px;text-align:center;">
        <div style="font-size:2.2rem;font-weight:700;color:{color};">{value:.0f}</div>
        <div style="font-size:0.95rem;color:#333;">{category}</div>
    </div>
    """


def recompute_breaches(current_aqi: float, forecast_dict: dict, threshold: int) -> dict:
    """Same logic as src/utils/alerts.py, but against the sidebar's live threshold slider."""
    breaches = {}
    if current_aqi is not None and current_aqi >= threshold:
        breaches["current"] = current_aqi
    for horizon, value in forecast_dict.items():
        if value is not None and value >= threshold:
            breaches[horizon] = value
    return breaches


# Session state

if "forecast" not in st.session_state:
    st.session_state.forecast = None
    st.session_state.forecast_error = None

# Sidebar

with st.sidebar:
    st.header("🌫️ Controls")
    st.caption(f"Tracking **{config.CITY_NAME}**")

    generate_clicked = st.button("🔮 Generate Forecast", type="primary", width="stretch")
    if st.session_state.forecast is not None:
        if st.button("🔄 Refresh Forecast", width="stretch"):
            generate_clicked = True

    if generate_clicked:
        with st.spinner("Fetching latest features, running models, computing SHAP explanations — up to ~30s..."):
            try:
                st.session_state.forecast = _run_inference()
                st.session_state.forecast_error = None
            except Exception as exc:  # noqa: BLE001
                st.session_state.forecast_error = sanitize_exception_message(
                    exc, "forecast unavailable"
                )
                st.session_state.forecast = None
        load_daily_history.clear()
        load_model_comparison.clear()

    st.divider()
    st.subheader("Display settings")
    alert_threshold = st.slider(
        "Alert threshold (AQI)", min_value=0, max_value=500,
        value=config.AQI_ALERT_THRESHOLD, step=10,
        help="Adjust to preview alerts at a different sensitivity — doesn't change the .env default.",
    )
    history_days = st.select_slider(
        "History window shown", options=[7, 14, 30, 60, 90, 180],
        value=60, help="How many past days to plot on the trend chart.",
    )

    st.divider()
    if st.button("🗑️ Clear cached data", width="stretch"):
        load_daily_history.clear()
        load_model_comparison.clear()
        st.rerun()

# Title

st.title(f"🌫️ AQI Forecast — {config.CITY_NAME}")
st.caption(
    "Serverless AQI forecasting pipeline: AQICN + OpenWeather → feature store → "
    f"{'Hopsworks' if config.USE_HOPSWORKS else 'local'} models → this dashboard."
)

# Idle, error, and result states

if st.session_state.forecast_error:
    st.error(
        "Couldn't generate a forecast. Make sure you've run, in order: "
        "`backfill_pipeline.py` (or let `feature_pipeline.py` run hourly for a while), "
        "`daily_aggregation.py`, then `training_pipeline.py`."
    )
    st.stop()

if st.session_state.forecast is None:
    st.info("👈 Click **Generate Forecast** in the sidebar to fetch the latest AQI prediction.")
    st.stop()

forecast = st.session_state.forecast

# Alert banner
breaches = recompute_breaches(forecast["current_aqi"], forecast["forecast"], alert_threshold)
if breaches:
    st.error(
        f"⚠️ Hazardous air quality: AQI is at/above {alert_threshold} for: "
        f"{', '.join(f'{k}={v:.0f}' for k, v in breaches.items())}. "
        "Consider limiting outdoor activity."
    )
else:
    st.success(f"✅ No forecasted AQI reaches the {alert_threshold} alert threshold.")

# Current and forecast cards
st.subheader(f"Current conditions (as of {forecast['as_of_date']})")
cols = st.columns(4)
with cols[0]:
    st.markdown(category_badge(forecast["current_category"], forecast["current_aqi"]), unsafe_allow_html=True)
    st.caption("Today")

for i, (horizon, value) in enumerate(forecast["forecast"].items(), start=1):
    with cols[i]:
        category = forecast["forecast_categories"][horizon]
        st.markdown(category_badge(category, value), unsafe_allow_html=True)
        st.caption(f"+{horizon.replace('d', ' day(s)')}")

st.download_button(
    "⬇️ Download this forecast (JSON)",
    data=json.dumps(forecast, indent=2, default=str),
    file_name=f"aqi_forecast_{config.CITY_NAME.lower()}_{forecast['as_of_date']}.json",
    mime="application/json",
)

st.divider()

tab_forecast, tab_explain, tab_models, tab_eda = st.tabs(
    ["📈 Trend & Forecast", "🔍 Why this forecast (SHAP)", "🏆 Model Comparison", "📊 EDA"]
)

# Trend and forecast tab
with tab_forecast:
    daily_df = load_daily_history()
    if daily_df.empty:
        st.info("No historical data yet.")
    else:
        history = daily_df[["date", "aqi"]].tail(history_days).copy()
        history["type"] = "Observed"

        last_date = history["date"].max()
        forecast_rows = [
            {"date": last_date + pd.Timedelta(days=int(h.replace("d", ""))), "aqi": v, "type": "Forecast"}
            for h, v in forecast["forecast"].items()
        ]
        combined = pd.concat([history, pd.DataFrame(forecast_rows)], ignore_index=True)

        fig = px.line(
            combined, x="date", y="aqi", color="type", markers=True,
            title=f"Daily average AQI — last {history_days} days + 3-day forecast",
        )
        fig.add_hline(
            y=alert_threshold, line_dash="dot", line_color="red",
            annotation_text="Alert threshold",
        )
        st.plotly_chart(fig, width="stretch")

# Explainability tab
with tab_explain:
    horizon_choice = st.selectbox("Forecast horizon", list(forecast["forecast"].keys()))
    exp = forecast["explanations"].get(horizon_choice)
    if not exp:
        st.info("No SHAP explanation available for this horizon.")
    else:
        contrib_df = pd.DataFrame(exp["top_contributions"]).sort_values("impact")
        fig = go.Figure(
            go.Bar(
                x=contrib_df["impact"], y=contrib_df["feature"], orientation="h",
                marker_color=["#ff0000" if v > 0 else "#00a000" for v in contrib_df["impact"]],
            )
        )
        fig.update_layout(
            title=f"Top feature contributions — {horizon_choice} forecast "
                  f"(base value: {exp['base_value']:.1f})",
            xaxis_title="Impact on predicted AQI",
        )
        st.plotly_chart(fig, width="stretch")
        st.caption("Red bars push AQI higher; green bars push it lower, relative to the base value.")

# Model comparison tab
with tab_models:
    comparison = load_model_comparison()
    if not comparison:
        st.info(
            "No model comparison data yet — this is written by `training_pipeline.py` "
            "on its next run (`python -m src.pipelines.training_pipeline`)."
        )
    else:
        st.caption(f"From the training run at {comparison['generated_at']} UTC.")
        rows = []
        for horizon_result in comparison["results"]:
            horizon = horizon_result["horizon"]
            for model_name, metrics in horizon_result["metrics"].items():
                rows.append(
                    {
                        "Horizon": horizon,
                        "Model": MODEL_DISPLAY_NAMES.get(model_name, model_name),
                        "RMSE": metrics["rmse"],
                        "MAE": metrics["mae"],
                        "R²": metrics["r2"],
                        "Deployed": "✅" if model_name == horizon_result["best_model"] else "",
                    }
                )
        comp_df = pd.DataFrame(rows)

        fig = px.bar(
            comp_df, x="Model", y="RMSE", color="Horizon", barmode="group",
            title="RMSE by model and horizon (lower is better)",
        )
        st.plotly_chart(fig, width="stretch")

        st.dataframe(
            comp_df.style.format({"RMSE": "{:.2f}", "MAE": "{:.2f}", "R²": "{:.3f}"}),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "The persistence baseline (\"tomorrow's AQI = today's AQI\") is the naive statistical "
            "benchmark every other model should beat. A model scoring worse than it, or a negative "
            "R², usually means there isn't yet enough real training history — check back as more "
            "hourly data accumulates."
        )

# EDA tab
with tab_eda:
    daily_df = load_daily_history()
    if daily_df.empty:
        st.info("No historical data yet.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                px.histogram(daily_df, x="aqi", nbins=30, title="Distribution of daily AQI"),
                width="stretch",
            )
        with c2:
            weather_cols = [c for c in ["temp_c_mean", "humidity_pct_mean", "wind_speed_ms_mean"] if c in daily_df.columns]
            if weather_cols:
                melted = daily_df.melt(id_vars=["date", "aqi"], value_vars=weather_cols)
                st.plotly_chart(
                    px.scatter(melted, x="value", y="aqi", facet_col="variable", trendline="ols",
                               title="AQI vs. weather variables"),
                    width="stretch",
                )

        numeric_cols = [c for c in daily_df.select_dtypes("number").columns if daily_df[c].notna().sum() > 5]
        if len(numeric_cols) > 1:
            corr = daily_df[numeric_cols].corr()
            st.plotly_chart(
                px.imshow(corr, title="Feature correlation matrix", color_continuous_scale="RdBu_r", zmin=-1, zmax=1),
                width="stretch",
            )

st.divider()
st.caption(f"Forecast generated at {forecast['generated_at']} UTC.")
