"""SHAP explanations for the selected model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def explain_prediction(model, model_type: str, background_df: pd.DataFrame, row_df: pd.DataFrame):
    """Return SHAP values and the base value for one row."""
    if model_type == "random_forest":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(row_df)
        base_value = explainer.expected_value
    else:
        # Use the generic SHAP path for non-tree models.
        predict_fn = model.predict if not hasattr(model, "predict_proba") else model.predict_proba
        background_sample = shap.sample(background_df, min(50, len(background_df)))
        explainer = shap.Explainer(predict_fn, background_sample)
        shap_result = explainer(row_df)
        shap_values = shap_result.values
        base_value = shap_result.base_values

    return np.array(shap_values).flatten(), np.array(base_value).flatten()[0]


def top_feature_contributions(feature_columns: list[str], shap_values: np.ndarray, top_n: int = 8):
    """Return the top feature impacts for a chart."""
    pairs = list(zip(feature_columns, shap_values))
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    return pairs[:top_n]
