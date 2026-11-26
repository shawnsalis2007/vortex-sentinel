"""Explainable AI — Interactive SHAP Feature Attribution Explorer."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.feature_extraction import PE_FEATURE_LABELS
from src.investigation_assistant import generate_triage_brief
from src.ml_engine import explain_and_score_features
from src.risk_scoring import score_probability
from src.triage_views import render_triage_assistant_panel
from src.ui_theme import (
    apply_vortex_plotly_theme,
    get_model_bundle,
    inject_vortex_theme,
    render_page_header,
    render_risk_score_hero,
)

st.set_page_config(page_title="Explainable AI | Vortex Sentinel", page_icon="assets/vortex_sentinel_logo.png", layout="wide")
inject_vortex_theme()

render_page_header(
    "🔍 Explainable AI (SHAP Feature Contributions)",
    "Adjust static feature values to inspect how positive and negative SHAP drivers shape the risk estimate.",
    badge="SHAP ATTRIBUTION",
)

bundle = get_model_bundle()
model_key = st.selectbox(
    "Model",
    options=["random_forest", "hist_gradient_boosting"],
    format_func=lambda k: "RandomForestClassifier" if k == "random_forest" else "HistGradientBoostingClassifier",
)

defaults = {
    "file_size": 240000.0,
    "file_entropy": 6.4,
    "section_count": 5.0,
    "mean_section_entropy": 5.6,
    "max_section_entropy": 6.8,
    "suspicious_section_names": 0.0,
    "import_count": 85.0,
    "imported_dll_count": 6.0,
    "suspicious_api_count": 2.0,
    "has_authenticode_signature": 1.0,
    "ascii_string_count": 620.0,
    "suspicious_string_count": 1.0,
    "virtual_to_raw_ratio": 1.1,
}

cols = st.columns(3)
inputs: dict[str, float] = {}
for idx, feat in enumerate(bundle.feature_columns):
    with cols[idx % 3]:
        inputs[feat] = st.number_input(PE_FEATURE_LABELS.get(feat, feat), value=float(defaults.get(feat, 0.0)))

pred = explain_and_score_features(bundle, inputs, model_key=model_key, top_n=13)
assess = score_probability(pred.malicious_probability, supported=pred.supported, warning=pred.warning)

render_risk_score_hero(
    assess.risk_score,
    assess.priority,
    label="Estimated Risk Score (0–100)",
    subtitle=pred.analysis_disclaimer,
)
st.info(pred.plain_language_summary)

shap_df = pd.DataFrame(
    [
        {
            "Feature": c.label,
            "SHAP Value": c.shap_value,
            "Direction": "Raises Risk (+)" if c.shap_value > 0 else "Lowers Risk (-)",
        }
        for c in pred.top_contributions
    ]
)
fig = px.bar(
    shap_df,
    x="Feature",
    y="SHAP Value",
    color="Direction",
    color_discrete_map={"Raises Risk (+)": "#8B5CF6", "Lowers Risk (-)": "#2DD4BF"},
)
fig.update_traces(hovertemplate="<b>%{x}</b><br>SHAP Value: %{y:+.4f}<extra></extra>")
st.plotly_chart(
    apply_vortex_plotly_theme(fig, "SHAP Positive & Negative Contributions"),
    use_container_width=True,
    config={"displayModeBar": False},
)

secondary_key = "hist_gradient_boosting" if model_key == "random_forest" else "random_forest"
sec_pred = explain_and_score_features(bundle, inputs, model_key=secondary_key, top_n=3)
sec_assess = score_probability(sec_pred.malicious_probability, supported=sec_pred.supported, warning=sec_pred.warning)
live_brief = generate_triage_brief(
    filename="interactive_shap_simulation.exe",
    file_type="PE",
    features=inputs,
    risk_score=assess.risk_score,
    model_name=pred.model_name,
    shap_contributions=pred.top_contributions,
    secondary_score=sec_assess.risk_score,
)
render_triage_assistant_panel(live_brief, key_prefix="xai_live_assist")
