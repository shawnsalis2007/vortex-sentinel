"""Model Comparison — RandomForestClassifier vs. HistGradientBoostingClassifier."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.ui_theme import apply_vortex_plotly_theme, get_model_bundle, inject_vortex_theme, render_page_header

st.set_page_config(page_title="Model Comparison | Vortex Sentinel", page_icon="assets/vortex_sentinel_logo.png", layout="wide")
inject_vortex_theme()

render_page_header(
    "⚖️ Model Comparison (Held-Out Stratified Split)",
    "Side-by-side evaluation of RandomForestClassifier and HistGradientBoostingClassifier on synthetic static features.",
    badge="SYNTHETIC BENCHMARK",
)
bundle = get_model_bundle()

if bundle.dataset_metadata.is_synthetic:
    st.warning(bundle.dataset_metadata.disclaimer)

rf = bundle.rf_metrics
hgb = bundle.hgb_metrics

metrics_df = pd.DataFrame(
    [
        {
            "Model": rf.model_name,
            "Accuracy": rf.accuracy,
            "Precision": rf.precision,
            "Recall": rf.recall,
            "F1-Score": rf.f1_score,
            "ROC-AUC": rf.roc_auc,
            "False Positive Rate": rf.false_positive_rate,
        },
        {
            "Model": hgb.model_name,
            "Accuracy": hgb.accuracy,
            "Precision": hgb.precision,
            "Recall": hgb.recall,
            "F1-Score": hgb.f1_score,
            "ROC-AUC": hgb.roc_auc,
            "False Positive Rate": hgb.false_positive_rate,
        },
    ]
)

from src.triage_views import render_model_agreement_badge

spread_pts = abs(rf.roc_auc - hgb.roc_auc) * 100.0
if spread_pts >= 30.0:
    mc_badge, mc_lvl = "DIVERGENT — MANUAL REVIEW", "critical"
    mc_status = f"Held-out ROC-AUC spread is {spread_pts:.1f} pts — bagging vs. sequential boosting diverge."
elif spread_pts >= 15.0:
    mc_badge, mc_lvl = "MODERATE VARIANCE", "medium"
    mc_status = f"Held-out ROC-AUC spread is {spread_pts:.1f} pts — mild variance between ensembles."
else:
    mc_badge, mc_lvl = "STRONG CONSENSUS", "low"
    mc_status = f"Both ensembles align closely on held-out split (ROC-AUC spread: {spread_pts:.2f} pts; Accuracy: {rf.accuracy:.3f} vs {hgb.accuracy:.3f})."

st.markdown(
    f'<div class="vortex-bento">'
    f'<div class="vortex-bento-header"><span>Ensemble Agreement (Bagging vs. Boosting)</span>'
    f'{render_model_agreement_badge({"badge": mc_badge, "badge_level": mc_lvl})}</div>'
    f'<p style="margin:0;color:#CBD5E1;font-size:0.88rem;">{mc_status}</p>'
    f'</div>',
    unsafe_allow_html=True,
)

st.dataframe(metrics_df, use_container_width=True, hide_index=True)

melted = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
fig = px.bar(
    melted,
    x="Metric",
    y="Score",
    color="Model",
    barmode="group",
    color_discrete_map={
        "RandomForestClassifier": "#38BDF8",
        "HistGradientBoostingClassifier": "#2DD4BF",
    },
)
fig.update_traces(hovertemplate="<b>%{x}</b><br>Score: %{y:.4f}<extra></extra>")
st.plotly_chart(
    apply_vortex_plotly_theme(fig, "Side-by-Side Held-Out Evaluation"),
    use_container_width=True,
    config={"displayModeBar": False},
)

with st.expander("Train/Test Split & Zero-Leakage Audit"):
    st.json(bundle.split_audit.to_dict())
