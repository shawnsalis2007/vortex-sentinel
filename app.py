"""Vortex Sentinel — Main Streamlit SOC Dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.ui_theme import (
    LOGO_PATH,
    apply_vortex_plotly_theme,
    get_db,
    get_model_bundle,
    inject_vortex_theme,
    render_priority_badge,
)

st.set_page_config(
    page_title="SOC Dashboard | Vortex Sentinel",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🛡️",
    layout="wide",
)

inject_vortex_theme()

if LOGO_PATH.exists():
    _dash_logo, _dash_hero = st.columns([0.09, 0.91], vertical_alignment="center")
    with _dash_logo:
        st.image(str(LOGO_PATH), width=88)
else:
    _dash_hero = st.container()

with _dash_hero:
    st.markdown(
        """
        <div class="vortex-hero" style="padding: 14px 18px; margin-bottom: 10px;">
            <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 8px;">
                <div>
                    <div class="vortex-telemetry-strip" style="margin-bottom: 4px;">
                        <span class="vortex-badge">SECURITY OPERATIONS CENTER</span>
                        <span class="vortex-badge-cyan">TEAM VORTEX • AV26-4C8B10F965</span>
                        <span class="vortex-badge-cyan">ZERO-EXECUTION STATIC TRIAGE</span>
                    </div>
                    <h2 style="margin: 0; font-size: 1.55rem; color: #F8FAFC;">
                        🛡️ Vortex Sentinel — SOC Triage Command
                    </h2>
                </div>
                <div style="font-size: 0.80rem; color: #94A3B8; max-width: 380px; text-align: right;">
                    Probabilistic static risk estimates (not proof). Browser auto-preview disabled (<span class="vortex-mono">preview_allowed=False</span>).
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

db = get_db()
bundle = get_model_bundle()
stats = db.get_dashboard_stats()
recent_scans = db.list_scans(limit=15)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f'<div class="vortex-bento"><div class="vortex-kpi-label">Total Scans</div>'
        f'<div class="vortex-kpi-value">{stats["total_scans"]}</div></div>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f'<div class="vortex-bento"><div class="vortex-kpi-label">High / Critical</div>'
        f'<div class="vortex-kpi-value" style="color: #FBBF24;">{stats["high_or_critical"]}</div></div>',
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f'<div class="vortex-bento-cyan"><div class="vortex-kpi-label">Unsupported / Incomplete</div>'
        f'<div class="vortex-kpi-value" style="color: #94A3B8;">{stats["unsupported_scans"]}</div></div>',
        unsafe_allow_html=True,
    )
with col4:
    avg_txt = f"{stats['avg_risk_score']:.1f}" if stats["avg_risk_score"] is not None else "N/A"
    st.markdown(
        f'<div class="vortex-bento-cyan"><div class="vortex-kpi-label">Avg Risk Score</div>'
        f'<div class="vortex-kpi-value" style="color: #2DD4BF;">{avg_txt}</div></div>',
        unsafe_allow_html=True,
    )

# Compact quick-action bar
qa0, qa1, qa2, qa3, qa4 = st.columns(5)
with qa0:
    if st.button("🧪 Triage Workbench", use_container_width=True):
        st.switch_page("pages/1_Static_Triage_Workbench.py")
with qa1:
    if st.button("🛡️ Scan Executable", use_container_width=True):
        st.switch_page("pages/1_Executable_Scanner.py")
with qa2:
    if st.button("📄 Inspect PDF", use_container_width=True):
        st.switch_page("pages/2_PDF_Inspector.py")
with qa3:
    if st.button("🔍 Explainable AI", use_container_width=True):
        st.switch_page("pages/5_Explainable_AI.py")
with qa4:
    if st.button("📑 Audit Reports", use_container_width=True):
        st.switch_page("pages/6_Investigation_Reports.py")

if bundle.dataset_metadata.is_synthetic:
    st.info(bundle.dataset_metadata.disclaimer)

left, mid, right = st.columns([1.0, 1.0, 1.35])

with left:
    st.markdown(
        '<div class="vortex-bento"><div class="vortex-bento-header">'
        '<span>Risk Priority Distribution</span><span class="vortex-mono">VERTICAL</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    prio_counts = stats["priority_counts"]
    if prio_counts:
        prio_df = pd.DataFrame([{"Priority": k, "Count": v} for k, v in prio_counts.items()])
        fig_prio = px.bar(
            prio_df,
            x="Priority",
            y="Count",
            color="Priority",
            color_discrete_map={
                "Low": "#2DD4BF",
                "Medium": "#38BDF8",
                "High": "#F59E0B",
                "Critical": "#F43F5E",
                "Unsupported": "#64748B",
            },
        )
        fig_prio.update_traces(hovertemplate="<b>%{x}</b><br>Scans: %{y}<extra></extra>")
        st.plotly_chart(
            apply_vortex_plotly_theme(fig_prio),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    else:
        st.caption("No scans recorded yet.")

with mid:
    st.markdown(
        '<div class="vortex-bento-cyan"><div class="vortex-bento-header">'
        '<span>Held-Out Model Benchmark</span><span class="vortex-mono">SYNTHETIC</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    rf = bundle.rf_metrics
    hgb = bundle.hgb_metrics
    bench_df = pd.DataFrame(
        [
            {"Metric": "Accuracy", "Model": "RF", "Score": rf.accuracy},
            {"Metric": "Precision", "Model": "RF", "Score": rf.precision},
            {"Metric": "Recall", "Model": "RF", "Score": rf.recall},
            {"Metric": "ROC-AUC", "Model": "RF", "Score": rf.roc_auc},
            {"Metric": "Accuracy", "Model": "HGB", "Score": hgb.accuracy},
            {"Metric": "Precision", "Model": "HGB", "Score": hgb.precision},
            {"Metric": "Recall", "Model": "HGB", "Score": hgb.recall},
            {"Metric": "ROC-AUC", "Model": "HGB", "Score": hgb.roc_auc},
        ]
    )
    fig_bench = px.bar(
        bench_df,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        color_discrete_map={"RF": "#38BDF8", "HGB": "#8B5CF6"},
    )
    fig_bench.update_traces(hovertemplate="<b>%{x}</b><br>Score: %{y:.3f}<extra></extra>")
    st.plotly_chart(
        apply_vortex_plotly_theme(fig_bench),
        use_container_width=True,
        config={"displayModeBar": False},
    )

with right:
    st.markdown(
        '<div class="vortex-bento"><div class="vortex-bento-header">'
        '<span>Recent Scans</span><span class="vortex-mono">SQLITE AUDIT</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    if recent_scans:
        latest = recent_scans[0]
        st.markdown(
            f'<div class="vortex-bento-cyan" style="padding: 10px 14px;">'
            f'Latest: <strong>{latest["file_name"]}</strong> {render_priority_badge(latest["priority"])}'
            f'<div class="vortex-mono" style="margin-top: 6px; word-break: break-all;">'
            f'SHA-256: {latest["sha256"] or "N/A"}'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        table_df = pd.DataFrame(
            [
                {
                    "ID": s["id"],
                    "File": s["file_name"],
                    "Type": s["file_type"],
                    "Priority": s["priority"],
                    "Risk": f"{s['risk_score']:.1f}" if s["risk_score"] is not None else "N/A",
                    "Timestamp": s["created_at"],
                }
                for s in recent_scans
            ]
        )
        st.dataframe(table_df, use_container_width=True, hide_index=True, height=280)
    else:
        st.caption("Scan history is empty.")
