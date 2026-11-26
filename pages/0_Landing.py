"""Landing Page — Vortex Sentinel Security Operations Center."""

from __future__ import annotations

import streamlit as st

from src.ui_theme import get_db, get_model_bundle, inject_vortex_theme, render_priority_badge

st.set_page_config(
    page_title="Landing | Vortex Sentinel",
    page_icon="assets/vortex_sentinel_logo.png",
    layout="wide",
)

inject_vortex_theme()

db = get_db()
bundle = get_model_bundle()
stats = db.get_dashboard_stats()
avg_txt = f"{stats['avg_risk_score']:.1f}" if stats["avg_risk_score"] is not None else "N/A"

from pathlib import Path

LOGO_FILE = Path("assets/vortex_sentinel_logo.png")
if LOGO_FILE.exists():
    hero_logo_col, hero_card_col = st.columns([0.16, 0.84], vertical_alignment="center")
    with hero_logo_col:
        st.image(str(LOGO_FILE), width=130)
else:
    hero_card_col = st.container()

with hero_card_col:
    st.markdown(
        """
        <div class="vortex-hero">
            <div class="vortex-telemetry-strip" style="margin-bottom: 8px;">
                <span class="vortex-badge">AVISHKAR'26 • TEAM VORTEX • ID: AV26-4C8B10F965</span>
                <span class="vortex-badge-cyan">SECURITY OPERATIONS CENTER</span>
                <span class="vortex-badge-cyan">ZERO-EXECUTION • PREVIEW_ALLOWED=FALSE</span>
            </div>
            <h1 style="margin: 0 0 6px 0; font-size: 1.95rem; font-weight: 700; color: #F8FAFC;">
                🛡️ Vortex Sentinel — Static Malware Triage &amp; Explainable AI
            </h1>
            <p style="margin: 0; font-size: 0.96rem; color: #CBD5E1; line-height: 1.5;">
                Zero-execution static risk scoring for Windows PE binaries
                (<span class="vortex-mono">.exe</span> <span class="vortex-mono">.dll</span>
                <span class="vortex-mono">.sys</span> <span class="vortex-mono">.scr</span>) and PDF documents—pairing
                <span class="vortex-mono">RandomForest</span> &amp; <span class="vortex-mono">HistGradientBoosting</span>
                with per-file <strong>SHAP</strong> drivers.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

cta1, cta2, cta3, cta4, cta5 = st.columns([1.3, 1.35, 1.25, 1.15, 1.1])
with cta1:
    if st.button("🚀 Launch SOC Dashboard", type="primary", use_container_width=True):
        st.switch_page("app.py")
with cta2:
    if st.button("🧪 Unified Triage Workbench", use_container_width=True):
        st.switch_page("pages/1_Static_Triage_Workbench.py")
with cta3:
    if st.button("🛡️ Open Executable Scanner", use_container_width=True):
        st.switch_page("pages/1_Executable_Scanner.py")
with cta4:
    if st.button("📄 Open PDF Inspector", use_container_width=True):
        st.switch_page("pages/2_PDF_Inspector.py")
with cta5:
    if st.button("📦 Batch CSV Triage", use_container_width=True):
        st.switch_page("pages/3_Batch_Analysis.py")

# Compact 3-column / 2-row SOC Bento grid above the fold
r1_c1, r1_c2, r1_c3 = st.columns([1.15, 1.15, 1.0])
with r1_c1:
    st.markdown(
        """
        <div class="vortex-bento">
            <div class="vortex-bento-header">
                <span>01 • PE Binary Triage</span>
                <span class="vortex-mono">IN-MEMORY</span>
            </div>
            <p style="margin: 0 0 8px 0; font-size: 0.88rem; color: #CBD5E1; line-height: 1.45;">
                Extracts PE headers, section entropy, Windows API imports, and strings strictly in memory—never executing binaries.
            </p>
            <div style="font-size: 0.78rem; color: #94A3B8;">
                <span class="vortex-mono">pefile</span> <span class="vortex-mono">shannon_entropy</span>
                <span class="vortex-mono">sha256</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with r1_c2:
    st.markdown(
        """
        <div class="vortex-bento-cyan">
            <div class="vortex-bento-header">
                <span>02 • PDF Document Inspector</span>
                <span class="vortex-mono">PREVIEW=FALSE</span>
            </div>
            <p style="margin: 0 0 8px 0; font-size: 0.88rem; color: #CBD5E1; line-height: 1.45;">
                Inspects PDFs under page, byte, and time limits with browser auto-preview disabled and routes embedded PEs safely.
            </p>
            <div style="font-size: 0.78rem; color: #94A3B8;">
                <span class="vortex-mono">/JS</span> <span class="vortex-mono">/OpenAction</span>
                <span class="vortex-mono">/Launch</span> <span class="vortex-mono">/EmbeddedFile</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with r1_c3:
    st.markdown(
        f"""
        <div class="vortex-bento">
            <div class="vortex-bento-header">
                <span>Live Telemetry</span>
                <span class="vortex-mono">SQLITE</span>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                <div>
                    <div class="vortex-kpi-label">Total Scans</div>
                    <div class="vortex-kpi-value" style="font-size: 1.45rem;">{stats["total_scans"]}</div>
                </div>
                <div>
                    <div class="vortex-kpi-label">High / Crit</div>
                    <div class="vortex-kpi-value" style="font-size: 1.45rem; color: #FBBF24;">{stats["high_or_critical"]}</div>
                </div>
                <div>
                    <div class="vortex-kpi-label">Unsupported</div>
                    <div class="vortex-kpi-value" style="font-size: 1.45rem; color: #94A3B8;">{stats["unsupported_scans"]}</div>
                </div>
                <div>
                    <div class="vortex-kpi-label">Avg Risk</div>
                    <div class="vortex-kpi-value" style="font-size: 1.45rem; color: #2DD4BF;">{avg_txt}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

r2_c1, r2_c2, r2_c3 = st.columns([1.15, 1.15, 1.0])
with r2_c1:
    st.markdown(
        """
        <div class="vortex-bento-cyan">
            <div class="vortex-bento-header">
                <span>03 • Explainable AI (SHAP)</span>
                <span class="vortex-mono">0–100 RISK</span>
            </div>
            <p style="margin: 0; font-size: 0.88rem; color: #CBD5E1; line-height: 1.45;">
                Decomposes every prediction into positive &amp; negative SHAP feature drivers and exports ReportLab PDF audit reports.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with r2_c2:
    st.markdown(
        f"""
        <div class="vortex-bento">
            <div class="vortex-bento-header">
                <span>Priority Tiers</span>
                <span class="vortex-mono">PROBABILISTIC</span>
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 6px; padding: 4px 0;">
                {render_priority_badge("Low")}
                {render_priority_badge("Medium")}
                {render_priority_badge("High")}
                {render_priority_badge("Critical")}
                {render_priority_badge("Unsupported")}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with r2_c3:
    st.markdown(
        """
        <div class="vortex-bento-cyan">
            <div class="vortex-bento-header">
                <span>Safety &amp; Benchmark Notice</span>
                <span class="vortex-mono">SYNTHETIC</span>
            </div>
            <p style="margin: 0; font-size: 0.82rem; color: #94A3B8; line-height: 1.4;">
                Built-in models use <strong>synthetic/simulated</strong> data (not real-world malware validation). Scores are risk estimates, not proof.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.expander("Static Triage Scope & Limitations"):
    st.markdown(
        "- **Entropy & Packing Context:** High Shannon entropy or packed sections raise triage priority for deeper inspection "
        "but do **not** guarantee maliciousness (legitimate installers and compressed binaries can also exhibit high entropy).\n"
        "- **Zero-Execution Queue Prioritization:** Static triage prioritizes SOC analyst queues in milliseconds without executing "
        "binaries or rendering PDFs (`preview_allowed=False`) prior to isolated sandbox review."
    )
