"""Investigation Reports — Downloadable ReportLab PDF Triage Reports."""

from __future__ import annotations

import json

import streamlit as st

from src.investigation_assistant import format_triage_markdown, generate_triage_brief
from src.reports_and_db import generate_pdf_investigation_report
from src.ui_theme import get_db, inject_vortex_theme, render_page_header, render_priority_badge

st.set_page_config(page_title="Investigation Reports | Vortex Sentinel", page_icon="assets/vortex_sentinel_logo.png", layout="wide")
inject_vortex_theme()

render_page_header(
    "📑 Investigation Reports (PDF Export)",
    "Generate ReportLab static-analysis audit reports from persisted SQLite scan records.",
    badge="REPORTLAB AUDIT EXPORT",
)
db = get_db()
scans = db.list_scans(limit=100)

if not scans:
    st.info("No scans found in SQLite history yet. Run a PE or PDF scan first to generate a report.")
else:
    options = {s["id"]: f"#{s['id']} — {s['file_name']} [{s['file_type']} | {s['priority']}]" for s in scans}
    selected_id = st.selectbox("Select Scan Record", options=list(options.keys()), format_func=lambda k: options[k])
    record = db.get_scan(int(selected_id))

    if record:
        score_txt = f"{record['risk_score']:.1f}" if record["risk_score"] is not None else "N/A"
        st.markdown(
            f'<div class="vortex-bento">'
            f'<div class="vortex-bento-header"><span>Record #{record["id"]} • {record["file_name"]}</span>'
            f'{render_priority_badge(record["priority"])}</div>'
            f'<div style="display:flex;flex-wrap:wrap;gap:14px;margin-bottom:8px;">'
            f'<div><span class="vortex-kpi-label">Risk Score:</span> <strong style="color:#38BDF8;">{score_txt}</strong></div>'
            f'<div><span class="vortex-kpi-label">Type:</span> <span class="vortex-mono">{record["file_type"]}</span></div>'
            f'</div>'
            f'<div class="vortex-mono" style="word-break:break-all;margin-bottom:8px;">SHA-256: {record["sha256"] or "N/A"}</div>'
            f'<p style="margin:0;color:#CBD5E1;font-size:0.90rem;">{record["summary"]}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        payload = record.get("payload") or {}
        brief = payload.get("triage_brief")
        if not isinstance(brief, dict):
            brief = generate_triage_brief(
                filename=str(record.get("file_name", "unknown")),
                file_type=str(record.get("file_type", "PE")),
                features={"sha256": record.get("sha256"), "indicators": payload.get("indicators") or []},
                risk_score=record.get("risk_score"),
                model_name=str(record.get("model_name") or "N/A"),
                shap_contributions=payload.get("top_contributions"),
                secondary_score=None,
            )

        pdf_bytes = generate_pdf_investigation_report(record)
        md_report = format_triage_markdown(brief)
        json_report = json.dumps(brief, indent=2, default=str)

        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button(
                label="Download PDF Investigation Report",
                data=pdf_bytes,
                file_name=f"vortex_sentinel_report_{record['id']}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                label="Download Inspection Report (.md)",
                data=md_report.encode("utf-8"),
                file_name=f"vortex_sentinel_report_{record['id']}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with d3:
            st.download_button(
                label="Download Triage Summary (.json)",
                data=json_report.encode("utf-8"),
                file_name=f"vortex_sentinel_summary_{record['id']}.json",
                mime="application/json",
                use_container_width=True,
            )
