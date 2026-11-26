"""Scan History — Search, Filter, and Side-by-Side Comparison."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ui_theme import get_db, inject_vortex_theme, render_page_header

st.set_page_config(page_title="Scan History | Vortex Sentinel", page_icon="assets/vortex_sentinel_logo.png", layout="wide")
inject_vortex_theme()

render_page_header(
    "🗂️ Scan History & Side-by-Side Comparison",
    "Filter persisted SQLite scan records by file type, priority tier, or SHA-256 / filename query.",
    badge="SQLITE AUDIT TRAIL",
)
db = get_db()

c1, c2, c3 = st.columns(3)
file_type_filter = c1.selectbox("File Type", ["All", "PE", "PDF", "CSV_RECORD"])
priority_filter = c2.selectbox("Priority", ["All", "Critical", "High", "Medium", "Low", "Unsupported"])
query = c3.text_input("Search (Filename, SHA-256, Summary)")

scans = db.list_scans(file_type=file_type_filter, priority=priority_filter, query=query, limit=200)

if not scans:
    st.info("No matching scan records found.")
else:
    df = pd.DataFrame(
        [
            {
                "ID": s["id"],
                "Timestamp": s["created_at"],
                "File": s["file_name"],
                "Type": s["file_type"],
                "Priority": s["priority"],
                "Risk Score": f"{s['risk_score']:.1f}" if s["risk_score"] is not None else "N/A",
                "SHA-256": s["sha256"][:16] + "..." if s["sha256"] else "N/A",
            }
            for s in scans
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

    if len(scans) >= 2:
        st.subheader("Compare Two Scans Side-by-Side")
        id_map = {s["id"]: f"#{s['id']} — {s['file_name']} ({s['priority']})" for s in scans}
        col_a, col_b = st.columns(2)
        id_a = col_a.selectbox("Scan A", options=list(id_map.keys()), format_func=lambda k: id_map[k], key="cmp_a")
        id_b = col_b.selectbox("Scan B", options=list(id_map.keys()), index=1, format_func=lambda k: id_map[k], key="cmp_b")
        comp = db.compare_scans(int(id_a), int(id_b))
        st.write(f"**Same SHA-256:** {comp['same_sha256']} | **Risk Score Delta (B - A):** {comp['risk_score_delta']}")
        ca, cb = st.columns(2)
        ca.json({"file": comp["scan_a"]["file_name"], "priority": comp["scan_a"]["priority"], "risk_score": comp["scan_a"]["risk_score"], "summary": comp["scan_a"]["summary"]})
        cb.json({"file": comp["scan_b"]["file_name"], "priority": comp["scan_b"]["priority"], "risk_score": comp["scan_b"]["risk_score"], "summary": comp["scan_b"]["summary"]})
