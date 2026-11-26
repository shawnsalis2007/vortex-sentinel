"""Vortex Sentinel — SQLite Scan History & ReportLab PDF Investigation Reports.

Provides local SQLite persistence for scan history (save, search/filter, and
side-by-side comparison) and generates downloadable PDF investigation reports
using ReportLab.
"""

from __future__ import annotations

from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sqlite3
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

DEFAULT_DB_PATH = Path("data") / "vortex_sentinel.db"


class ScanHistoryDB:
    """Lightweight SQLite store for static scan history and comparisons."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    md5 TEXT NOT NULL,
                    classification TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    risk_score REAL,
                    model_name TEXT,
                    supported INTEGER NOT NULL,
                    summary TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def save_scan(
        self,
        *,
        file_name: str,
        file_type: str,
        sha256: str,
        md5: str,
        classification: str,
        priority: str,
        risk_score: float | None,
        model_name: str | None,
        supported: bool,
        summary: str,
        payload: dict[str, Any],
    ) -> int:
        """Persist a completed or unsupported scan record and return its row ID."""
        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO scans (
                    created_at, file_name, file_type, sha256, md5,
                    classification, priority, risk_score, model_name,
                    supported, summary, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    file_name,
                    file_type,
                    sha256,
                    md5,
                    classification,
                    priority,
                    risk_score,
                    model_name,
                    1 if supported else 0,
                    summary,
                    json.dumps(payload, default=str),
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def list_scans(
        self,
        *,
        file_type: str | None = None,
        priority: str | None = None,
        query: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Query scan history with optional file type, priority, and text filters."""
        clauses: list[str] = []
        params: list[Any] = []

        if file_type and file_type != "All":
            clauses.append("file_type = ?")
            params.append(file_type)
        if priority and priority != "All":
            clauses.append("priority = ?")
            params.append(priority)
        if query and query.strip():
            like = f"%{query.strip()}%"
            clauses.append("(file_name LIKE ? OR sha256 LIKE ? OR summary LIKE ?)")
            params.extend([like, like, like])

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM scans {where_sql} ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_scan(self, scan_id: int) -> dict[str, Any] | None:
        """Fetch a single scan record by ID."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def get_dashboard_stats(self) -> dict[str, Any]:
        """Compute aggregate counts and priority breakdown for the dashboard."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
            high_crit = conn.execute(
                "SELECT COUNT(*) FROM scans WHERE priority IN ('High', 'Critical')"
            ).fetchone()[0]
            unsupported = conn.execute(
                "SELECT COUNT(*) FROM scans WHERE supported = 0 OR priority = 'Unsupported'"
            ).fetchone()[0]
            avg_score = conn.execute(
                "SELECT AVG(risk_score) FROM scans WHERE risk_score IS NOT NULL"
            ).fetchone()[0]
            prio_rows = conn.execute(
                "SELECT priority, COUNT(*) AS cnt FROM scans GROUP BY priority"
            ).fetchall()

        return {
            "total_scans": int(total),
            "high_or_critical": int(high_crit),
            "unsupported_scans": int(unsupported),
            "avg_risk_score": round(float(avg_score), 2) if avg_score is not None else None,
            "priority_counts": {r["priority"]: int(r["cnt"]) for r in prio_rows},
        }

    def compare_scans(self, scan_id_a: int, scan_id_b: int) -> dict[str, Any]:
        """Return two scan records side-by-side with key metric deltas."""
        a = self.get_scan(scan_id_a)
        b = self.get_scan(scan_id_b)
        if not a or not b:
            raise ValueError("One or both selected scan IDs were not found.")

        score_delta: float | None = None
        if a["risk_score"] is not None and b["risk_score"] is not None:
            score_delta = round(float(b["risk_score"]) - float(a["risk_score"]), 2)

        return {
            "scan_a": a,
            "scan_b": b,
            "same_sha256": bool(a["sha256"] and a["sha256"] == b["sha256"]),
            "risk_score_delta": score_delta,
        }

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["supported"] = bool(item["supported"])
        try:
            item["payload"] = json.loads(item.pop("payload_json", "{}"))
        except json.JSONDecodeError:
            item["payload"] = {}
        return item


def generate_pdf_investigation_report(scan_record: dict[str, Any]) -> bytes:
    """Generate a downloadable ReportLab PDF investigation report for a scan."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        leftMargin=0.65 * inch,
        rightMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title=f"Vortex Sentinel Report - {scan_record.get('file_name', 'Artifact')}",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "VortexTitle",
        parent=styles["Heading1"],
        fontSize=16,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=6,
    )
    h2_style = ParagraphStyle(
        "VortexH2",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=colors.HexColor("#1d4ed8"),
        spaceBefore=10,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "VortexBody",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=13,
    )

    payload: dict[str, Any] = scan_record.get("payload", {}) or {}
    score_val = scan_record.get("risk_score")
    score_text = f"{score_val:.2f} / 100" if score_val is not None else "N/A (Unsupported / Non-Scored)"

    logo_path = Path(__file__).resolve().parent.parent / "assets" / "vortex_sentinel_logo.png"
    title_block = [
        Paragraph("Vortex Sentinel — Static Malware Triage Report", title_style),
        Paragraph(
            "Team Vortex (ID: AV26-4C8B10F965) • AVISHKAR'26 • Zero-Execution Static Analysis",
            body_style,
        ),
    ]
    if logo_path.exists():
        hdr_tbl = Table(
            [[Image(str(logo_path), width=0.65 * inch, height=0.65 * inch), title_block]],
            colWidths=[0.8 * inch, 6.2 * inch],
        )
        hdr_tbl.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        elements: list[Any] = [hdr_tbl, Spacer(1, 10)]
    else:
        elements = [*title_block, Spacer(1, 10)]

    meta_rows = [
        ["Field", "Value"],
        ["File Name", str(scan_record.get("file_name", "N/A"))],
        ["File Type", str(scan_record.get("file_type", "N/A"))],
        ["SHA-256", str(scan_record.get("sha256", "N/A"))],
        ["MD5", str(scan_record.get("md5", "N/A"))],
        ["Classification", str(scan_record.get("classification", "N/A"))],
        ["Investigation Priority", str(scan_record.get("priority", "N/A"))],
        ["Risk Score (Estimate)", score_text],
        ["Model", str(scan_record.get("model_name") or "N/A")],
        ["Timestamp", str(scan_record.get("created_at", "N/A"))],
    ]
    meta_table = Table(meta_rows, colWidths=[1.8 * inch, 5.2 * inch])
    meta_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    elements.extend([meta_table, Spacer(1, 10)])

    # Unsupported / Incomplete Analysis Notice
    warnings = payload.get("warnings") or []
    if not scan_record.get("supported", True) or warnings:
        elements.append(Paragraph("Unsupported / Incomplete Analysis Notices", h2_style))
        if not scan_record.get("supported", True):
            elements.append(
                Paragraph(
                    f"• Analysis Status: Unsupported or incomplete ({payload.get('warning', 'See summary')}).",
                    body_style,
                )
            )
        for w in warnings[:10]:
            elements.append(Paragraph(f"• {w}", body_style))

    # Findings Summary
    elements.extend(
        [
            Paragraph("Findings & Risk Summary", h2_style),
            Paragraph(str(scan_record.get("summary", "No summary recorded.")), body_style),
        ]
    )

    # SHAP Positive & Negative Contributions (PE / CSV)
    pos = payload.get("positive_contributions") or []
    neg = payload.get("negative_contributions") or []
    if pos or neg:
        elements.append(Paragraph("Explainable AI (SHAP) Feature Contributions", h2_style))
        shap_rows = [["Direction", "Feature", "Observed Value", "SHAP Impact"]]
        for item in pos[:5]:
            shap_rows.append(
                [
                    "Raises Risk (+)",
                    str(item.get("label", item.get("feature"))),
                    str(item.get("value")),
                    f"{float(item.get('shap_value', 0.0)):+.4f}",
                ]
            )
        for item in neg[:5]:
            shap_rows.append(
                [
                    "Lowers Risk (-)",
                    str(item.get("label", item.get("feature"))),
                    str(item.get("value")),
                    f"{float(item.get('shap_value', 0.0)):+.4f}",
                ]
            )
        shap_table = Table(shap_rows, colWidths=[1.3 * inch, 2.7 * inch, 1.5 * inch, 1.5 * inch])
        shap_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ]
            )
        )
        elements.extend([shap_table, Spacer(1, 8)])

    # PDF-specific details (Page numbers, structures, attachments)
    indicators = payload.get("indicators") or []
    attachments = payload.get("attachments") or []
    if indicators or attachments or scan_record.get("file_type") == "PDF":
        elements.append(Paragraph("PDF Structure, Page Indicators & Attachments", h2_style))
        elements.append(
            Paragraph(
                f"Pages Inspected: {payload.get('pages_inspected', 0)} / {payload.get('total_pages', 0)} "
                f"| Browser Auto-Preview: Disabled (preview_allowed=False)",
                body_style,
            )
        )
        for ind in indicators[:12]:
            elements.append(
                Paragraph(
                    f"• [{ind.get('category')}] {ind.get('location')}: {ind.get('detail')}",
                    body_style,
                )
            )
        for att in attachments[:10]:
            elements.append(
                Paragraph(
                    f"• Attachment '{att.get('filename')}' ({att.get('size_bytes')} bytes) -> {att.get('routing_status')}: {att.get('review_reason')}",
                    body_style,
                )
            )

    # Deterministic SOC Triage Investigation Brief (if recorded in payload)
    triage_brief = payload.get("triage_brief")
    if isinstance(triage_brief, dict):
        elements.append(Paragraph("SOC Triage Investigation Brief (Deterministic)", h2_style))
        consensus = triage_brief.get("model_consensus") or {}
        if consensus.get("status"):
            elements.append(Paragraph(f"• Model Consensus: {consensus['status']}", body_style))
        for f in (triage_brief.get("findings") or [])[:6]:
            elements.append(Paragraph(f"• {f.get('signal')}: {f.get('detail')}", body_style))
        for idx, step in enumerate(triage_brief.get("next_steps") or [], start=1):
            elements.append(Paragraph(f"{idx}. {step}", body_style))

    # Recommended Next Steps & Disclaimer
    next_steps = payload.get("recommendation") or (
        "Review findings in an isolated environment. Do not execute untrusted artifacts on production hosts."
    )
    elements.extend(
        [
            Paragraph("Recommended Next Steps", h2_style),
            Paragraph(str(next_steps), body_style),
            Spacer(1, 8),
            Paragraph(
                "Disclaimer: Scores are statistical risk estimates, not proof of malware or safety. "
                "Single static indicators never prove malice, and zero indicators do not guarantee safety.",
                body_style,
            ),
        ]
    )

    doc.build(elements)
    return buffer.getvalue()
