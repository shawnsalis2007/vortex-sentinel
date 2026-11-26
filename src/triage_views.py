"""Shared UI views for PE Binary Scanner, PDF Inspector, and Batch CSV Analysis."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import streamlit as st

from src.feature_extraction import (
    extract_pe_features,
    normalize_public_static_dataset,
    validate_tabular_dataset,
)
from src.investigation_assistant import (
    answer_analyst_question,
    format_triage_markdown,
    generate_triage_brief,
)
from src.ml_engine import explain_and_score_features, score_pe_extraction
from src.pdf_inspection import PDFInspectionLimits, inspect_pdf_bytes
from src.risk_scoring import assess_pdf_priority, score_probability
from src.ui_theme import (
    apply_vortex_plotly_theme,
    get_db,
    get_model_bundle,
    render_priority_badge,
    render_risk_score_hero,
)

PE_EXTENSIONS = {".exe", ".dll", ".sys", ".scr"}


def render_model_agreement_badge(consensus: dict[str, Any]) -> str:
    """Return HTML pill badge for STRONG CONSENSUS / MODERATE VARIANCE / DIVERGENT."""
    badge_text = str(consensus.get("badge") or "SINGLE MODEL")
    level = str(consensus.get("badge_level") or "medium").lower()
    css_class = {
        "low": "vortex-pill-low",
        "medium": "vortex-pill-high",
        "critical": "vortex-pill-critical",
    }.get(level, "vortex-pill-unsupported")
    dot_color = {"low": "#2DD4BF", "medium": "#F59E0B", "critical": "#F43F5E"}.get(level, "#94A3B8")
    return (
        f'<span class="vortex-pill {css_class}">'
        f'<span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:{dot_color};"></span>'
        f'{badge_text}</span>'
    )


def render_triage_assistant_panel(brief: dict[str, Any], key_prefix: str = "assist") -> None:
    """Render the offline deterministic SOC Triage Investigation Assistant card."""
    consensus = brief.get("model_consensus") or {}
    badge_html = render_model_agreement_badge(consensus)
    st.markdown(
        f'<div class="vortex-bento">'
        f'<div class="vortex-bento-header">'
        f'<span>🤖 SOC Triage Investigation Assistant ({brief.get("attribution_source", "SHAP")})</span>'
        f'{badge_html}'
        f'</div>'
        f'<p style="margin:0 0 6px 0;color:#CBD5E1;font-size:0.88rem;">'
        f'<strong>Model Agreement:</strong> {consensus.get("status", "Single-model evaluation")}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("**Static Rule Findings**")
        for f in brief.get("findings") or []:
            st.markdown(f"- **{f['signal']}**: {f['detail']}")
    with c_right:
        st.markdown("**Recommended SOC Next Steps**")
        for idx, step in enumerate(brief.get("next_steps") or [], start=1):
            st.markdown(f"{idx}. {step}")

    st.caption(brief.get("disclaimer", ""))

    # 4 one-click analyst prompt chips + custom deterministic Q&A
    q1, q2, q3, q4 = st.columns(4)
    qa_key = f"{key_prefix}_qa_answer"
    if q1.button("Why this score?", key=f"{key_prefix}_q_why", use_container_width=True):
        st.session_state[qa_key] = answer_analyst_question("Why was this flagged?", brief)
    if q2.button("Recommended next steps", key=f"{key_prefix}_q_next", use_container_width=True):
        st.session_state[qa_key] = answer_analyst_question("What should I check next?", brief)
    if q3.button("False-positive check", key=f"{key_prefix}_q_fp", use_container_width=True):
        st.session_state[qa_key] = answer_analyst_question("Could this be a false positive?", brief)
    if q4.button("Model agreement", key=f"{key_prefix}_q_model", use_container_width=True):
        st.session_state[qa_key] = answer_analyst_question("How do the two models compare?", brief)

    custom_q = st.text_input(
        "Ask Offline Playbook Helper",
        placeholder="e.g., Why not execute the file? / Could this be a false positive?",
        key=f"{key_prefix}_custom_q",
    )
    if custom_q:
        st.session_state[qa_key] = answer_analyst_question(custom_q, brief)

    if st.session_state.get(qa_key):
        st.info(st.session_state[qa_key])

    md_note = format_triage_markdown(brief)
    json_note = json.dumps(brief, indent=2, default=str)
    stem = Path(str(brief.get("filename", "triage"))).stem
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "Download Inspection Report (.md)",
            data=md_note.encode("utf-8"),
            file_name=f"{stem}_inspection_report.md",
            mime="text/markdown",
            key=f"{key_prefix}_dl_md",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "Download Triage Summary (.json)",
            data=json_note.encode("utf-8"),
            file_name=f"{stem}_triage_summary.json",
            mime="application/json",
            key=f"{key_prefix}_dl_json",
            use_container_width=True,
        )

    with st.expander("Preview / Copy Triage Note (Markdown)"):
        st.code(md_note, language="markdown")


def render_pe_scanner_view(key_prefix: str = "pe") -> None:
    """Render the zero-execution Windows PE static analysis workflow."""
    bundle = get_model_bundle()
    db = get_db()

    model_key = st.selectbox(
        "Classifier Model",
        options=["random_forest", "hist_gradient_boosting"],
        format_func=lambda k: "RandomForestClassifier" if k == "random_forest" else "HistGradientBoostingClassifier",
        key=f"{key_prefix}_model",
    )

    uploaded = st.file_uploader(
        "Upload PE Binary (.exe, .dll, .sys, .scr)",
        type=["exe", "dll", "sys", "scr"],
        key=f"{key_prefix}_uploader",
    )

    if uploaded is not None:
        raw_bytes = uploaded.read()
        extraction = extract_pe_features(
            file_bytes=raw_bytes,
            file_name=uploaded.name,
            expected_features=bundle.feature_columns,
        )

        if not extraction.supported:
            st.error(extraction.warning or "Unsupported or incomplete PE binary.")
            assessment = score_probability(None, supported=False, warning=extraction.warning)
            db.save_scan(
                file_name=uploaded.name,
                file_type="PE",
                sha256=extraction.sha256,
                md5=extraction.md5,
                classification="Unsupported",
                priority=assessment.priority,
                risk_score=None,
                model_name=None,
                supported=False,
                summary=extraction.warning or "Unsupported PE binary.",
                payload={"warning": extraction.warning, "recommendation": assessment.recommendation},
            )
            return

        prediction = score_pe_extraction(bundle, extraction, model_key=model_key, top_n=8)
        assessment = score_probability(
            prediction.malicious_probability,
            supported=prediction.supported,
            warning=prediction.warning,
        )

        render_risk_score_hero(
            assessment.risk_score,
            assessment.priority,
            label="PE Static Risk Score",
            subtitle=assessment.disclaimer,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("File Entropy", f"{extraction.features['file_entropy']:.2f}")
        c2.metric("Authenticode Dir", "Present" if extraction.features["has_authenticode_signature"] else "Absent")
        c3.metric("Model", prediction.model_name)

        st.markdown(
            f'<div class="vortex-bento-cyan">'
            f'<div class="vortex-bento-header"><span>Risk Factor Summary</span>'
            f'<span class="vortex-mono">SHA-256: {extraction.sha256[:16]}...</span></div>'
            f'<p style="margin:0;color:#E2E8F0;font-size:0.92rem;">{prediction.plain_language_summary}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if extraction.warning:
            st.warning(extraction.warning)

        if prediction.top_contributions:
            shap_df = pd.DataFrame(
                [
                    {
                        "Feature": c.label,
                        "SHAP Impact": c.shap_value,
                        "Direction": "Raises Risk" if c.shap_value > 0 else "Lowers Risk",
                    }
                    for c in prediction.top_contributions
                ]
            )
            fig = px.bar(
                shap_df,
                x="Feature",
                y="SHAP Impact",
                color="Direction",
                color_discrete_map={"Raises Risk": "#8B5CF6", "Lowers Risk": "#2DD4BF"},
            )
            fig.update_traces(hovertemplate="<b>%{x}</b><br>SHAP Impact: %{y:+.4f}<extra></extra>")
            st.plotly_chart(
                apply_vortex_plotly_theme(fig, "Top Per-File SHAP Contributions"),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        secondary_key = "hist_gradient_boosting" if model_key == "random_forest" else "random_forest"
        sec_pred = score_pe_extraction(bundle, extraction, model_key=secondary_key, top_n=3)
        sec_assess = score_probability(sec_pred.malicious_probability, supported=sec_pred.supported)
        triage_brief = generate_triage_brief(
            filename=uploaded.name,
            file_type="PE",
            features={
                **extraction.features,
                "sha256": extraction.sha256,
                "suspicious_apis": extraction.suspicious_apis,
                "suspicious_strings": extraction.suspicious_strings,
            },
            risk_score=assessment.risk_score,
            model_name=prediction.model_name,
            shap_contributions=prediction.top_contributions,
            secondary_score=sec_assess.risk_score,
        )

        with st.expander("Extracted Static Features & Indicators"):
            st.json(
                {
                    "sha256": extraction.sha256,
                    "md5": extraction.md5,
                    "suspicious_apis": extraction.suspicious_apis,
                    "suspicious_strings": extraction.suspicious_strings[:10],
                    "features": extraction.features,
                }
            )

        render_triage_assistant_panel(triage_brief, key_prefix=f"{key_prefix}_pe_assist")

        scan_id = db.save_scan(
            file_name=uploaded.name,
            file_type="PE",
            sha256=extraction.sha256,
            md5=extraction.md5,
            classification=prediction.risk_tier,
            priority=assessment.priority,
            risk_score=assessment.risk_score,
            model_name=prediction.model_name,
            supported=True,
            summary=prediction.plain_language_summary,
            payload={
                **prediction.to_dict(),
                "recommendation": assessment.recommendation,
                "sha256": extraction.sha256,
                "md5": extraction.md5,
                "triage_brief": triage_brief,
            },
        )
        st.success(f"Scan saved to SQLite history (Scan ID #{scan_id}).")


def render_pdf_inspector_view(key_prefix: str = "pdf") -> None:
    """Render the zero-execution PDF static inspection workflow (preview_allowed=False)."""
    bundle = get_model_bundle()
    db = get_db()

    col_a, col_b = st.columns(2)
    max_pages = col_a.number_input(
        "Max Pages to Inspect",
        min_value=1,
        max_value=200,
        value=50,
        key=f"{key_prefix}_max_pages",
    )
    max_seconds = col_b.number_input(
        "Time Limit (seconds)",
        min_value=1.0,
        max_value=30.0,
        value=10.0,
        key=f"{key_prefix}_max_seconds",
    )

    uploaded = st.file_uploader(
        "Upload PDF Document (.pdf)",
        type=["pdf"],
        key=f"{key_prefix}_uploader",
    )

    if uploaded is not None:
        pdf_bytes = uploaded.read()
        limits = PDFInspectionLimits(max_pages=int(max_pages), max_processing_seconds=float(max_seconds))
        result = inspect_pdf_bytes(
            pdf_bytes,
            file_name=uploaded.name,
            limits=limits,
            expected_pe_features=bundle.feature_columns,
        )
        assessment = assess_pdf_priority(result.assessment_tier, result.supported, result.assessment_summary)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Status", result.status.upper())
        m2.metric("Priority", assessment.priority)
        m3.metric("Pages Inspected", f"{result.pages_inspected} / {result.total_pages}")
        m4.metric("Attachments", len(result.attachments))

        st.markdown(
            f'<div class="vortex-bento-cyan">'
            f'<div class="vortex-bento-header"><span>PDF Triage Assessment</span>'
            f'{render_priority_badge(assessment.priority)}</div>'
            f'<p style="margin:0 0 6px 0;color:#E2E8F0;font-size:0.92rem;">{result.assessment_summary}</p>'
            f'<div class="vortex-mono" style="word-break:break-all;">SHA-256: {result.sha256}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        for w in result.warnings:
            st.warning(w)

        if result.indicators:
            st.subheader("Static Indicators Observed")
            ind_df = pd.DataFrame([i.to_dict() for i in result.indicators])
            st.dataframe(ind_df, use_container_width=True, hide_index=True)

        if result.attachments:
            st.subheader("Embedded Attachments & Routing")
            for att in result.attachments:
                with st.expander(f"{att.filename} ({att.size_bytes:,} bytes) — {att.routing_status}"):
                    st.write(f"**SHA-256:** `{att.sha256}`")
                    st.write(f"**Routing Reason:** {att.review_reason}")
                    if att.pe_extraction and att.pe_extraction.supported:
                        pe_pred = score_pe_extraction(bundle, att.pe_extraction, model_key="random_forest")
                        pe_assess = score_probability(pe_pred.malicious_probability, supported=pe_pred.supported)
                        st.write(f"**Embedded PE Risk Score:** {pe_assess.risk_score} ({pe_assess.priority})")
                        st.write(pe_pred.plain_language_summary)

        with st.expander("Metadata, Links & Epistemic Limitations"):
            st.json({"metadata": result.metadata, "links": result.links[:30], "annotations": result.annotation_summary})
            for lim in result.limitations:
                st.caption(f"• {lim}")

        scan_id = db.save_scan(
            file_name=uploaded.name,
            file_type="PDF",
            sha256=result.sha256,
            md5=result.md5,
            classification=result.assessment_tier,
            priority=assessment.priority,
            risk_score=None,
            model_name=None,
            supported=result.supported,
            summary=result.assessment_summary,
            payload={
                **result.to_dict(),
                "recommendation": assessment.recommendation,
            },
        )
        st.success(f"PDF inspection saved to SQLite history (Scan ID #{scan_id}).")


SAMPLE_PUBLIC_DATASETS: dict[str, Path] = {
    "EMBER-Style Extracted PE Benchmark (40 rows)": Path("data/ember_extracted_pe_benchmark_sample.csv"),
    "Kaggle PE Header/Section Sample + Labels (40 rows)": Path("data/kaggle_pe_static_features_sample.csv"),
}


def _score_csv_dataframe(df: pd.DataFrame, model_key: str) -> dict[str, Any]:
    """Normalize public aliases, strictly validate schema, and score up to 500 rows."""
    bundle = get_model_bundle()
    norm_df, alias_map, label_col = normalize_public_static_dataset(
        df,
        expected_features=bundle.feature_columns,
    )
    valid, err = validate_tabular_dataset(norm_df, expected_features=bundle.feature_columns, require_label=False)
    if not valid:
        return {
            "valid": False,
            "error": err or "Incompatible CSV schema.",
            "alias_map": alias_map,
            "label_col": None,
            "eval_metrics": None,
            "out_df": None,
            "rows_out": [],
        }

    rows_out: list[dict[str, Any]] = []
    y_true: list[int] = []
    y_prob: list[float] = []
    y_pred: list[int] = []
    invert_legit = label_col is not None and label_col.strip().lower() == "legitimate"

    for idx, row in norm_df.head(500).iterrows():
        feat_dict = {col: row[col] for col in bundle.feature_columns}
        pred = explain_and_score_features(bundle, feat_dict, model_key=model_key, top_n=3)
        assess = score_probability(pred.malicious_probability, supported=pred.supported, warning=pred.warning)
        top_driver = pred.top_contributions[0].label if pred.top_contributions else "N/A"
        row_record: dict[str, Any] = {
            "Row": idx,
            "Priority": assess.priority,
            "Risk Score": assess.risk_score,
            "Top SHAP Driver": top_driver,
            "Summary": pred.plain_language_summary,
        }
        if label_col is not None and pred.malicious_probability is not None:
            raw_lbl = int(float(row[label_col]))
            true_mal = 1 - raw_lbl if invert_legit else raw_lbl
            row_record["Ground Truth"] = "Malicious (1)" if true_mal == 1 else "Benign (0)"
            y_true.append(true_mal)
            y_prob.append(float(pred.malicious_probability))
            y_pred.append(1 if float(pred.malicious_probability) >= 0.5 else 0)
        rows_out.append(row_record)

    eval_metrics: dict[str, float | None] | None = None
    if y_true:
        roc = float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) == 2 else None
        eval_metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": roc,
            "inverted_legitimate": 1.0 if invert_legit else 0.0,
        }

    out_df = pd.DataFrame(rows_out).sort_values(by="Risk Score", ascending=False, na_position="last")
    return {
        "valid": True,
        "error": None,
        "alias_map": alias_map,
        "label_col": label_col,
        "eval_metrics": eval_metrics,
        "out_df": out_df,
        "rows_out": rows_out,
    }


def _render_csv_scored_output(
    scored: dict[str, Any],
    download_name: str,
    key_prefix: str,
) -> None:
    """Display alias mapping notice, optional ground-truth metrics, and scored table."""
    alias_map = scored.get("alias_map") or {}
    if alias_map:
        mapped_preview = ", ".join(f"{src} -> {dst}" for src, dst in list(alias_map.items())[:6])
        suffix = f" (+{len(alias_map) - 6} more)" if len(alias_map) > 6 else ""
        st.info(f"Public static-feature schema adapter mapped {len(alias_map)} column(s): `{mapped_preview}{suffix}`")

    eval_metrics = scored.get("eval_metrics")
    if eval_metrics:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{eval_metrics['accuracy']:.3f}")
        m2.metric("Precision", f"{eval_metrics['precision']:.3f}")
        m3.metric("Recall", f"{eval_metrics['recall']:.3f}")
        m4.metric("F1-Score", f"{eval_metrics['f1']:.3f}")
        m5.metric("ROC-AUC", f"{eval_metrics['roc_auc']:.3f}" if eval_metrics["roc_auc"] is not None else "N/A")
        if eval_metrics.get("inverted_legitimate"):
            st.caption("Detected Kaggle-style `legitimate` column (1=Benign, 0=Malicious) and aligned to Malicious=1 for metrics.")

    out_df = scored["out_df"].sort_values(by="Risk Score", ascending=False, na_position="last")
    scores_series = pd.to_numeric(out_df["Risk Score"], errors="coerce")
    total_n = len(out_df)
    high_n = int((scores_series >= 70.0).sum())
    med_n = int(((scores_series >= 35.0) & (scores_series < 70.0)).sum())
    low_n = int((scores_series < 35.0).sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Records", total_n)
    k2.metric("High Priority (≥70)", high_n)
    k3.metric("Medium Priority (35–69)", med_n)
    k4.metric("Low Priority (<35)", low_n)

    tier_filter = st.selectbox(
        "Filter by Risk Tier",
        options=["All", "HIGH (≥70)", "MEDIUM (35–69)", "LOW (<35)"],
        key=f"{key_prefix}_tier_filter",
    )
    if tier_filter == "HIGH (≥70)":
        filtered_df = out_df[scores_series >= 70.0]
    elif tier_filter == "MEDIUM (35–69)":
        filtered_df = out_df[(scores_series >= 35.0) & (scores_series < 70.0)]
    elif tier_filter == "LOW (<35)":
        filtered_df = out_df[scores_series < 35.0]
    else:
        filtered_df = out_df

    st.dataframe(filtered_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download Prioritized Batch CSV",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name=download_name,
        mime="text/csv",
        key=f"{key_prefix}_download",
    )


def render_batch_analysis_view(key_prefix: str = "csv") -> None:
    """Render the strict-schema CSV batch static feature scoring workflow."""
    model_key = st.selectbox(
        "Scoring Model",
        options=["random_forest", "hist_gradient_boosting"],
        format_func=lambda k: "RandomForestClassifier" if k == "random_forest" else "HistGradientBoostingClassifier",
        key=f"{key_prefix}_model",
    )

    sample_col, btn_col = st.columns([1.4, 1.0])
    with sample_col:
        sample_choice = st.selectbox(
            "Sample Public Static Benchmark",
            options=list(SAMPLE_PUBLIC_DATASETS.keys()),
            key=f"{key_prefix}_sample_choice",
        )
    with btn_col:
        st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
        if st.button(
            "Load Sample Public Static Dataset (EMBER / PE Benchmark)",
            key=f"{key_prefix}_load_sample",
            use_container_width=True,
        ):
            sample_path = SAMPLE_PUBLIC_DATASETS[sample_choice]
            if sample_path.exists():
                st.session_state[f"{key_prefix}_demo_path"] = str(sample_path)
            else:
                st.error(f"Sample dataset not found: {sample_path}")

    uploaded = st.file_uploader(
        "Upload Static Feature CSV",
        type=["csv"],
        key=f"{key_prefix}_uploader",
    )

    df: pd.DataFrame | None = None
    source_name = "vortex_sentinel_batch_results.csv"
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        source_name = f"{Path(uploaded.name).stem}_batch_results.csv"
    elif st.session_state.get(f"{key_prefix}_demo_path"):
        demo_path = Path(st.session_state[f"{key_prefix}_demo_path"])
        if demo_path.exists():
            st.caption(f"Loaded demo dataset: `{demo_path}`")
            df = pd.read_csv(demo_path)
            source_name = f"{demo_path.stem}_batch_results.csv"

    if df is not None:
        scored = _score_csv_dataframe(df, model_key=model_key)
        if not scored["valid"]:
            st.error(scored["error"] or "Incompatible CSV schema.")
        else:
            _render_csv_scored_output(scored, download_name=source_name, key_prefix=key_prefix)


def _process_single_pe(raw_bytes: bytes, file_name: str, model_key: str) -> dict[str, Any]:
    """Run existing PE static extraction, ML/SHAP scoring, and SQLite save."""
    bundle = get_model_bundle()
    db = get_db()
    extraction = extract_pe_features(
        file_bytes=raw_bytes,
        file_name=file_name,
        expected_features=bundle.feature_columns,
    )
    if not extraction.supported:
        assessment = score_probability(None, supported=False, warning=extraction.warning)
        summary = extraction.warning or "Unsupported or incomplete PE binary."
        scan_id = db.save_scan(
            file_name=file_name,
            file_type="PE",
            sha256=extraction.sha256,
            md5=extraction.md5,
            classification="Unsupported",
            priority=assessment.priority,
            risk_score=None,
            model_name=None,
            supported=False,
            summary=summary,
            payload={"warning": extraction.warning, "recommendation": assessment.recommendation},
        )
        return {
            "file_name": file_name,
            "detected_type": "PE",
            "risk_display": "Unsupported",
            "risk_score": None,
            "priority": assessment.priority,
            "sqlite_status": f"Saved (#{scan_id})",
            "summary": summary,
            "supported": False,
            "extraction": extraction,
            "prediction": None,
            "assessment": assessment,
            "scan_id": scan_id,
        }

    prediction = score_pe_extraction(bundle, extraction, model_key=model_key, top_n=8)
    assessment = score_probability(
        prediction.malicious_probability,
        supported=prediction.supported,
        warning=prediction.warning,
    )
    secondary_key = "hist_gradient_boosting" if model_key == "random_forest" else "random_forest"
    sec_pred = score_pe_extraction(bundle, extraction, model_key=secondary_key, top_n=3)
    sec_assess = score_probability(sec_pred.malicious_probability, supported=sec_pred.supported)
    triage_brief = generate_triage_brief(
        filename=file_name,
        file_type="PE",
        features={
            **extraction.features,
            "sha256": extraction.sha256,
            "suspicious_apis": extraction.suspicious_apis,
            "suspicious_strings": extraction.suspicious_strings,
        },
        risk_score=assessment.risk_score,
        model_name=prediction.model_name,
        shap_contributions=prediction.top_contributions,
        secondary_score=sec_assess.risk_score,
    )
    scan_id = db.save_scan(
        file_name=file_name,
        file_type="PE",
        sha256=extraction.sha256,
        md5=extraction.md5,
        classification=prediction.risk_tier,
        priority=assessment.priority,
        risk_score=assessment.risk_score,
        model_name=prediction.model_name,
        supported=True,
        summary=prediction.plain_language_summary,
        payload={
            **prediction.to_dict(),
            "recommendation": assessment.recommendation,
            "sha256": extraction.sha256,
            "md5": extraction.md5,
            "triage_brief": triage_brief,
        },
    )
    score_txt = f"{assessment.risk_score:.1f} / 100" if assessment.risk_score is not None else "N/A"
    return {
        "file_name": file_name,
        "detected_type": "PE",
        "risk_display": score_txt,
        "risk_score": assessment.risk_score,
        "priority": assessment.priority,
        "sqlite_status": f"Saved (#{scan_id})",
        "summary": prediction.plain_language_summary,
        "supported": True,
        "extraction": extraction,
        "prediction": prediction,
        "assessment": assessment,
        "triage_brief": triage_brief,
        "scan_id": scan_id,
    }


def _process_single_pdf(
    raw_bytes: bytes,
    file_name: str,
    max_pages: int,
    max_seconds: float,
) -> dict[str, Any]:
    """Run existing zero-execution PDF inspection and SQLite save."""
    bundle = get_model_bundle()
    db = get_db()
    limits = PDFInspectionLimits(max_pages=int(max_pages), max_processing_seconds=float(max_seconds))
    result = inspect_pdf_bytes(
        raw_bytes,
        file_name=file_name,
        limits=limits,
        expected_pe_features=bundle.feature_columns,
    )
    assessment = assess_pdf_priority(result.assessment_tier, result.supported, result.assessment_summary)
    scan_id = db.save_scan(
        file_name=file_name,
        file_type="PDF",
        sha256=result.sha256,
        md5=result.md5,
        classification=result.assessment_tier,
        priority=assessment.priority,
        risk_score=None,
        model_name=None,
        supported=result.supported,
        summary=result.assessment_summary,
        payload={
            **result.to_dict(),
            "recommendation": assessment.recommendation,
        },
    )
    return {
        "file_name": file_name,
        "detected_type": "PDF",
        "risk_display": f"{result.status.upper()} ({result.assessment_tier})",
        "risk_score": None,
        "priority": assessment.priority,
        "sqlite_status": f"Saved (#{scan_id})",
        "summary": result.assessment_summary,
        "supported": result.supported,
        "pdf_result": result,
        "assessment": assessment,
        "scan_id": scan_id,
    }


def _process_single_csv(raw_bytes: bytes, file_name: str, model_key: str) -> dict[str, Any]:
    """Run existing strict-schema CSV validation and batch static feature scoring."""
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        err = f"Failed to parse CSV: {exc}"
        return {
            "file_name": file_name,
            "detected_type": "CSV",
            "risk_display": "Parse Error",
            "risk_score": None,
            "priority": "Unsupported",
            "sqlite_status": "Not Saved (Batch)",
            "summary": err,
            "valid": False,
            "error": err,
            "scored": None,
            "out_df": None,
        }

    scored = _score_csv_dataframe(df, model_key=model_key)
    if not scored["valid"]:
        msg = scored["error"] or "Incompatible CSV schema."
        return {
            "file_name": file_name,
            "detected_type": "CSV",
            "risk_display": "Schema Rejected",
            "risk_score": None,
            "priority": "Unsupported",
            "sqlite_status": "Not Saved (Batch)",
            "summary": msg,
            "valid": False,
            "error": msg,
            "scored": scored,
            "out_df": None,
        }

    rows_out = scored["rows_out"]
    valid_scores = [r["Risk Score"] for r in rows_out if r["Risk Score"] is not None]
    max_score = max(valid_scores) if valid_scores else None
    top_prio = "Low"
    for candidate in ("Critical", "High", "Medium", "Low"):
        if any(r["Priority"] == candidate for r in rows_out):
            top_prio = candidate
            break
    summary = f"Validated {len(rows_out)} row(s); highest risk score: {max_score:.1f}" if max_score is not None else f"Validated {len(rows_out)} row(s)."
    return {
        "file_name": file_name,
        "detected_type": "CSV",
        "risk_display": f"Max {max_score:.1f} ({len(rows_out)} rows)" if max_score is not None else f"{len(rows_out)} rows",
        "risk_score": max_score,
        "priority": top_prio,
        "sqlite_status": "Batch Mode (CSV)",
        "summary": summary,
        "valid": True,
        "error": None,
        "scored": scored,
        "out_df": scored["out_df"],
    }


def process_mixed_uploaded_files(
    uploaded_files: list[Any],
    model_key: str,
    max_pages: int = 50,
    max_seconds: float = 10.0,
) -> list[dict[str, Any]]:
    """Process a mixed list of PE, PDF, and CSV uploads using existing handlers."""
    results: list[dict[str, Any]] = []
    for uploaded in uploaded_files:
        name = uploaded.name
        ext = Path(name).suffix.lower()
        raw_bytes = uploaded.getvalue()
        if ext in PE_EXTENSIONS:
            results.append(_process_single_pe(raw_bytes, name, model_key))
        elif ext == ".pdf":
            results.append(_process_single_pdf(raw_bytes, name, max_pages, max_seconds))
        elif ext == ".csv":
            results.append(_process_single_csv(raw_bytes, name, model_key))
        else:
            results.append(
                {
                    "file_name": name,
                    "detected_type": ext.lstrip(".").upper() or "UNKNOWN",
                    "risk_display": "Unsupported",
                    "risk_score": None,
                    "priority": "Unsupported",
                    "sqlite_status": "Not Saved",
                    "summary": f"Unsupported file extension '{ext or '(none)'}'. Supported: .exe, .dll, .sys, .scr, .pdf, .csv.",
                    "supported": False,
                }
            )
    return results


def render_mixed_triage_results(results: list[dict[str, Any]], key_prefix: str = "wb_mixed") -> None:
    """Render combined summary, unified results table, and per-item drill-down."""
    bundle = get_model_bundle()
    pe_count = sum(1 for r in results if r["detected_type"] == "PE")
    pdf_count = sum(1 for r in results if r["detected_type"] == "PDF")
    csv_count = sum(1 for r in results if r["detected_type"] == "CSV")
    high_crit = sum(1 for r in results if r["priority"] in {"High", "Critical"})
    saved_count = sum(1 for r in results if str(r["sqlite_status"]).startswith("Saved"))

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f'<div class="vortex-bento"><div class="vortex-kpi-label">Files Processed</div>'
            f'<div class="vortex-kpi-value">{len(results)}</div></div>',
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f'<div class="vortex-bento-cyan"><div class="vortex-kpi-label">PE / PDF / CSV</div>'
            f'<div class="vortex-kpi-value" style="font-size:1.55rem;">{pe_count} / {pdf_count} / {csv_count}</div></div>',
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f'<div class="vortex-bento"><div class="vortex-kpi-label">High / Critical</div>'
            f'<div class="vortex-kpi-value" style="color:#FBBF24;">{high_crit}</div></div>',
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f'<div class="vortex-bento-cyan"><div class="vortex-kpi-label">Saved to SQLite</div>'
            f'<div class="vortex-kpi-value" style="color:#2DD4BF;">{saved_count}</div></div>',
            unsafe_allow_html=True,
        )

    if bundle.dataset_metadata.is_synthetic:
        st.info(bundle.dataset_metadata.disclaimer)

    st.markdown(
        '<div class="vortex-bento"><div class="vortex-bento-header">'
        '<span>Combined Multi-File Triage Results</span>'
        '<span class="vortex-mono">ZERO-EXECUTION</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    table_df = pd.DataFrame(
        [
            {
                "File": r["file_name"],
                "Detected Type": r["detected_type"],
                "Risk Score / Status": r["risk_display"],
                "Priority": r["priority"],
                "SQLite Status": r["sqlite_status"],
                "Summary": r["summary"],
            }
            for r in results
        ]
    )
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    st.subheader("Per-File Drill-Down Inspector")
    labels = [f"[{r['detected_type']}] {r['file_name']} — {r['priority']}" for r in results]
    selected_idx = st.selectbox(
        "Select File or Batch to Inspect",
        options=list(range(len(results))),
        format_func=lambda i: labels[i],
        key=f"{key_prefix}_drilldown",
    )
    item = results[int(selected_idx)]
    dtype = item["detected_type"]

    if dtype == "PE":
        extraction = item["extraction"]
        if not item["supported"]:
            st.error(item["summary"])
        else:
            prediction = item["prediction"]
            assessment = item["assessment"]
            render_risk_score_hero(
                assessment.risk_score,
                assessment.priority,
                label=f"PE Static Risk Score — {item['file_name']}",
                subtitle=assessment.disclaimer,
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("File Entropy", f"{extraction.features['file_entropy']:.2f}")
            c2.metric("Authenticode Dir", "Present" if extraction.features["has_authenticode_signature"] else "Absent")
            c3.metric("Model", prediction.model_name)

            st.markdown(
                f'<div class="vortex-bento-cyan">'
                f'<div class="vortex-bento-header"><span>Risk Factor Summary</span>'
                f'<span class="vortex-mono">SHA-256: {extraction.sha256[:16]}...</span></div>'
                f'<p style="margin:0;color:#E2E8F0;font-size:0.92rem;">{prediction.plain_language_summary}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if extraction.warning:
                st.warning(extraction.warning)

            if prediction.top_contributions:
                shap_df = pd.DataFrame(
                    [
                        {
                            "Feature": c.label,
                            "SHAP Impact": c.shap_value,
                            "Direction": "Raises Risk" if c.shap_value > 0 else "Lowers Risk",
                        }
                        for c in prediction.top_contributions
                    ]
                )
                fig = px.bar(
                    shap_df,
                    x="Feature",
                    y="SHAP Impact",
                    color="Direction",
                    color_discrete_map={"Raises Risk": "#8B5CF6", "Lowers Risk": "#2DD4BF"},
                )
                fig.update_traces(hovertemplate="<b>%{x}</b><br>SHAP Impact: %{y:+.4f}<extra></extra>")
                st.plotly_chart(
                    apply_vortex_plotly_theme(fig, "Top Per-File SHAP Contributions"),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

            with st.expander("Extracted Static Features & Indicators"):
                st.json(
                    {
                        "sha256": extraction.sha256,
                        "md5": extraction.md5,
                        "suspicious_apis": extraction.suspicious_apis,
                        "suspicious_strings": extraction.suspicious_strings[:10],
                        "features": extraction.features,
                    }
                )
            if item.get("triage_brief"):
                render_triage_assistant_panel(item["triage_brief"], key_prefix=f"{key_prefix}_pe_{selected_idx}")
            st.success(f"PE scan saved to SQLite history (Scan ID #{item['scan_id']}).")

    elif dtype == "PDF":
        result = item["pdf_result"]
        assessment = item["assessment"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Status", result.status.upper())
        m2.metric("Priority", assessment.priority)
        m3.metric("Pages Inspected", f"{result.pages_inspected} / {result.total_pages}")
        m4.metric("Attachments", len(result.attachments))

        st.markdown(
            f'<div class="vortex-bento-cyan">'
            f'<div class="vortex-bento-header"><span>PDF Triage Assessment</span>'
            f'{render_priority_badge(assessment.priority)}</div>'
            f'<p style="margin:0 0 6px 0;color:#E2E8F0;font-size:0.92rem;">{result.assessment_summary}</p>'
            f'<div class="vortex-mono" style="word-break:break-all;">SHA-256: {result.sha256}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        for w in result.warnings:
            st.warning(w)

        if result.indicators:
            st.subheader("Static Indicators Observed")
            ind_df = pd.DataFrame([i.to_dict() for i in result.indicators])
            st.dataframe(ind_df, use_container_width=True, hide_index=True)

        if result.attachments:
            st.subheader("Embedded Attachments & Routing")
            for att in result.attachments:
                with st.expander(f"{att.filename} ({att.size_bytes:,} bytes) — {att.routing_status}"):
                    st.write(f"**SHA-256:** `{att.sha256}`")
                    st.write(f"**Routing Reason:** {att.review_reason}")
                    if att.pe_extraction and att.pe_extraction.supported:
                        pe_pred = score_pe_extraction(bundle, att.pe_extraction, model_key="random_forest")
                        pe_assess = score_probability(pe_pred.malicious_probability, supported=pe_pred.supported)
                        st.write(f"**Embedded PE Risk Score:** {pe_assess.risk_score} ({pe_assess.priority})")
                        st.write(pe_pred.plain_language_summary)

        with st.expander("Metadata, Links & Epistemic Limitations"):
            st.json({"metadata": result.metadata, "links": result.links[:30], "annotations": result.annotation_summary})
            for lim in result.limitations:
                st.caption(f"• {lim}")
        pdf_score_hint = {"Low": 15.0, "Medium": 50.0, "High": 78.0, "Critical": 90.0}.get(assessment.priority)
        pdf_brief = generate_triage_brief(
            filename=item["file_name"],
            file_type="PDF",
            features={"sha256": result.sha256, "indicators": [i.to_dict() for i in result.indicators]},
            risk_score=pdf_score_hint,
            model_name="Static PDF Structural Inspector",
            shap_contributions=None,
            secondary_score=None,
        )
        render_triage_assistant_panel(pdf_brief, key_prefix=f"{key_prefix}_pdf_{selected_idx}")
        st.success(f"PDF inspection saved to SQLite history (Scan ID #{item['scan_id']}).")

    elif dtype == "CSV":
        if not item["valid"]:
            st.error(item["error"] or "Incompatible CSV schema.")
        else:
            _render_csv_scored_output(
                item["scored"],
                download_name=f"{Path(item['file_name']).stem}_batch_results.csv",
                key_prefix=f"{key_prefix}_{selected_idx}",
            )
    else:
        st.warning(item["summary"])
