"""Vortex Sentinel — Offline Deterministic SOC Triage Investigation Assistant.

Pure-Python, zero-API analyst assistant grounded strictly in extracted static
features, calibrated model risk scores, and SHAP feature attributions.
"""

from __future__ import annotations

from typing import Any


TRIAGE_DISCLAIMER = (
    "Analyst Triage Disclaimer: Findings are probabilistic static triage indicators "
    "based on extracted metadata and synthetic-benchmark ML models — NOT proof of "
    "maliciousness or safety. Never execute unverified binaries directly on a host."
)

PDF_INDICATOR_KEYS: tuple[str, ...] = (
    "/JS",
    "/JavaScript",
    "/OpenAction",
    "/AA",
    "/Launch",
    "/EmbeddedFile",
)


def _tier_from_score(score: float | None) -> str:
    """Map a 0-100 score to LOW / MEDIUM / HIGH triage tier."""
    if score is None:
        return "UNSUPPORTED"
    if score >= 70.0:
        return "HIGH"
    if score >= 35.0:
        return "MEDIUM"
    return "LOW"


def _normalize_shap_items(
    shap_contributions: dict[str, Any] | list[Any] | None,
    features: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    """Extract top 3 risk-increasing and top 2 risk-mitigating drivers.

    Returns ``(top_risk_drivers, top_mitigating_factors, used_shap)``.
    """
    items: list[dict[str, Any]] = []
    if isinstance(shap_contributions, dict):
        for k, v in shap_contributions.items():
            try:
                val = float(v)
            except (TypeError, ValueError):
                continue
            items.append({"feature": str(k), "label": str(k), "shap_value": val})
    elif isinstance(shap_contributions, list):
        for entry in shap_contributions:
            if isinstance(entry, dict):
                feat = str(entry.get("feature", entry.get("label", "unknown")))
                lbl = str(entry.get("label", feat))
                raw_v = entry.get("shap_value", 0.0)
            else:
                feat = str(getattr(entry, "feature", getattr(entry, "label", "unknown")))
                lbl = str(getattr(entry, "label", feat))
                raw_v = getattr(entry, "shap_value", 0.0)
            try:
                val = float(raw_v)
            except (TypeError, ValueError):
                continue
            items.append({"feature": feat, "label": lbl, "shap_value": val})

    if items:
        pos = sorted([i for i in items if i["shap_value"] > 0], key=lambda x: x["shap_value"], reverse=True)[:3]
        neg = sorted([i for i in items if i["shap_value"] < 0], key=lambda x: x["shap_value"])[:2]
        return pos, neg, True

    # Deterministic fallback when SHAP values are omitted: normalized rule weights from features
    fallback_pos: list[dict[str, Any]] = []
    fallback_neg: list[dict[str, Any]] = []
    entropy = float(features.get("file_entropy", 0.0) or 0.0)
    max_sec_ent = float(features.get("max_section_entropy", 0.0) or 0.0)
    sus_apis = float(features.get("suspicious_api_count", 0.0) or 0.0)
    sus_strs = float(features.get("suspicious_string_count", 0.0) or 0.0)
    sus_secs = float(features.get("suspicious_section_names", 0.0) or 0.0)
    v_ratio = float(features.get("virtual_to_raw_ratio", 1.0) or 1.0)
    signed = float(features.get("has_authenticode_signature", 0.0) or 0.0)

    if max(entropy, max_sec_ent) >= 7.0:
        fallback_pos.append({"feature": "file_entropy", "label": "High Shannon Entropy", "shap_value": round((max(entropy, max_sec_ent) - 6.5) * 0.12, 4)})
    if sus_apis > 0:
        fallback_pos.append({"feature": "suspicious_api_count", "label": "Suspicious Windows APIs", "shap_value": round(min(sus_apis * 0.04, 0.35), 4)})
    if sus_strs > 0:
        fallback_pos.append({"feature": "suspicious_string_count", "label": "Suspicious String Indicators", "shap_value": round(min(sus_strs * 0.03, 0.25), 4)})
    if sus_secs > 0 or v_ratio > 2.5:
        fallback_pos.append({"feature": "virtual_to_raw_ratio", "label": "Section / Virtual-to-Raw Anomaly", "shap_value": 0.12})
    if signed == 0:
        fallback_pos.append({"feature": "has_authenticode_signature", "label": "Unsigned Binary (No Authenticode)", "shap_value": 0.08})
    else:
        fallback_neg.append({"feature": "has_authenticode_signature", "label": "Authenticode Directory Present", "shap_value": -0.15})

    if entropy > 0 and entropy < 6.2:
        fallback_neg.append({"feature": "file_entropy", "label": "Normal File Entropy", "shap_value": -0.10})

    fallback_pos = sorted(fallback_pos, key=lambda x: x["shap_value"], reverse=True)[:3]
    fallback_neg = sorted(fallback_neg, key=lambda x: x["shap_value"])[:2]
    return fallback_pos, fallback_neg, False


def generate_triage_brief(
    filename: str,
    file_type: str,
    features: dict[str, Any],
    risk_score: float | None,
    model_name: str,
    shap_contributions: dict[str, Any] | list[Any] | None = None,
    secondary_score: float | None = None,
) -> dict[str, Any]:
    """Build a deterministic, grounded SOC triage brief from static features and SHAP."""
    feats = features or {}
    triage_tier = _tier_from_score(risk_score)
    top_pos, top_neg, used_shap = _normalize_shap_items(shap_contributions, feats)

    findings: list[dict[str, str]] = []
    next_steps: list[str] = []

    # 1. Entropy check
    file_ent = float(feats.get("file_entropy", 0.0) or 0.0)
    max_sec_ent = float(feats.get("max_section_entropy", 0.0) or 0.0)
    if file_ent >= 7.0 or max_sec_ent >= 7.2:
        findings.append(
            {
                "signal": "High Shannon Entropy",
                "severity": "elevated",
                "detail": (
                    f"File entropy={file_ent:.2f}, max section entropy={max_sec_ent:.2f} "
                    "(>= 7.0/7.2 threshold). Suggests potential packing, compression, or obfuscation."
                ),
            }
        )
        next_steps.append("Inspect per-section entropy distribution and printable strings for packer stubs or encrypted payloads.")

    # 2. Suspicious Windows APIs & Strings
    sus_api_cnt = int(float(feats.get("suspicious_api_count", 0) or 0))
    sus_str_cnt = int(float(feats.get("suspicious_string_count", 0) or 0))
    sus_api_names = feats.get("suspicious_apis") or []
    sus_str_names = feats.get("suspicious_strings") or []
    if sus_api_cnt > 0 or sus_str_cnt > 0:
        api_preview = f" ({', '.join(map(str, sus_api_names[:5]))})" if sus_api_names else ""
        str_preview = f" ({', '.join(map(str, sus_str_names[:4]))})" if sus_str_names else ""
        findings.append(
            {
                "signal": "Suspicious API / String Indicators",
                "severity": "elevated",
                "detail": (
                    f"Detected {sus_api_cnt} suspicious Windows API(s){api_preview} and "
                    f"{sus_str_cnt} suspicious string indicator(s){str_preview}."
                ),
            }
        )
        next_steps.append("Review the Import Address Table (IAT) and API call clusters in a static disassembler (e.g., Ghidra/IDA).")

    # 3. Authenticode signature status (PE)
    if str(file_type).upper() == "PE" or "has_authenticode_signature" in feats:
        signed = int(float(feats.get("has_authenticode_signature", 0) or 0))
        if signed == 0:
            findings.append(
                {
                    "signal": "Unsigned Binary",
                    "severity": "advisory",
                    "detail": (
                        "Authenticode signature directory is absent (0). Note: legitimate internal, "
                        "open-source, or legacy utilities can also be unsigned."
                    ),
                }
            )
        else:
            findings.append(
                {
                    "signal": "Authenticode Directory Present",
                    "severity": "mitigating",
                    "detail": "PE security directory is populated (1); verify publisher certificate chain offline.",
                }
            )

    # 4. Structural section padding / anomaly
    v_ratio = float(feats.get("virtual_to_raw_ratio", 1.0) or 1.0)
    sus_sec = int(float(feats.get("suspicious_section_names", 0) or 0))
    if v_ratio > 2.5 or sus_sec > 0:
        findings.append(
            {
                "signal": "PE Section / Size Anomaly",
                "severity": "elevated",
                "detail": (
                    f"Virtual-to-raw ratio={v_ratio:.2f} (> 2.5 threshold) or "
                    f"non-standard section count={sus_sec}. Indicates possible section padding or custom packer."
                ),
            }
        )

    # 5. PDF structural indicators
    pdf_indicators_found: list[str] = []
    raw_inds = feats.get("indicators") or []
    for ind in raw_inds:
        token = str(ind.get("indicator", ind) if isinstance(ind, dict) else ind)
        for key in PDF_INDICATOR_KEYS:
            if key.lower() in token.lower() and key not in pdf_indicators_found:
                pdf_indicators_found.append(key)
    for key in PDF_INDICATOR_KEYS:
        if feats.get(key) and key not in pdf_indicators_found:
            pdf_indicators_found.append(key)

    if pdf_indicators_found or str(file_type).upper() == "PDF":
        if pdf_indicators_found:
            findings.append(
                {
                    "signal": "PDF Active / Structural Indicators",
                    "severity": "elevated",
                    "detail": (
                        f"Observed PDF keywords: {', '.join(pdf_indicators_found)}. "
                        "Inspect object tree statically without rendering or executing scripts."
                    ),
                }
            )
            next_steps.append("Inspect flagged PDF object streams (/JS, /OpenAction, /EmbeddedFile) statically using pdfid/pikepdf; keep preview_allowed=False.")
        else:
            findings.append(
                {
                    "signal": "No High-Risk PDF Action Keywords",
                    "severity": "mitigating",
                    "detail": "No /JS, /OpenAction, /Launch, or /EmbeddedFile indicators observed within inspection limits.",
                }
            )

    # Model consensus check
    consensus: dict[str, Any] = {
        "available": False,
        "badge": "SINGLE MODEL",
        "badge_level": "medium",
        "status": "Single-model evaluation",
        "spread": None,
        "divergent": False,
    }
    if secondary_score is not None and risk_score is not None:
        sec_tier = _tier_from_score(secondary_score)
        spread = abs(float(risk_score) - float(secondary_score))
        if spread >= 30.0 or sec_tier != triage_tier:
            badge = "DIVERGENT — MANUAL REVIEW"
            badge_level = "critical"
            divergent = True
            status = (
                f"Models diverge by {spread:.1f} pts ({triage_tier} vs {sec_tier}) — "
                "bagging (RandomForest) vs. sequential gradient boosting (HistGradientBoosting) "
                "weigh edge-case static features differently; manual analyst review recommended."
            )
        elif spread >= 15.0:
            badge = "MODERATE VARIANCE"
            badge_level = "medium"
            divergent = True
            status = (
                f"Moderate variance ({spread:.1f} pts) within {triage_tier} tier — "
                "bagging vs. boosting ensembles show mild sensitivity to static feature weights."
            )
        else:
            badge = "STRONG CONSENSUS"
            badge_level = "low"
            divergent = False
            status = f"Strong consensus on {triage_tier} tier (spread: {spread:.1f} pts < 15.0)."

        sec_name = (
            "HistGradientBoostingClassifier"
            if "RandomForest" in str(model_name)
            else "RandomForestClassifier"
        )
        consensus = {
            "available": True,
            "badge": badge,
            "badge_level": badge_level,
            "primary_model": model_name,
            "secondary_model": sec_name,
            "primary_score": round(float(risk_score), 1),
            "secondary_score": round(float(secondary_score), 1),
            "primary_tier": triage_tier,
            "secondary_tier": sec_tier,
            "spread": round(spread, 1),
            "divergent": divergent,
            "status": status,
        }

    # Defensive SOC next steps tailored to tier
    sha256 = str(feats.get("sha256") or "").strip()
    hash_hint = f" ({sha256[:16]}...)" if sha256 else ""
    next_steps.insert(0, f"Query file SHA-256{hash_hint} in offline/internal threat-intel and allowlist repositories.")

    if triage_tier == "HIGH":
        next_steps.append("Quarantine artifact and detonate ONLY in an isolated, air-gapped sandbox after completing static review—never execute on host.")
    elif triage_tier == "MEDIUM":
        next_steps.append("Correlate static indicators with provenance (email/download source) and verify publisher signature before whitelisting.")
    else:
        next_steps.append("Document static triage outcome; re-verify only if behavioral telemetry or provenance raises suspicion.")

    # Deduplicate while preserving order and cap at 4 steps
    dedup_steps: list[str] = []
    for step in next_steps:
        if step not in dedup_steps:
            dedup_steps.append(step)

    return {
        "filename": filename,
        "file_type": file_type,
        "sha256": sha256 or None,
        "model_name": model_name,
        "risk_score": round(float(risk_score), 1) if risk_score is not None else None,
        "triage_tier": triage_tier,
        "attribution_source": "SHAP" if used_shap else "Rule-Weight Fallback",
        "top_risk_drivers": top_pos,
        "top_mitigating_factors": top_neg,
        "findings": findings,
        "static_indicators": {
            "file_entropy": feats.get("file_entropy"),
            "max_section_entropy": feats.get("max_section_entropy"),
            "suspicious_api_count": feats.get("suspicious_api_count"),
            "suspicious_apis": (feats.get("suspicious_apis") or [])[:10],
            "suspicious_string_count": feats.get("suspicious_string_count"),
            "has_authenticode_signature": feats.get("has_authenticode_signature"),
            "virtual_to_raw_ratio": feats.get("virtual_to_raw_ratio"),
            "pdf_keywords": pdf_indicators_found,
        },
        "model_consensus": consensus,
        "next_steps": dedup_steps[:4],
        "disclaimer": TRIAGE_DISCLAIMER,
    }


def format_triage_markdown(brief: dict[str, Any]) -> str:
    """Render a deterministic triage brief dictionary as a structured SOC Markdown report."""
    score_txt = f"{brief['risk_score']:.1f} / 100" if brief.get("risk_score") is not None else "N/A"
    consensus = brief.get("model_consensus") or {}
    badge = consensus.get("badge", "SINGLE MODEL")
    sec_line = (
        f"- **Secondary Model ({consensus.get('secondary_model', 'N/A')}):** "
        f"{consensus.get('secondary_score', 'N/A')} / 100 (**{consensus.get('secondary_tier', 'N/A')}**)"
        if consensus.get("available")
        else "- **Secondary Model:** Not evaluated"
    )
    ind = brief.get("static_indicators") or {}
    ind_parts = [f"{k}={v}" for k, v in ind.items() if v not in (None, [], "")]
    lines = [
        f"### Vortex Sentinel — SOC Inspection Report (`{brief.get('filename', 'unknown')}`)",
        f"- **Type:** {brief.get('file_type', 'N/A')} | **SHA-256:** `{brief.get('sha256') or 'N/A'}`",
        f"- **Primary Model ({brief.get('model_name', 'N/A')}):** {score_txt} (**{brief.get('triage_tier', 'UNSUPPORTED')}**)",
        sec_line,
        f"- **Model Agreement:** **{badge}** — {consensus.get('status', 'N/A')}",
        f"- **Static Indicators:** {', '.join(ind_parts) if ind_parts else 'See findings below'}",
        "",
        f"#### Top Risk-Increasing Drivers ({brief.get('attribution_source', 'SHAP')})",
    ]
    pos = brief.get("top_risk_drivers") or []
    if pos:
        for d in pos:
            lines.append(f"- `{d['label']}` (`{float(d['shap_value']):+.4f}`)")
    else:
        lines.append("- None observed.")

    lines.append("")
    lines.append("#### Top Risk-Mitigating Factors")
    neg = brief.get("top_mitigating_factors") or []
    if neg:
        for d in neg:
            lines.append(f"- `{d['label']}` (`{float(d['shap_value']):+.4f}`)")
    else:
        lines.append("- None observed.")

    lines.append("")
    lines.append("#### Static Rule Findings")
    for f in brief.get("findings") or []:
        lines.append(f"- **{f['signal']}**: {f['detail']}")

    lines.append("")
    lines.append("#### Recommended SOC Next Steps")
    for idx, step in enumerate(brief.get("next_steps") or [], start=1):
        lines.append(f"{idx}. {step}")

    lines.append("")
    lines.append(f"> {brief.get('disclaimer', TRIAGE_DISCLAIMER)}")
    return "\n".join(lines)


def answer_analyst_question(question: str, brief: dict[str, Any]) -> str:
    """Deterministically answer common SOC analyst questions from the computed brief."""
    q = (question or "").strip().lower()
    pos = brief.get("top_risk_drivers") or []
    neg = brief.get("top_mitigating_factors") or []
    findings = brief.get("findings") or []
    consensus = brief.get("model_consensus") or {}
    tier = brief.get("triage_tier", "UNSUPPORTED")
    score = brief.get("risk_score")
    score_txt = f"{score:.1f}/100" if score is not None else "N/A"

    if any(k in q for k in ("model", "compare", "agree", "consensus", "randomforest", "hist")):
        if consensus.get("available"):
            return (
                f"**Model Consensus Check:** {consensus['status']} "
                f"(Primary `{brief.get('model_name')}`: {consensus['primary_score']}/100 [{consensus['primary_tier']}], "
                f"Secondary: {consensus['secondary_score']}/100 [{consensus['secondary_tier']}])."
            )
        return (
            f"**Model Info:** Scored by `{brief.get('model_name')}` ({score_txt}, {tier}). "
            "Run with a secondary model score to evaluate cross-model consensus."
        )

    if any(k in q for k in ("why", "score", "flagged", "driver", "shap")):
        drivers_txt = ", ".join(f"{d['label']} ({float(d['shap_value']):+.4f})" for d in pos) or "no strong positive drivers"
        findings_txt = "; ".join(f"{f['signal']}: {f['detail']}" for f in findings[:3]) or "no elevated rule triggers"
        return (
            f"**Score Breakdown ({score_txt}, {tier}):** Primary risk-increasing drivers ({brief.get('attribution_source')}) "
            f"are {drivers_txt}. Static findings: {findings_txt}."
        )

    if any(k in q for k in ("next", "step", "action", "playbook", "what should i do")):
        steps = brief.get("next_steps") or []
        return "**Recommended Defensive Next Steps:**\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))

    if any(k in q for k in ("false positive", "benign", "safe", "legit")):
        mit_txt = ", ".join(f"{d['label']} ({float(d['shap_value']):+.4f})" for d in neg) or "none"
        return (
            f"**False-Positive Assessment:** Risk-mitigating factors observed: {mit_txt}. "
            "High entropy or missing Authenticode signatures also occur in legitimate packed installers, "
            "open-source tools, and internal utilities. Verify file hash and publisher provenance before escalating. "
            f"{brief.get('disclaimer', TRIAGE_DISCLAIMER)}"
        )

    if any(k in q for k in ("execute", "run", "sandbox", "open", "preview")):
        return (
            "**Zero-Execution Guardrail:** Vortex Sentinel never executes binaries or renders PDFs (`preview_allowed=False`). "
            "Executing untrusted artifacts on an analyst workstation risks host compromise. "
            "Complete static IAT/string/object-tree review first, and only detonate `HIGH`-risk samples inside an isolated, air-gapped sandbox."
        )

    return format_triage_markdown(brief)
