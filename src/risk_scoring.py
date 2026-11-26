"""Vortex Sentinel — Risk Scoring & Investigation Priority Mapping.

Converts calibrated model probability estimates into a 0-100 risk score and
assigns investigation priorities (Low, Medium, High, Critical) using clearly
documented thresholds. Unsupported or incomplete analyses are never assigned
fabricated numeric scores.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

# Documented 0-100 risk score thresholds for SOC investigation priority.
PRIORITY_THRESHOLDS: tuple[tuple[float, str, str], ...] = (
    (
        80.0,
        "Critical",
        "Score >= 80.0: Strong static risk indicators; isolate/quarantine artifact and escalate for sandbox/reverse-engineering review.",
    ),
    (
        55.0,
        "High",
        "Score 55.0-79.9: Elevated static risk; hold execution, verify provenance, and inspect flagged imports/strings.",
    ),
    (
        25.0,
        "Medium",
        "Score 25.0-54.9: Mixed static signals; verify publisher signature/reputation before permitting execution.",
    ),
    (
        0.0,
        "Low",
        "Score < 25.0: Static profile aligns with low-risk software; log event and rely on standard endpoint monitoring.",
    ),
)

RISK_ESTIMATE_DISCLAIMER = (
    "Risk Estimate Only: Scores are statistical risk estimates derived from static "
    "features—not proof of maliciousness or a guarantee of safety."
)


@dataclass(frozen=True)
class RiskAssessment:
    """Structured risk score, priority tier, and epistemic disclaimer."""

    supported: bool
    risk_score: float | None
    probability: float | None
    priority: str  # "Low" | "Medium" | "High" | "Critical" | "Unsupported"
    threshold_explanation: str
    recommendation: str
    disclaimer: str = RISK_ESTIMATE_DISCLAIMER
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_probability(
    probability: float | None,
    supported: bool = True,
    warning: str | None = None,
) -> RiskAssessment:
    """Convert a valid probability [0, 1] into a 0-100 risk score and priority.

    Never fabricates a score when ``supported`` is False or ``probability`` is
    missing/non-finite.
    """
    if not supported or probability is None:
        return RiskAssessment(
            supported=False,
            risk_score=None,
            probability=None,
            priority="Unsupported",
            threshold_explanation="Analysis unsupported or incomplete; no numeric risk score was fabricated.",
            recommendation="Route artifact for manual SOC inspection in an isolated environment.",
            warning=warning or "Unsupported or incomplete analysis.",
        )

    prob = float(probability)
    if not math.isfinite(prob):
        return RiskAssessment(
            supported=False,
            risk_score=None,
            probability=None,
            priority="Unsupported",
            threshold_explanation="Non-finite probability estimate; no numeric risk score was fabricated.",
            recommendation="Route artifact for manual SOC inspection.",
            warning=warning or "Model returned a non-finite probability.",
        )

    clamped_prob = max(0.0, min(1.0, prob))
    risk_score = round(clamped_prob * 100.0, 2)

    for min_score, priority, explanation in PRIORITY_THRESHOLDS:
        if risk_score >= min_score:
            return RiskAssessment(
                supported=True,
                risk_score=risk_score,
                probability=round(clamped_prob, 4),
                priority=priority,
                threshold_explanation=explanation,
                recommendation=explanation,
                warning=warning,
            )

    return RiskAssessment(
        supported=True,
        risk_score=risk_score,
        probability=round(clamped_prob, 4),
        priority="Low",
        threshold_explanation=PRIORITY_THRESHOLDS[-1][2],
        recommendation=PRIORITY_THRESHOLDS[-1][2],
        warning=warning,
    )


def assess_pdf_priority(assessment_tier: str, supported: bool, summary: str) -> RiskAssessment:
    """Map static PDF inspection tiers to investigation priority without fabricating ML scores."""
    if not supported or assessment_tier == "INCOMPLETE_INSPECTION":
        return RiskAssessment(
            supported=False,
            risk_score=None,
            probability=None,
            priority="Unsupported",
            threshold_explanation="PDF inspection was incomplete (encrypted, malformed, oversized, or timed out).",
            recommendation=summary,
            warning="Incomplete PDF inspection; no numeric risk score assigned.",
        )

    tier_map = {
        "NO_INDICATORS_OBSERVED": (
            "Low",
            "Zero static indicators observed (does NOT prove the PDF is safe).",
        ),
        "SINGLE_INDICATOR_REVIEW": (
            "Medium",
            "Single static indicator observed; never labeled malicious from one indicator alone.",
        ),
        "MULTIPLE_INDICATORS_SUSPICIOUS": (
            "High",
            "Multiple distinct static indicator categories observed; escalate for isolated review.",
        ),
    }
    priority, explanation = tier_map.get(
        assessment_tier,
        ("Medium", "Manual analyst review recommended."),
    )
    return RiskAssessment(
        supported=True,
        risk_score=None,
        probability=None,
        priority=priority,
        threshold_explanation=explanation,
        recommendation=summary,
    )
