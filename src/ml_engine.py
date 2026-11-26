"""Vortex Sentinel — Explainable Static Malware Risk ML Engine.

Provides reproducible stratified model training with zero data leakage,
clearly labeled synthetic/simulated static-feature benchmark generation when
no external dataset is supplied, model persistence via ``joblib`` under
``models/``, and per-file SHAP explanations in plain language.

EPISTEMIC & ETHICS SAFEGUARDS:
- The built-in dataset generator creates SYNTHETIC / SIMULATED static-feature
  benchmark data only. It does NOT represent real-world malware validation.
- Single static features NEVER prove a file is malicious or benign.
- Model outputs are calibrated statistical risk estimates, NOT proof.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

try:
    from src.feature_extraction import (
        PE_FEATURE_COLUMNS,
        PE_FEATURE_LABELS,
        PEFeatureExtractionResult,
        SCHEMA_VERSION,
        validate_feature_compatibility,
        validate_tabular_dataset,
    )
except ImportError:  # Allow running directly from within src/
    from feature_extraction import (  # type: ignore[no-redef]
        PE_FEATURE_COLUMNS,
        PE_FEATURE_LABELS,
        PEFeatureExtractionResult,
        SCHEMA_VERSION,
        validate_feature_compatibility,
        validate_tabular_dataset,
    )

DEFAULT_MODELS_DIR = Path("models")
DEFAULT_BUNDLE_FILENAME = "vortex_sentinel_bundle.joblib"

SYNTHETIC_DATASET_NOTICE = (
    "SYNTHETIC / SIMULATED BENCHMARK DATA ONLY: Metrics reflect simulated static PE "
    "feature distributions and do NOT constitute real-world malware detection validation."
)

ANALYSIS_DISCLAIMER = (
    "Risk Estimate Only: Scores are probabilistic risk estimates derived from static "
    "structure—not proof of maliciousness or safety. No single static feature "
    "proves malice (e.g., legitimate software can be compressed or unsigned)."
)

RISK_TIERS: tuple[tuple[float, str], ...] = (
    (80.0, "CRITICAL"),
    (55.0, "HIGH / SUSPICIOUS"),
    (25.0, "MODERATE"),
    (0.0, "LOW"),
)


@dataclass(frozen=True)
class DatasetMetadata:
    """Provenance and synthetic-vs-external disclosure for training data."""

    is_synthetic: bool
    source_label: str
    n_samples: int
    positive_count: int
    negative_count: int
    disclaimer: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SplitAudit:
    """Audit record confirming stratified splitting and zero train/test leakage."""

    train_size: int
    test_size: int
    train_positive_rate: float
    test_positive_rate: float
    stratified: bool
    random_state: int
    zero_leakage_verified: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelMetrics:
    """Held-out evaluation metrics for a trained binary risk classifier."""

    model_key: str
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    false_positive_rate: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ShapContribution:
    """Plain-language per-file SHAP contribution for one static feature."""

    feature: str
    label: str
    value: float
    shap_value: float
    abs_shap: float
    direction: str
    plain_explanation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RiskPredictionResult:
    """Complete risk estimate, separated positive/negative SHAP drivers, and caveats."""

    supported: bool
    warning: str | None
    model_key: str
    model_name: str
    risk_score: float | None
    malicious_probability: float | None
    risk_tier: str
    positive_contributions: list[ShapContribution] = field(default_factory=list)
    negative_contributions: list[ShapContribution] = field(default_factory=list)
    top_contributions: list[ShapContribution] = field(default_factory=list)
    plain_language_summary: str = ""
    analysis_disclaimer: str = ANALYSIS_DISCLAIMER
    is_synthetic_model: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "warning": self.warning,
            "model_key": self.model_key,
            "model_name": self.model_name,
            "risk_score": self.risk_score,
            "malicious_probability": self.malicious_probability,
            "risk_tier": self.risk_tier,
            "positive_contributions": [c.to_dict() for c in self.positive_contributions],
            "negative_contributions": [c.to_dict() for c in self.negative_contributions],
            "top_contributions": [c.to_dict() for c in self.top_contributions],
            "plain_language_summary": self.plain_language_summary,
            "analysis_disclaimer": self.analysis_disclaimer,
            "is_synthetic_model": self.is_synthetic_model,
        }


@dataclass
class ModelBundle:
    """Persisted bundle holding both trained models, metrics, schema, and provenance."""

    rf_model: RandomForestClassifier
    hgb_model: HistGradientBoostingClassifier
    rf_metrics: ModelMetrics
    hgb_metrics: ModelMetrics
    feature_columns: tuple[str, ...]
    dataset_metadata: DatasetMetadata
    split_audit: SplitAudit
    schema_version: str = SCHEMA_VERSION
    trained_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def get_model(self, model_key: str) -> tuple[Any, ModelMetrics]:
        if model_key == "random_forest":
            return self.rf_model, self.rf_metrics
        if model_key == "hist_gradient_boosting":
            return self.hgb_model, self.hgb_metrics
        raise ValueError(
            f"Unsupported model_key '{model_key}'. Expected 'random_forest' or 'hist_gradient_boosting'."
        )


def generate_synthetic_benchmark_dataset(
    n_samples: int = 2400,
    malware_ratio: float = 0.45,
    random_state: int = 42,
) -> tuple[pd.DataFrame, DatasetMetadata]:
    """Generate a SYNTHETIC / SIMULATED static PE feature benchmark dataset.

    Used ONLY when no external labeled static-feature dataset is provided.
    Produces simulated benign and suspicious structural profiles (including
    overlapping edge cases like unsigned/compressed benign installers) without
    using real malware binaries or personal/financial data.
    """
    if n_samples < 100:
        raise ValueError("n_samples must be at least 100 for stratified train/test splitting.")
    if not 0.1 <= malware_ratio <= 0.9:
        raise ValueError("malware_ratio must be between 0.1 and 0.9.")

    rng = np.random.default_rng(random_state)
    n_mal = int(round(n_samples * malware_ratio))
    n_ben = n_samples - n_mal

    n_ben_std = int(round(n_ben * 0.80))
    n_ben_edge = n_ben - n_ben_std

    n_mal_packed = int(round(n_mal * 0.52))
    n_mal_unpacked = n_mal - n_mal_packed

    def _simulate_group(
        count: int,
        label: int,
        size_range: tuple[float, float],
        file_ent: tuple[float, float],
        sec_cnt: tuple[int, int],
        mean_sec_ent: tuple[float, float],
        max_sec_ent: tuple[float, float],
        susp_sec: tuple[float, int],
        imports: tuple[int, int],
        dlls: tuple[int, int],
        susp_api: tuple[float, int],
        sig_prob: float,
        ascii_str: tuple[int, int],
        susp_str: tuple[float, int],
        v_ratio: tuple[float, float],
    ) -> pd.DataFrame:
        log_min, log_max = math.log10(size_range[0]), math.log10(size_range[1])
        file_size = np.round(10 ** rng.uniform(log_min, log_max, size=count)).astype(float)
        file_entropy = np.clip(rng.normal(file_ent[0], file_ent[1], size=count), 1.5, 7.99)
        section_count = rng.integers(sec_cnt[0], sec_cnt[1] + 1, size=count).astype(float)
        mean_section_entropy = np.clip(
            rng.normal(mean_sec_ent[0], mean_sec_ent[1], size=count),
            1.5,
            7.98,
        )
        max_section_entropy = np.clip(
            np.maximum(
                mean_section_entropy + rng.uniform(0.15, 0.85, size=count),
                rng.normal(max_sec_ent[0], max_sec_ent[1], size=count),
            ),
            2.0,
            8.0,
        )
        suspicious_section_names = np.minimum(
            section_count,
            np.clip(rng.poisson(susp_sec[0], size=count), 0, susp_sec[1]),
        ).astype(float)
        import_count = rng.integers(imports[0], imports[1] + 1, size=count).astype(float)
        imported_dll_count = np.minimum(
            import_count,
            rng.integers(dlls[0], dlls[1] + 1, size=count),
        ).astype(float)
        suspicious_api_count = np.minimum(
            import_count,
            np.clip(rng.poisson(susp_api[0], size=count), 0, susp_api[1]),
        ).astype(float)
        has_authenticode_signature = rng.binomial(1, sig_prob, size=count).astype(float)
        ascii_string_count = rng.integers(ascii_str[0], ascii_str[1] + 1, size=count).astype(float)
        suspicious_string_count = np.minimum(
            ascii_string_count,
            np.clip(rng.poisson(susp_str[0], size=count), 0, susp_str[1]),
        ).astype(float)
        virtual_to_raw_ratio = np.clip(rng.normal(v_ratio[0], v_ratio[1], size=count), 0.5, 25.0)

        return pd.DataFrame(
            {
                "file_size": file_size,
                "file_entropy": np.round(file_entropy, 4),
                "section_count": section_count,
                "mean_section_entropy": np.round(mean_section_entropy, 4),
                "max_section_entropy": np.round(max_section_entropy, 4),
                "suspicious_section_names": suspicious_section_names,
                "import_count": import_count,
                "imported_dll_count": imported_dll_count,
                "suspicious_api_count": suspicious_api_count,
                "has_authenticode_signature": has_authenticode_signature,
                "ascii_string_count": ascii_string_count,
                "suspicious_string_count": suspicious_string_count,
                "virtual_to_raw_ratio": np.round(virtual_to_raw_ratio, 4),
                "label": np.full(count, label, dtype=int),
            }
        )

    frames = [
        # Simulated standard signed benign software
        _simulate_group(
            count=n_ben_std,
            label=0,
            size_range=(45_000, 12_000_000),
            file_ent=(5.55, 0.55),
            sec_cnt=(4, 7),
            mean_sec_ent=(4.95, 0.55),
            max_sec_ent=(6.20, 0.45),
            susp_sec=(0.15, 2),
            imports=(45, 260),
            dlls=(4, 18),
            susp_api=(1.4, 7),
            sig_prob=0.82,
            ascii_str=(350, 3200),
            susp_str=(0.6, 4),
            v_ratio=(1.05, 0.14),
        ),
        # Simulated unsigned / compressed benign tools & installers (edge-case overlap)
        _simulate_group(
            count=n_ben_edge,
            label=0,
            size_range=(80_000, 18_000_000),
            file_ent=(6.65, 0.45),
            sec_cnt=(4, 8),
            mean_sec_ent=(5.80, 0.48),
            max_sec_ent=(7.05, 0.40),
            susp_sec=(0.60, 2),
            imports=(25, 160),
            dlls=(3, 12),
            susp_api=(2.8, 9),
            sig_prob=0.35,
            ascii_str=(150, 1400),
            susp_str=(1.8, 7),
            v_ratio=(1.45, 0.45),
        ),
        # Simulated packed / obfuscated high-risk binaries
        _simulate_group(
            count=n_mal_packed,
            label=1,
            size_range=(18_000, 1_800_000),
            file_ent=(7.35, 0.35),
            sec_cnt=(2, 6),
            mean_sec_ent=(6.75, 0.40),
            max_sec_ent=(7.68, 0.22),
            susp_sec=(2.1, 5),
            imports=(4, 42),
            dlls=(1, 5),
            susp_api=(5.5, 16),
            sig_prob=0.05,
            ascii_str=(35, 320),
            susp_str=(3.2, 14),
            v_ratio=(3.40, 1.80),
        ),
        # Simulated unpacked dropper / loader static profiles
        _simulate_group(
            count=n_mal_unpacked,
            label=1,
            size_range=(28_000, 3_500_000),
            file_ent=(6.35, 0.55),
            sec_cnt=(5, 10),
            mean_sec_ent=(5.75, 0.50),
            max_sec_ent=(6.85, 0.48),
            susp_sec=(1.1, 4),
            imports=(35, 175),
            dlls=(3, 14),
            susp_api=(8.5, 24),
            sig_prob=0.10,
            ascii_str=(140, 1200),
            susp_str=(6.2, 22),
            v_ratio=(1.55, 0.65),
        ),
    ]

    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    metadata = DatasetMetadata(
        is_synthetic=True,
        source_label="SYNTHETIC / SIMULATED STATIC-FEATURE BENCHMARK",
        n_samples=len(df),
        positive_count=int(df["label"].sum()),
        negative_count=int((df["label"] == 0).sum()),
        disclaimer=SYNTHETIC_DATASET_NOTICE,
    )
    return df, metadata


def _evaluate_classifier(
    model_key: str,
    model_name: str,
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> ModelMetrics:
    """Compute held-out binary classification metrics including false-positive rate."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    unique_classes = np.unique(y_true)
    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(unique_classes) == 2 else 0.0

    return ModelMetrics(
        model_key=model_key,
        model_name=model_name,
        accuracy=round(float(accuracy_score(y_true, y_pred)), 4),
        precision=round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        recall=round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        f1_score=round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        roc_auc=round(roc_auc, 4),
        false_positive_rate=round(fpr, 4),
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
    )


def train_models(
    dataset: pd.DataFrame | None = None,
    feature_columns: Sequence[str] = PE_FEATURE_COLUMNS,
    test_size: float = 0.2,
    random_state: int = 42,
    dataset_source_label: str | None = None,
) -> ModelBundle:
    """Train RandomForestClassifier and HistGradientBoostingClassifier on identical splits.

    Enforces strict schema validation, reproducible stratified splitting, and
    disjoint train/test verification so zero data leakage occurs.
    """
    cols = tuple(feature_columns)

    if dataset is None:
        df, ds_meta = generate_synthetic_benchmark_dataset(random_state=random_state)
    else:
        valid, err_msg = validate_tabular_dataset(dataset, expected_features=cols, require_label=True)
        if not valid:
            raise ValueError(err_msg or "Invalid external dataset.")
        df = dataset.reset_index(drop=True).copy()
        ds_meta = DatasetMetadata(
            is_synthetic=False,
            source_label=dataset_source_label or "External User-Provided Static Dataset",
            n_samples=len(df),
            positive_count=int(df["label"].sum()),
            negative_count=int((df["label"] == 0).sum()),
            disclaimer=(
                "External static-feature dataset provided by user. Held-out metrics "
                "reflect only the split distribution of that dataset."
            ),
        )

    X = df.loc[:, list(cols)].astype(float)
    y = df["label"].astype(int)

    # Split strictly BEFORE any model fitting to guarantee zero data leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    overlap = set(X_train.index) & set(X_test.index)
    if overlap:
        raise RuntimeError("Train/test split leakage detected: overlapping indices found.")

    split_audit = SplitAudit(
        train_size=len(X_train),
        test_size=len(X_test),
        train_positive_rate=round(float(y_train.mean()), 4),
        test_positive_rate=round(float(y_test.mean()), 4),
        stratified=True,
        random_state=random_state,
        zero_leakage_verified=True,
    )

    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]
    rf_pred = (rf_prob >= 0.5).astype(int)
    rf_metrics = _evaluate_classifier(
        "random_forest",
        "RandomForestClassifier",
        y_test,
        rf_pred,
        rf_prob,
    )

    hgb_model = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=8,
        learning_rate=0.06,
        random_state=random_state,
    )
    hgb_model.fit(X_train, y_train)
    hgb_prob = hgb_model.predict_proba(X_test)[:, 1]
    hgb_pred = (hgb_prob >= 0.5).astype(int)
    hgb_metrics = _evaluate_classifier(
        "hist_gradient_boosting",
        "HistGradientBoostingClassifier",
        y_test,
        hgb_pred,
        hgb_prob,
    )

    return ModelBundle(
        rf_model=rf_model,
        hgb_model=hgb_model,
        rf_metrics=rf_metrics,
        hgb_metrics=hgb_metrics,
        feature_columns=cols,
        dataset_metadata=ds_meta,
        split_audit=split_audit,
    )


def save_model_bundle(
    bundle: ModelBundle,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    filename: str = DEFAULT_BUNDLE_FILENAME,
) -> Path:
    """Persist trained model bundle with ``joblib`` under ``models/``."""
    target_dir = Path(models_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename
    joblib.dump(bundle, target_path)
    return target_path


def load_model_bundle(
    model_path: str | Path = DEFAULT_MODELS_DIR / DEFAULT_BUNDLE_FILENAME,
) -> ModelBundle:
    """Load a persisted ``ModelBundle`` from disk via ``joblib``."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Saved model bundle not found at '{path}'.")
    bundle = joblib.load(path)
    if not isinstance(bundle, ModelBundle):
        raise ValueError(f"Artifact at '{path}' is not a valid Vortex Sentinel ModelBundle.")
    return bundle


def load_or_train_model_bundle(
    model_path: str | Path = DEFAULT_MODELS_DIR / DEFAULT_BUNDLE_FILENAME,
    random_state: int = 42,
) -> ModelBundle:
    """Load persisted models under ``models/`` or train and save synthetic benchmark models."""
    path = Path(model_path)
    if path.is_file():
        return load_model_bundle(path)
    bundle = train_models(random_state=random_state)
    save_model_bundle(bundle, models_dir=path.parent, filename=path.name)
    return bundle


def _extract_positive_class_shap(model: Any, X_row: pd.DataFrame) -> np.ndarray:
    """Extract 1D class-1 SHAP values across RF (3D/list) and HGB (2D) TreeExplainer outputs."""
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_row, check_additivity=False)
    if isinstance(raw, list):
        arr = np.asarray(raw[1] if len(raw) > 1 else raw[0], dtype=float)
        return arr[0]
    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 3:
        class_idx = 1 if arr.shape[2] > 1 else 0
        return arr[0, :, class_idx]
    if arr.ndim == 2:
        return arr[0]
    return arr.ravel()


def _classify_risk_tier(risk_score: float) -> str:
    """Map calibrated 0-100 risk score to SOC triage tier."""
    for threshold, tier in RISK_TIERS:
        if risk_score >= threshold:
            return tier
    return "LOW"


def _format_plain_contribution(feature: str, value: float, shap_val: float) -> str:
    """Translate one SHAP value into clear analyst-friendly language."""
    label = PE_FEATURE_LABELS.get(feature, feature)
    val_str = f"{int(value):,}" if float(value).is_integer() else f"{value:.2f}"
    if shap_val > 0:
        effect = f"pushed the risk estimate upward ({shap_val:+.4f})"
    elif shap_val < 0:
        effect = f"pulled the risk estimate downward ({shap_val:+.4f})"
    else:
        effect = "had negligible impact (0.0000)"

    notes = {
        "file_entropy": "elevated byte randomness can occur in packed, encrypted, or compressed files"
        if value >= 7.0
        else "byte randomness is within typical uncompressed ranges",
        "max_section_entropy": "high section entropy can indicate compressed/encrypted sections or legitimate installers"
        if value >= 7.2
        else "section entropy is within normal bounds",
        "mean_section_entropy": "average section entropy reflects overall code/resource density",
        "suspicious_section_names": "non-standard section names warrant checking for custom packers or compilers"
        if value >= 1
        else "section names follow standard PE conventions",
        "suspicious_api_count": "dual-use Windows APIs (memory, process, registry, or network) are present"
        if value >= 4
        else "few sensitive dual-use Windows APIs imported",
        "has_authenticode_signature": "Authenticode directory entry is present (presence alone is not cryptographic trust verification)"
        if value >= 1
        else "no Authenticode certificate directory entry found (common in both internal tools and suspicious binaries)",
        "suspicious_string_count": "URL, shell, or system command strings detected in static text"
        if value >= 3
        else "few command or URL string patterns observed",
        "virtual_to_raw_ratio": "virtual size significantly exceeds raw disk size (often seen in packed stubs)"
        if value >= 2.0
        else "virtual and raw section sizes are closely aligned",
        "import_count": "very few imported APIs (sometimes seen in packed stubs or minimal binaries)"
        if value <= 25
        else "typical imported function count",
    }
    context = notes.get(feature, "observed static structural measurement")
    return f"{label} = {val_str}: {effect} — {context}."


def _build_plain_summary(
    risk_score: float,
    risk_tier: str,
    model_name: str,
    pos_drivers: list[ShapContribution],
    neg_drivers: list[ShapContribution],
) -> str:
    """Create a balanced plain-language risk summary with explicit epistemic caveats."""
    pos_names = ", ".join(c.label for c in pos_drivers[:3]) or "none"
    neg_names = ", ".join(c.label for c in neg_drivers[:3]) or "none"
    return (
        f"{risk_tier} risk estimate ({risk_score:.1f}/100 via {model_name}). "
        f"Main factors raising the estimate: {pos_names}. "
        f"Main factors lowering the estimate: {neg_names}. "
        "Note: individual static indicators do not prove malice, and this score is a "
        "triage risk estimate—not proof of maliciousness or safety."
    )


def explain_and_score_features(
    bundle: ModelBundle,
    features: Mapping[str, Any],
    model_key: str = "random_forest",
    top_n: int = 6,
) -> RiskPredictionResult:
    """Validate schema compatibility, score static features, and compute SHAP drivers."""
    model, metrics = bundle.get_model(model_key)

    compat = validate_feature_compatibility(
        features,
        expected_features=bundle.feature_columns,
        strict_extra=False,
    )
    if not compat.compatible or compat.aligned_features is None:
        return RiskPredictionResult(
            supported=False,
            warning=compat.warning,
            model_key=model_key,
            model_name=metrics.model_name,
            risk_score=None,
            malicious_probability=None,
            risk_tier="UNSUPPORTED",
            plain_language_summary=compat.warning or "Unsupported feature schema.",
            is_synthetic_model=bundle.dataset_metadata.is_synthetic,
        )

    X_row = pd.DataFrame([compat.aligned_features], columns=list(bundle.feature_columns))
    prob = float(model.predict_proba(X_row)[0, 1])
    risk_score = round(prob * 100.0, 2)
    risk_tier = _classify_risk_tier(risk_score)

    shap_values = _extract_positive_class_shap(model, X_row)

    contributions: list[ShapContribution] = []
    for idx, col in enumerate(bundle.feature_columns):
        s_val = float(shap_values[idx])
        f_val = compat.aligned_features[col]
        direction = "positive" if s_val > 0 else ("negative" if s_val < 0 else "neutral")
        contributions.append(
            ShapContribution(
                feature=col,
                label=PE_FEATURE_LABELS.get(col, col),
                value=f_val,
                shap_value=round(s_val, 4),
                abs_shap=round(abs(s_val), 4),
                direction=direction,
                plain_explanation=_format_plain_contribution(col, f_val, s_val),
            )
        )

    positive_contributions = sorted(
        [c for c in contributions if c.shap_value > 0],
        key=lambda c: c.shap_value,
        reverse=True,
    )
    negative_contributions = sorted(
        [c for c in contributions if c.shap_value < 0],
        key=lambda c: c.shap_value,
    )
    top_contributions = sorted(contributions, key=lambda c: c.abs_shap, reverse=True)[: max(1, top_n)]

    summary = _build_plain_summary(
        risk_score=risk_score,
        risk_tier=risk_tier,
        model_name=metrics.model_name,
        pos_drivers=positive_contributions,
        neg_drivers=negative_contributions,
    )

    return RiskPredictionResult(
        supported=True,
        warning=None,
        model_key=model_key,
        model_name=metrics.model_name,
        risk_score=risk_score,
        malicious_probability=round(prob, 4),
        risk_tier=risk_tier,
        positive_contributions=positive_contributions,
        negative_contributions=negative_contributions,
        top_contributions=top_contributions,
        plain_language_summary=summary,
        is_synthetic_model=bundle.dataset_metadata.is_synthetic,
    )


def score_pe_extraction(
    bundle: ModelBundle,
    extraction: PEFeatureExtractionResult,
    model_key: str = "random_forest",
    top_n: int = 6,
) -> RiskPredictionResult:
    """Score a PEFeatureExtractionResult only if extraction & schema are fully supported."""
    _, metrics = bundle.get_model(model_key)
    if not extraction.supported:
        warning = extraction.warning or "Unsupported or incomplete PE binary."
        return RiskPredictionResult(
            supported=False,
            warning=warning,
            model_key=model_key,
            model_name=metrics.model_name,
            risk_score=None,
            malicious_probability=None,
            risk_tier="UNSUPPORTED",
            plain_language_summary=warning,
            is_synthetic_model=bundle.dataset_metadata.is_synthetic,
        )

    return explain_and_score_features(
        bundle=bundle,
        features=extraction.features,
        model_key=model_key,
        top_n=top_n,
    )
