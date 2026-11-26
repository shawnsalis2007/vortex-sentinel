"""Vortex Sentinel — Explainable Static Malware Risk Classification Pipeline.

Defensive-only, zero-execution static analysis pipeline for Portable Executable
(PE) binaries and static-feature tabular datasets.

IMPORTANT ETHICS & SAFETY NOTICE:
- All PE inspection operates strictly on raw bytes in memory via ``pefile``;
  binaries are NEVER executed, loaded as code, or detonated.
- The built-in dataset generator produces SYNTHETIC / SIMULATED static-feature
  benchmark data only. It contains no real malware binaries and no personal or
  financial data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import math
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pefile
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

# ---------------------------------------------------------------------------
# Feature Schema & Static Triage Reference Lists
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: tuple[str, ...] = (
    "file_size",
    "file_entropy",
    "max_section_entropy",
    "section_count",
    "suspicious_section_names",
    "import_count",
    "suspicious_api_count",
    "has_authenticode_signature",
    "ascii_string_count",
    "suspicious_string_count",
    "virtual_to_raw_ratio",
)

FEATURE_LABELS: dict[str, str] = {
    "file_size": "File Size (bytes)",
    "file_entropy": "File Shannon Entropy",
    "max_section_entropy": "Max PE Section Entropy",
    "section_count": "PE Section Count",
    "suspicious_section_names": "Suspicious Section Names",
    "import_count": "Total Imported APIs",
    "suspicious_api_count": "Suspicious Windows APIs",
    "has_authenticode_signature": "Authenticode Signature Present",
    "ascii_string_count": "Printable ASCII Strings",
    "suspicious_string_count": "Suspicious String Indicators",
    "virtual_to_raw_ratio": "Virtual-to-Raw Size Ratio",
}

# Standard compiler/linker PE section names typically observed in benign builds.
STANDARD_PE_SECTIONS: frozenset[str] = frozenset(
    {
        ".text",
        ".rdata",
        ".data",
        ".pdata",
        ".rsrc",
        ".reloc",
        ".idata",
        ".edata",
        ".bss",
        ".tls",
        ".debug",
        ".xdata",
        ".didat",
        ".crt",
        "code",
        "data",
        "bss",
        "init",
        "page",
    }
)

# Dual-use Windows APIs commonly audited during SOC static triage (case-insensitive).
SUSPICIOUS_WINDOWS_APIS: frozenset[str] = frozenset(
    {
        "virtualalloc",
        "virtualallocex",
        "virtualprotect",
        "virtualprotectex",
        "writeprocessmemory",
        "readprocessmemory",
        "createremotethread",
        "createremotethreadex",
        "ntunmapviewofsection",
        "ntcreatesection",
        "queueuserapc",
        "setthreadcontext",
        "resumethread",
        "openprocess",
        "loadlibrarya",
        "loadlibraryw",
        "loadlibraryexa",
        "loadlibraryexw",
        "getprocaddress",
        "winexec",
        "shellexecutea",
        "shellexecutew",
        "shellexecuteexa",
        "shellexecuteexw",
        "createprocessa",
        "createprocessw",
        "createprocessinternalw",
        "isdebuggerpresent",
        "checkremotedebuggerpresent",
        "ntqueryinformationprocess",
        "outputdebugstringa",
        "regsetvalueexa",
        "regsetvalueexw",
        "regcreatekeyexa",
        "regcreatekeyexw",
        "createservicea",
        "createservicew",
        "startservicea",
        "startservicew",
        "urldownloadtofilea",
        "urldownloadtofilew",
        "internetopena",
        "internetopenw",
        "internetopenurla",
        "internetopenurlw",
        "httpsendrequesta",
        "httpsendrequestw",
        "wsastartup",
        "connect",
        "cryptencrypt",
        "cryptdecrypt",
        "cryptacquirecontexta",
        "cryptacquirecontextw",
        "setwindowshookexa",
        "setwindowshookexw",
        "getasynckeystate",
        "adjusttokenprivileges",
        "lookupprivilegevaluea",
        "lookupprivilegevaluew",
    }
)

_ASCII_STRING_RE = re.compile(rb"[\x20-\x7e]{4,}")

# Static string patterns frequently investigated in defensive malware triage.
_SUSPICIOUS_STRING_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"https?://", re.IGNORECASE),
    re.compile(r"\b(?:cmd\.exe|powershell(?:\.exe)?|pwsh|wscript|cscript|mshta|rundll32|regsvr32|certutil|bitsadmin)\b", re.IGNORECASE),
    re.compile(r"(?:-encodedcommand|-enc\b|bypass\b|hidden\b)", re.IGNORECASE),
    re.compile(r"(?:software\\microsoft\\windows\\currentversion\\run|\\appdata\\|\\temp\\)", re.IGNORECASE),
    re.compile(r"\b(?:vssadmin|bcdedit|wbadmin|schtasks|net\s+user|sc\s+create)\b", re.IGNORECASE),
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
)

# Risk tier thresholds on a 0-100 calibrated scale.
RISK_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (80.0, "CRITICAL"),
    (55.0, "HIGH / SUSPICIOUS"),
    (25.0, "MODERATE"),
    (0.0, "LOW"),
)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PEInspectionArtifact:
    """Result of zero-execution static PE byte inspection."""

    sha256: str
    features: dict[str, float | int]
    suspicious_sections: list[str] = field(default_factory=list)
    suspicious_apis: list[str] = field(default_factory=list)
    suspicious_string_samples: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvaluationMetrics:
    """Binary classification metrics for SOC model validation."""

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
class FeatureContribution:
    """Single per-file SHAP attribution entry."""

    feature: str
    label: str
    value: float
    shap_value: float
    abs_shap: float
    direction: str
    explanation: str


@dataclass(frozen=True)
class TriageAnalysisResult:
    """Complete per-file risk prediction, SHAP breakdown, and analyst guidance."""

    risk_score: float
    malicious_probability: float
    risk_tier: str
    recommendation: str
    top_contributors: list[FeatureContribution]
    features: dict[str, float]
    model_name: str
    sha256: str | None = None
    observed_indicators: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_score": self.risk_score,
            "malicious_probability": self.malicious_probability,
            "risk_tier": self.risk_tier,
            "recommendation": self.recommendation,
            "model_name": self.model_name,
            "sha256": self.sha256,
            "features": self.features,
            "observed_indicators": self.observed_indicators,
            "top_contributors": [asdict(item) for item in self.top_contributors],
        }


@dataclass
class TrainedSentinelBundle:
    """Trained Random Forest primary model, optional comparison model, and SHAP explainer."""

    rf_model: RandomForestClassifier
    rf_metrics: EvaluationMetrics
    rf_explainer: shap.TreeExplainer
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS
    hgb_model: HistGradientBoostingClassifier | None = None
    hgb_metrics: EvaluationMetrics | None = None
    dataset_notice: str = (
        "SYNTHETIC / SIMULATED STATIC BENCHMARK DATA — NO REAL MALWARE OR PII"
    )

    def save(self, output_path: str | Path) -> Path:
        """Persist trained models and metrics to disk (excluding non-picklable state if any)."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "rf_model": self.rf_model,
            "rf_metrics": self.rf_metrics,
            "hgb_model": self.hgb_model,
            "hgb_metrics": self.hgb_metrics,
            "feature_columns": self.feature_columns,
            "dataset_notice": self.dataset_notice,
        }
        joblib.dump(payload, target)
        return target

    @classmethod
    def load(cls, model_path: str | Path) -> "TrainedSentinelBundle":
        """Load a saved Sentinel model bundle and rebuild its SHAP TreeExplainer."""
        payload = joblib.load(Path(model_path))
        rf_model: RandomForestClassifier = payload["rf_model"]
        return cls(
            rf_model=rf_model,
            rf_metrics=payload["rf_metrics"],
            rf_explainer=shap.TreeExplainer(rf_model),
            feature_columns=tuple(payload.get("feature_columns", FEATURE_COLUMNS)),
            hgb_model=payload.get("hgb_model"),
            hgb_metrics=payload.get("hgb_metrics"),
            dataset_notice=payload.get("dataset_notice", ""),
        )


# ---------------------------------------------------------------------------
# 1. Zero-Execution Static Feature Extraction
# ---------------------------------------------------------------------------


def compute_shannon_entropy(data: bytes) -> float:
    """Compute byte-level Shannon entropy in bits per byte (0.0 to 8.0)."""
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    nonzero = counts[counts > 0]
    probabilities = nonzero / len(data)
    entropy = -float(np.sum(probabilities * np.log2(probabilities)))
    return round(max(0.0, min(8.0, entropy)), 4)


def _decode_section_name(raw_name: bytes) -> str:
    """Decode an 8-byte PE section name safely without raising on non-ASCII bytes."""
    return raw_name.split(b"\x00", 1)[0].decode("ascii", errors="replace").strip()


def _is_suspicious_section_name(name: str) -> bool:
    """Return True if a PE section name is empty, non-ASCII, or non-standard."""
    if not name or "\ufffd" in name:
        return True
    if any(ord(ch) < 32 or ord(ch) > 126 for ch in name):
        return True
    return name.lower() not in STANDARD_PE_SECTIONS


def _check_authenticode_present(pe: pefile.PE, total_bytes: int) -> int:
    """Return 1 if the PE security directory points to a bounded certificate blob."""
    if not hasattr(pe, "OPTIONAL_HEADER") or not hasattr(pe.OPTIONAL_HEADER, "DATA_DIRECTORY"):
        return 0
    sec_idx = pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_SECURITY"]
    data_dirs = pe.OPTIONAL_HEADER.DATA_DIRECTORY
    if len(data_dirs) <= sec_idx:
        return 0
    sec_dir = data_dirs[sec_idx]
    if sec_dir.VirtualAddress > 0 and sec_dir.Size > 0:
        # Certificate table VirtualAddress is a raw file offset per PE/COFF spec.
        if sec_dir.VirtualAddress + sec_dir.Size <= total_bytes:
            return 1
    return 0


def _extract_imports(pe: pefile.PE) -> tuple[int, list[str]]:
    """Count total imported symbols and identify security-relevant Windows APIs."""
    total_imports = 0
    suspicious_apis: list[str] = []
    seen_suspicious: set[str] = set()

    for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []) or []:
        for imp in getattr(entry, "imports", []) or []:
            total_imports += 1
            if not imp.name:
                continue
            api_name = imp.name.decode("ascii", errors="ignore")
            if api_name.lower() in SUSPICIOUS_WINDOWS_APIS and api_name not in seen_suspicious:
                seen_suspicious.add(api_name)
                suspicious_apis.append(api_name)

    return total_imports, suspicious_apis


def _extract_string_stats(file_bytes: bytes, sample_limit: int = 10) -> tuple[int, int, list[str]]:
    """Count printable ASCII strings (>=4 chars) and static triage indicator matches."""
    raw_matches = _ASCII_STRING_RE.findall(file_bytes)
    ascii_count = len(raw_matches)
    suspicious_count = 0
    samples: list[str] = []

    for raw in raw_matches:
        text = raw.decode("ascii", errors="ignore")
        if any(pattern.search(text) for pattern in _SUSPICIOUS_STRING_PATTERNS):
            suspicious_count += 1
            if len(samples) < sample_limit:
                samples.append(text[:120])

    return ascii_count, suspicious_count, samples


def inspect_pe_bytes(file_bytes: bytes) -> PEInspectionArtifact:
    """Safely extract static features and triage evidence from raw PE bytes.

    Files are parsed purely in memory with ``pefile`` and are never executed.
    Raises ``ValueError`` if ``file_bytes`` is empty or not a valid PE structure.
    """
    if not file_bytes:
        raise ValueError("Input byte buffer is empty.")

    sha256 = hashlib.sha256(file_bytes).hexdigest()
    file_size = len(file_bytes)
    file_entropy = compute_shannon_entropy(file_bytes)

    try:
        pe = pefile.PE(data=file_bytes, fast_load=False)
    except pefile.PEFormatError as exc:
        raise ValueError(f"Input bytes are not a valid Portable Executable (PE): {exc}") from exc

    try:
        sections = getattr(pe, "sections", []) or []
        section_count = len(sections)
        section_entropies: list[float] = []
        suspicious_sections: list[str] = []
        total_virtual = 0
        total_raw = 0

        for section in sections:
            sec_name = _decode_section_name(section.Name)
            if _is_suspicious_section_name(sec_name):
                suspicious_sections.append(sec_name or "<unnamed>")

            sec_data = section.get_data()
            section_entropies.append(compute_shannon_entropy(sec_data))
            total_virtual += int(getattr(section, "Misc_VirtualSize", 0) or 0)
            total_raw += int(getattr(section, "SizeOfRawData", 0) or 0)

        max_section_entropy = max(section_entropies) if section_entropies else 0.0

        # Guard against zero raw section sizes (common in custom packers).
        if total_raw > 0:
            virtual_to_raw_ratio = min(round(total_virtual / total_raw, 4), 100.0)
        elif total_virtual > 0:
            virtual_to_raw_ratio = min(round(total_virtual / max(file_size, 1), 4), 100.0)
        else:
            virtual_to_raw_ratio = 0.0

        import_count, suspicious_apis = _extract_imports(pe)
        has_signature = _check_authenticode_present(pe, file_size)
    finally:
        pe.close()

    ascii_string_count, suspicious_string_count, string_samples = _extract_string_stats(file_bytes)

    features: dict[str, float | int] = {
        "file_size": int(file_size),
        "file_entropy": float(file_entropy),
        "max_section_entropy": float(max_section_entropy),
        "section_count": int(section_count),
        "suspicious_section_names": int(len(suspicious_sections)),
        "import_count": int(import_count),
        "suspicious_api_count": int(len(suspicious_apis)),
        "has_authenticode_signature": int(has_signature),
        "ascii_string_count": int(ascii_string_count),
        "suspicious_string_count": int(suspicious_string_count),
        "virtual_to_raw_ratio": float(virtual_to_raw_ratio),
    }

    return PEInspectionArtifact(
        sha256=sha256,
        features=features,
        suspicious_sections=suspicious_sections,
        suspicious_apis=suspicious_apis,
        suspicious_string_samples=string_samples,
    )


def extract_pe_features(file_bytes: bytes) -> dict[str, float | int]:
    """Convenience wrapper returning only the ordered static numeric feature dict."""
    return inspect_pe_bytes(file_bytes).features


# ---------------------------------------------------------------------------
# 2. Synthetic / Simulated Static Feature Benchmark Generator
# ---------------------------------------------------------------------------


def generate_synthetic_pe_dataset(
    n_samples: int = 2000,
    malware_ratio: float = 0.45,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a SYNTHETIC / SIMULATED static PE feature benchmark dataset.

    This helper produces purely simulated tabular records modeled after typical
    benign vs. suspicious PE static traits (including realistic edge cases such
    as unsigned benign utilities and packed payloads). It contains NO real
    malware binaries and NO personal or financial data.
    """
    if n_samples < 50:
        raise ValueError("n_samples must be at least 50 for meaningful train/test evaluation.")
    if not 0.1 <= malware_ratio <= 0.9:
        raise ValueError("malware_ratio must be between 0.1 and 0.9.")

    rng = np.random.default_rng(random_state)
    n_mal = int(round(n_samples * malware_ratio))
    n_ben = n_samples - n_mal

    # Split benign into typical signed software (82%) and noisy unsigned/compressed tools (18%)
    n_ben_std = int(round(n_ben * 0.82))
    n_ben_noisy = n_ben - n_ben_std

    # Split simulated malicious into packed/crypted (52%) and unpacked trojan/dropper (48%)
    n_mal_packed = int(round(n_mal * 0.52))
    n_mal_unpacked = n_mal - n_mal_packed

    def _build_block(
        count: int,
        label: int,
        file_size_range: tuple[float, float],
        file_ent: tuple[float, float],
        max_sec_ent: tuple[float, float],
        sec_cnt: tuple[int, int],
        susp_sec: tuple[float, int],
        imports: tuple[int, int],
        susp_api: tuple[float, int],
        sig_prob: float,
        ascii_str: tuple[int, int],
        susp_str: tuple[float, int],
        v_ratio: tuple[float, float],
    ) -> pd.DataFrame:
        log_low, log_high = math.log10(file_size_range[0]), math.log10(file_size_range[1])
        file_sizes = np.round(10 ** rng.uniform(log_low, log_high, size=count)).astype(int)
        file_entropy = np.clip(rng.normal(file_ent[0], file_ent[1], size=count), 1.5, 7.99)
        max_section_entropy = np.clip(
            np.maximum(file_entropy + rng.uniform(0.05, 0.55, size=count), rng.normal(max_sec_ent[0], max_sec_ent[1], size=count)),
            2.0,
            8.0,
        )
        section_count = rng.integers(sec_cnt[0], sec_cnt[1] + 1, size=count)
        suspicious_section_names = np.clip(rng.poisson(susp_sec[0], size=count), 0, susp_sec[1])
        import_count = rng.integers(imports[0], imports[1] + 1, size=count)
        suspicious_api_count = np.minimum(
            import_count,
            np.clip(rng.poisson(susp_api[0], size=count), 0, susp_api[1]),
        )
        has_signature = rng.binomial(1, sig_prob, size=count)
        ascii_string_count = rng.integers(ascii_str[0], ascii_str[1] + 1, size=count)
        suspicious_string_count = np.minimum(
            ascii_string_count,
            np.clip(rng.poisson(susp_str[0], size=count), 0, susp_str[1]),
        )
        virtual_to_raw_ratio = np.clip(rng.normal(v_ratio[0], v_ratio[1], size=count), 0.5, 25.0)

        return pd.DataFrame(
            {
                "file_size": file_sizes,
                "file_entropy": np.round(file_entropy, 4),
                "max_section_entropy": np.round(max_section_entropy, 4),
                "section_count": section_count,
                "suspicious_section_names": suspicious_section_names,
                "import_count": import_count,
                "suspicious_api_count": suspicious_api_count,
                "has_authenticode_signature": has_signature,
                "ascii_string_count": ascii_string_count,
                "suspicious_string_count": suspicious_string_count,
                "virtual_to_raw_ratio": np.round(virtual_to_raw_ratio, 4),
                "label": np.full(count, label, dtype=int),
            }
        )

    blocks = [
        # Benign standard builds
        _build_block(
            count=n_ben_std,
            label=0,
            file_size_range=(45_000, 12_000_000),
            file_ent=(5.6, 0.55),
            max_sec_ent=(6.2, 0.45),
            sec_cnt=(4, 7),
            susp_sec=(0.15, 2),
            imports=(45, 260),
            susp_api=(1.4, 7),
            sig_prob=0.82,
            ascii_str=(350, 3200),
            susp_str=(0.6, 4),
            v_ratio=(1.05, 0.14),
        ),
        # Benign installers / compressed or unsigned utility tools (realistic edge cases)
        _build_block(
            count=n_ben_noisy,
            label=0,
            file_size_range=(80_000, 18_000_000),
            file_ent=(6.65, 0.45),
            max_sec_ent=(7.05, 0.40),
            sec_cnt=(4, 8),
            susp_sec=(0.6, 2),
            imports=(25, 160),
            susp_api=(2.8, 9),
            sig_prob=0.35,
            ascii_str=(150, 1400),
            susp_str=(1.8, 7),
            v_ratio=(1.45, 0.45),
        ),
        # Simulated packed / obfuscated suspicious binaries
        _build_block(
            count=n_mal_packed,
            label=1,
            file_size_range=(18_000, 1_800_000),
            file_ent=(7.35, 0.35),
            max_sec_ent=(7.68, 0.22),
            sec_cnt=(2, 6),
            susp_sec=(2.1, 5),
            imports=(4, 42),
            susp_api=(5.5, 16),
            sig_prob=0.05,
            ascii_str=(35, 320),
            susp_str=(3.2, 14),
            v_ratio=(3.4, 1.8),
        ),
        # Simulated unpacked dropper / trojan static profiles
        _build_block(
            count=n_mal_unpacked,
            label=1,
            file_size_range=(28_000, 3_500_000),
            file_ent=(6.35, 0.55),
            max_sec_ent=(6.85, 0.48),
            sec_cnt=(5, 10),
            susp_sec=(1.1, 4),
            imports=(35, 175),
            susp_api=(8.5, 24),
            sig_prob=0.10,
            ascii_str=(140, 1200),
            susp_str=(6.2, 22),
            v_ratio=(1.55, 0.65),
        ),
    ]

    dataset = pd.concat(blocks, ignore_index=True)
    dataset = dataset.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    dataset.attrs["synthetic"] = True
    dataset.attrs["notice"] = (
        "SYNTHETIC / SIMULATED TEST DATA ONLY — no real malware binaries or PII."
    )
    return dataset


# ---------------------------------------------------------------------------
# 3. Model Training & Evaluation
# ---------------------------------------------------------------------------


def evaluate_binary_classifier(
    model_name: str,
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> EvaluationMetrics:
    """Compute Accuracy, Precision, Recall, F1, ROC-AUC, and False Positive Rate."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # Guard ROC-AUC if a tiny slice happened to contain a single class.
    unique_classes = np.unique(y_true)
    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(unique_classes) == 2 else 0.0

    return EvaluationMetrics(
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


def train_and_evaluate(
    dataset: pd.DataFrame | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    compare_hgb: bool = True,
) -> TrainedSentinelBundle:
    """Train RandomForestClassifier (and optional HistGradientBoosting comparison).

    Uses a stratified train/test split and initializes a SHAP ``TreeExplainer``
    on the primary Random Forest model for per-file explanations.
    """
    df = dataset if dataset is not None else generate_synthetic_pe_dataset(random_state=random_state)

    missing = [col for col in (*FEATURE_COLUMNS, "label") if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    X = df.loc[:, list(FEATURE_COLUMNS)].astype(float)
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
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
    rf_metrics = evaluate_binary_classifier("RandomForestClassifier", y_test, rf_pred, rf_prob)

    hgb_model: HistGradientBoostingClassifier | None = None
    hgb_metrics: EvaluationMetrics | None = None

    if compare_hgb:
        hgb_model = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=8,
            learning_rate=0.06,
            random_state=random_state,
        )
        hgb_model.fit(X_train, y_train)
        hgb_prob = hgb_model.predict_proba(X_test)[:, 1]
        hgb_pred = (hgb_prob >= 0.5).astype(int)
        hgb_metrics = evaluate_binary_classifier(
            "HistGradientBoostingClassifier",
            y_test,
            hgb_pred,
            hgb_prob,
        )

    rf_explainer = shap.TreeExplainer(rf_model)

    return TrainedSentinelBundle(
        rf_model=rf_model,
        rf_metrics=rf_metrics,
        rf_explainer=rf_explainer,
        hgb_model=hgb_model,
        hgb_metrics=hgb_metrics,
    )


# ---------------------------------------------------------------------------
# 4. Per-File SHAP Explanations & Plain-English Analyst Recommendations
# ---------------------------------------------------------------------------


def _extract_positive_class_shap(
    explainer: shap.TreeExplainer,
    X_row: pd.DataFrame,
) -> np.ndarray:
    """Normalize SHAP output across SHAP versions for binary Random Forest trees.

    SHAP <0.45 returns a list ``[class0_2d, class1_2d]`` whereas SHAP >=0.45 often
    returns a 3D ndarray of shape ``(n_samples, n_features, 2)``.
    """
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
    """Map calibrated 0-100 risk score to standard SOC triage tier."""
    for threshold, tier in RISK_THRESHOLDS:
        if risk_score >= threshold:
            return tier
    return "LOW"


def _describe_feature_impact(feature: str, value: float, shap_val: float) -> str:
    """Create a concise plain-English explanation for one SHAP contribution."""
    label = FEATURE_LABELS.get(feature, feature)
    direction_phrase = "increased malicious risk" if shap_val >= 0 else "reduced malicious risk"
    formatted_val = f"{int(value):,}" if float(value).is_integer() else f"{value:.2f}"

    context_hints = {
        "file_entropy": "high byte randomness may indicate packing or encryption"
        if value >= 7.0
        else "moderate/expected byte entropy",
        "max_section_entropy": "packed or encrypted section characteristics"
        if value >= 7.2
        else "normal section entropy range",
        "suspicious_api_count": "notable presence of memory/process/execution APIs"
        if value >= 4
        else "low count of sensitive Windows APIs",
        "suspicious_section_names": "non-standard or packer-like PE section names"
        if value >= 1
        else "standard PE section naming",
        "has_authenticode_signature": "Authenticode signature directory present"
        if value >= 1
        else "binary lacks an Authenticode signature entry",
        "suspicious_string_count": "shell, network, or persistence string indicators found"
        if value >= 3
        else "few suspicious string indicators",
        "virtual_to_raw_ratio": "large virtual-vs-raw expansion consistent with packing"
        if value >= 2.0
        else "normal virtual-to-raw section alignment",
        "import_count": "sparse import table (often seen in packed stubs)"
        if value <= 25
        else "typical imported function surface",
    }
    hint = context_hints.get(feature, "observed static structural measurement")
    return f"{label} = {formatted_val} ({hint}; {direction_phrase}, SHAP {shap_val:+.4f})."


def _build_analyst_recommendation(
    risk_score: float,
    risk_tier: str,
    top_contributors: list[FeatureContribution],
) -> str:
    """Generate an actionable plain-English SOC triage recommendation."""
    risk_drivers = [c.label for c in top_contributors if c.shap_value > 0][:3]
    drivers_text = ", ".join(risk_drivers) if risk_drivers else "overall static feature profile"

    if risk_tier == "CRITICAL":
        return (
            f"CRITICAL ({risk_score:.1f}/100): Strong static indicators driven by {drivers_text}. "
            "Immediately quarantine/block the artifact on endpoints, log its SHA-256 hash, and "
            "escalate to Tier-2/3 malware analysis in an isolated dynamic sandbox before any handling."
        )
    if risk_tier == "HIGH / SUSPICIOUS":
        return (
            f"HIGH / SUSPICIOUS ({risk_score:.1f}/100): Elevated static risk linked to {drivers_text}. "
            "Hold execution, verify publisher/source provenance, inspect flagged imports and strings, "
            "and submit to a sandbox if business justification is claimed."
        )
    if risk_tier == "MODERATE":
        return (
            f"MODERATE ({risk_score:.1f}/100): Mixed static signals primarily involving {drivers_text}. "
            "Review Authenticode provenance and allowlist/hash reputation before permitting execution; "
            "monitor endpoint telemetry if deployed."
        )
    return (
        f"LOW ({risk_score:.1f}/100): Static structural properties appear consistent with benign software. "
        "No immediate escalation required; archive triage log and rely on standard endpoint monitoring."
    )


def analyze_feature_record(
    bundle: TrainedSentinelBundle,
    features: dict[str, float | int],
    top_n: int = 5,
    sha256: str | None = None,
    observed_indicators: dict[str, list[str]] | None = None,
) -> TriageAnalysisResult:
    """Score one static feature dictionary and explain it with SHAP TreeExplainer."""
    ordered_row = {col: float(features[col]) for col in bundle.feature_columns}
    X_row = pd.DataFrame([ordered_row], columns=list(bundle.feature_columns))

    malicious_prob = float(bundle.rf_model.predict_proba(X_row)[0, 1])
    risk_score = round(malicious_prob * 100.0, 2)
    risk_tier = _classify_risk_tier(risk_score)

    shap_vals = _extract_positive_class_shap(bundle.rf_explainer, X_row)

    contributions: list[FeatureContribution] = []
    for idx, col in enumerate(bundle.feature_columns):
        s_val = float(shap_vals[idx])
        val = ordered_row[col]
        contributions.append(
            FeatureContribution(
                feature=col,
                label=FEATURE_LABELS.get(col, col),
                value=val,
                shap_value=round(s_val, 4),
                abs_shap=round(abs(s_val), 4),
                direction="increases_risk" if s_val >= 0 else "decreases_risk",
                explanation=_describe_feature_impact(col, val, s_val),
            )
        )

    contributions.sort(key=lambda item: item.abs_shap, reverse=True)
    top_contributors = contributions[: max(1, top_n)]
    recommendation = _build_analyst_recommendation(risk_score, risk_tier, top_contributors)

    return TriageAnalysisResult(
        risk_score=risk_score,
        malicious_probability=round(malicious_prob, 4),
        risk_tier=risk_tier,
        recommendation=recommendation,
        top_contributors=top_contributors,
        features=ordered_row,
        model_name=bundle.rf_metrics.model_name,
        sha256=sha256,
        observed_indicators=observed_indicators or {},
    )


def analyze_pe_bytes(
    bundle: TrainedSentinelBundle,
    file_bytes: bytes,
    top_n: int = 5,
) -> TriageAnalysisResult:
    """End-to-end zero-execution PE static triage: extract -> classify -> SHAP explain."""
    artifact = inspect_pe_bytes(file_bytes)
    indicators = {
        "suspicious_sections": artifact.suspicious_sections,
        "suspicious_apis": artifact.suspicious_apis,
        "suspicious_string_samples": artifact.suspicious_string_samples,
    }
    return analyze_feature_record(
        bundle=bundle,
        features=artifact.features,
        top_n=top_n,
        sha256=artifact.sha256,
        observed_indicators=indicators,
    )


# ---------------------------------------------------------------------------
# Self-Test / Local CLI Verification
# ---------------------------------------------------------------------------


def _build_harmless_test_pe_bytes() -> bytes:
    """Build a minimal inert synthetic PE header buffer for parser smoke testing.

    Contains only structural PE headers and plain ASCII text padding—zero machine
    code and never executed.
    """
    e_lfanew = 0x80
    buf = bytearray(512)
    buf[0:2] = b"MZ"
    buf[0x3C:0x40] = e_lfanew.to_bytes(4, "little")

    # PE signature + COFF FileHeader (20 bytes)
    buf[e_lfanew : e_lfanew + 4] = b"PE\x00\x00"
    coff = e_lfanew + 4
    buf[coff : coff + 2] = (0x014C).to_bytes(2, "little")  # IMAGE_FILE_MACHINE_I386
    buf[coff + 2 : coff + 4] = (1).to_bytes(2, "little")  # NumberOfSections = 1
    opt_size = 224  # PE32 Optional Header size
    buf[coff + 16 : coff + 18] = opt_size.to_bytes(2, "little")
    buf[coff + 18 : coff + 20] = (0x0102).to_bytes(2, "little")  # EXECUTABLE_IMAGE | 32BIT

    # Optional Header (PE32)
    opt = coff + 20
    buf[opt : opt + 2] = (0x010B).to_bytes(2, "little")  # Magic = PE32
    buf[opt + 56 : opt + 60] = (0x2000).to_bytes(4, "little")  # SizeOfImage
    buf[opt + 60 : opt + 64] = (0x200).to_bytes(4, "little")  # SizeOfHeaders
    buf[opt + 92 : opt + 96] = (16).to_bytes(4, "little")  # NumberOfRvaAndSizes

    # Single Section Header (.text)
    sec = opt + opt_size
    buf[sec : sec + 8] = b".text\x00\x00\x00"
    buf[sec + 8 : sec + 12] = (0x100).to_bytes(4, "little")  # Misc_VirtualSize
    buf[sec + 12 : sec + 16] = (0x1000).to_bytes(4, "little")  # VirtualAddress
    buf[sec + 16 : sec + 20] = (0x100).to_bytes(4, "little")  # SizeOfRawData
    buf[sec + 20 : sec + 24] = (0x200).to_bytes(4, "little")  # PointerToRawData

    # Harmless inert text payload for section body
    section_body = (b"VortexSentinelStaticTestFixture_BenignString_" * 8)[:256]
    buf.extend(section_body)
    return bytes(buf)


def main() -> None:
    """Run an end-to-end synthetic benchmark + SHAP explainability smoke check."""
    dataset = generate_synthetic_pe_dataset(n_samples=2000, random_state=42)
    print(f"[Notice] {dataset.attrs['notice']}")
    print(f"Generated {len(dataset)} synthetic records ({int(dataset['label'].sum())} simulated suspicious).\n")

    bundle = train_and_evaluate(dataset=dataset, compare_hgb=True)

    print("=== Model Evaluation (Held-out 20% Synthetic Test Set) ===")
    for metrics in (bundle.rf_metrics, bundle.hgb_metrics):
        if metrics is None:
            continue
        print(
            f"{metrics.model_name}: "
            f"Acc={metrics.accuracy:.4f}, Prec={metrics.precision:.4f}, "
            f"Rec={metrics.recall:.4f}, F1={metrics.f1_score:.4f}, "
            f"AUC={metrics.roc_auc:.4f}, FPR={metrics.false_positive_rate:.4f}"
        )

    print("\n=== Synthetic High-Risk Record Triage + SHAP Explanation ===")
    suspicious_sample = dataset[dataset["label"] == 1].iloc[0].drop("label").to_dict()
    result = analyze_feature_record(bundle, suspicious_sample, top_n=5)
    print(f"Recommendation: {result.recommendation}")
    for item in result.top_contributors:
        print(f"  - {item.explanation}")

    print("\n=== Inert PE Byte Buffer Smoke Test (Zero-Execution) ===")
    inert_pe = _build_harmless_test_pe_bytes()
    pe_result = analyze_pe_bytes(bundle, inert_pe, top_n=3)
    print(f"SHA-256: {pe_result.sha256[:16]}... | {pe_result.recommendation}")


if __name__ == "__main__":
    main()
