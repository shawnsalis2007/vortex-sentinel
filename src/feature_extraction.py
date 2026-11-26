"""Vortex Sentinel — Safe Zero-Execution Static PE Feature Extraction.

Extracts static Portable Executable (PE) metadata, hashes, Shannon entropy,
section statistics, API import counts, Authenticode directory presence, and
ASCII string indicators purely from raw bytes in memory using ``pefile``.

Files are NEVER executed, loaded into an OS process, or written out for
execution. Feature compatibility with the trained model schema is strictly
enforced: incomplete or schema-mismatched inputs return explicit
unsupported-analysis warnings instead of guessing missing values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pefile

SCHEMA_VERSION = "1.0.0"
DEFAULT_MAX_PE_BYTES = 50 * 1024 * 1024  # 50 MB defensive memory cap

# Canonical ordered feature schema shared by extraction and ML models.
PE_FEATURE_COLUMNS: tuple[str, ...] = (
    "file_size",
    "file_entropy",
    "section_count",
    "mean_section_entropy",
    "max_section_entropy",
    "suspicious_section_names",
    "import_count",
    "imported_dll_count",
    "suspicious_api_count",
    "has_authenticode_signature",
    "ascii_string_count",
    "suspicious_string_count",
    "virtual_to_raw_ratio",
)

PE_FEATURE_LABELS: dict[str, str] = {
    "file_size": "File Size (bytes)",
    "file_entropy": "File Shannon Entropy",
    "section_count": "PE Section Count",
    "mean_section_entropy": "Mean PE Section Entropy",
    "max_section_entropy": "Max PE Section Entropy",
    "suspicious_section_names": "Suspicious Section Names",
    "import_count": "Total Imported APIs",
    "imported_dll_count": "Imported DLL Count",
    "suspicious_api_count": "Suspicious Windows APIs",
    "has_authenticode_signature": "Authenticode Signature Directory",
    "ascii_string_count": "Printable ASCII Strings",
    "suspicious_string_count": "Suspicious String Indicators",
    "virtual_to_raw_ratio": "Virtual-to-Raw Size Ratio",
}

# Standard compiler/linker PE section names typically seen in benign software.
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

_SUSPICIOUS_STRING_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"https?://", re.IGNORECASE),
    re.compile(
        r"\b(?:cmd\.exe|powershell(?:\.exe)?|pwsh|wscript|cscript|mshta|rundll32|regsvr32|certutil|bitsadmin)\b",
        re.IGNORECASE,
    ),
    re.compile(r"(?:-encodedcommand|-enc\b|bypass\b|hidden\b)", re.IGNORECASE),
    re.compile(
        r"(?:software\\microsoft\\windows\\currentversion\\run|\\appdata\\|\\temp\\)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:vssadmin|bcdedit|wbadmin|schtasks|net\s+user|sc\s+create)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
)


@dataclass(frozen=True)
class FeatureCompatibilityResult:
    """Strict schema compatibility verification result."""

    compatible: bool
    warning: str | None
    missing_features: list[str] = field(default_factory=list)
    invalid_features: list[str] = field(default_factory=list)
    unexpected_features: list[str] = field(default_factory=list)
    aligned_features: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PEFeatureExtractionResult:
    """Structured output of zero-execution PE static feature extraction."""

    supported: bool
    status: str
    warning: str | None
    file_name: str | None
    sha256: str
    md5: str
    file_size: int
    features: dict[str, float] = field(default_factory=dict)
    section_details: list[dict[str, Any]] = field(default_factory=list)
    imported_dlls: list[str] = field(default_factory=list)
    suspicious_apis: list[str] = field(default_factory=list)
    suspicious_strings: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_shannon_entropy(data: bytes) -> float:
    """Compute byte-level Shannon entropy (0.0 to 8.0 bits/byte)."""
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    nonzero = counts[counts > 0]
    probabilities = nonzero / len(data)
    entropy = -float(np.sum(probabilities * np.log2(probabilities)))
    return round(max(0.0, min(8.0, entropy)), 4)


def compute_file_hashes(file_bytes: bytes) -> dict[str, str]:
    """Compute cryptographic SHA-256 and MD5 digests on raw bytes."""
    return {
        "sha256": hashlib.sha256(file_bytes).hexdigest(),
        "md5": hashlib.md5(file_bytes).hexdigest(),
    }


def validate_feature_compatibility(
    features: Mapping[str, Any],
    expected_features: Sequence[str] = PE_FEATURE_COLUMNS,
    strict_extra: bool = False,
) -> FeatureCompatibilityResult:
    """Strictly verify a feature dictionary matches the expected model schema.

    Never guesses, imputes, or coerces missing/non-finite values. Returns an
    explicit compatibility warning if any required feature is absent or invalid.
    """
    expected_tuple = tuple(expected_features)
    missing: list[str] = []
    invalid: list[str] = []
    aligned: dict[str, float] = {}

    for col in expected_tuple:
        if col not in features:
            missing.append(col)
            continue
        raw_val = features[col]
        if raw_val is None or isinstance(raw_val, bool):
            invalid.append(col)
            continue
        try:
            numeric = float(raw_val)
        except (TypeError, ValueError):
            invalid.append(col)
            continue
        if not math.isfinite(numeric):
            invalid.append(col)
            continue
        aligned[col] = numeric

    unexpected = sorted(set(features.keys()) - set(expected_tuple))
    problems: list[str] = []
    if missing:
        problems.append(f"missing required features {missing}")
    if invalid:
        problems.append(f"non-numeric or non-finite values for {invalid}")
    if strict_extra and unexpected:
        problems.append(f"unexpected features {unexpected}")

    if problems:
        warning = (
            "Unsupported or incompatible feature vector for model inference: "
            + "; ".join(problems)
            + ". Analysis halted instead of guessing missing values."
        )
        return FeatureCompatibilityResult(
            compatible=False,
            warning=warning,
            missing_features=missing,
            invalid_features=invalid,
            unexpected_features=unexpected,
            aligned_features=None,
        )

    return FeatureCompatibilityResult(
        compatible=True,
        warning=None,
        missing_features=[],
        invalid_features=[],
        unexpected_features=unexpected,
        aligned_features=aligned,
    )


def validate_tabular_dataset(
    df: pd.DataFrame,
    expected_features: Sequence[str] = PE_FEATURE_COLUMNS,
    require_label: bool = True,
) -> tuple[bool, str | None]:
    """Validate an external DataFrame against the model feature schema without imputing."""
    if df.empty:
        return False, "Unsupported dataset: input table is empty."

    required = list(expected_features) + (["label"] if require_label else [])
    missing = [col for col in required if col not in df.columns]
    if missing:
        return (
            False,
            f"Unsupported or incompatible dataset schema: missing columns {missing}.",
        )

    for col in expected_features:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.isna().any() or not np.isfinite(series.to_numpy(dtype=float)).all():
            return (
                False,
                f"Incompatible dataset column '{col}': contains missing, non-numeric, or non-finite values.",
            )

    if require_label:
        labels = pd.to_numeric(df["label"], errors="coerce")
        if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
            return (
                False,
                "Incompatible dataset column 'label': must contain strictly binary values (0 or 1).",
            )

    return True, None


def _norm_col_key(name: str) -> str:
    """Normalize a column name for case- and separator-insensitive alias matching."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


PUBLIC_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "file_size": ("filesize", "size", "sizeofimage", "bytecount", "pefilesize"),
    "file_entropy": ("fileentropy", "entropy", "meanentropy", "overallentropy", "shannonentropy"),
    "section_count": ("sectioncount", "numberofsections", "sectionslength", "pesectioncount", "numsections"),
    "mean_section_entropy": ("meansectionentropy", "sectionsmeanentropy", "avgsectionentropy", "sectionentropymean"),
    "max_section_entropy": ("maxsectionentropy", "sectionsmaxentropy", "peaksectionentropy", "sectionentropymax"),
    "suspicious_section_names": (
        "suspicioussectionnames",
        "suspicioussections",
        "nonstandardsections",
        "weirdsections",
    ),
    "import_count": ("importcount", "importsnb", "importedfunctions", "totalimportedapis", "numimports"),
    "imported_dll_count": ("importeddllcount", "importsnbdll", "dllcharacteristics", "importeddlls", "numdlls"),
    "suspicious_api_count": ("suspiciousapicount", "suspiciousimportfunctions", "suspiciousapis", "riskyapis"),
    "has_authenticode_signature": (
        "hasauthenticodesignature",
        "hasauthenticode",
        "issigned",
        "signaturepresent",
        "authenticode",
    ),
    "ascii_string_count": ("asciistringcount", "strings", "printablestrings", "numstrings", "stringcount"),
    "suspicious_string_count": ("suspiciousstringcount", "suspiciousstrings", "riskystrings"),
    "virtual_to_raw_ratio": ("virtualtorawratio", "vtorratio", "sizeratio", "sectionratio"),
}

GROUND_TRUTH_LABEL_KEYS: tuple[str, ...] = ("label", "malware", "ismalware", "target", "class", "legitimate")


def normalize_public_static_dataset(
    df: pd.DataFrame,
    expected_features: Sequence[str] = PE_FEATURE_COLUMNS,
) -> tuple[pd.DataFrame, dict[str, str], str | None]:
    """Map common public PE static-feature CSV aliases to canonical schema columns.

    Canonical columns already present are preserved untouched. Returns
    ``(normalized_df, alias_mapping_used, ground_truth_label_col)``.
    """
    if df.empty:
        return df.copy(), {}, None

    normalized = df.copy()
    norm_to_original: dict[str, str] = {}
    for col in normalized.columns:
        key = _norm_col_key(col)
        if key and key not in norm_to_original:
            norm_to_original[key] = str(col)

    alias_mapping: dict[str, str] = {}
    for canonical in expected_features:
        if canonical in normalized.columns:
            continue
        for alias_key in PUBLIC_COLUMN_ALIASES.get(canonical, ()):
            src_col = norm_to_original.get(alias_key)
            if src_col is not None:
                normalized[canonical] = normalized[src_col]
                alias_mapping[src_col] = canonical
                break

    ground_truth_col: str | None = None
    for lbl_key in GROUND_TRUTH_LABEL_KEYS:
        src_col = norm_to_original.get(lbl_key)
        if src_col is not None:
            series = pd.to_numeric(normalized[src_col], errors="coerce")
            if not series.isna().any() and set(series.unique()).issubset({0, 1}):
                ground_truth_col = src_col
                break

    return normalized, alias_mapping, ground_truth_col


def _decode_section_name(raw_name: bytes) -> str:
    """Decode an 8-byte PE section name safely without raising on binary noise."""
    return raw_name.split(b"\x00", 1)[0].decode("ascii", errors="replace").strip()


def _is_suspicious_section_name(name: str) -> bool:
    """Return True if a section name is empty, non-ASCII, or non-standard."""
    if not name or "\ufffd" in name:
        return True
    if any(ord(ch) < 32 or ord(ch) > 126 for ch in name):
        return True
    return name.lower() not in STANDARD_PE_SECTIONS


def _check_authenticode_present(pe: pefile.PE, total_bytes: int) -> int:
    """Return 1 if the PE security directory points to a bounded certificate table."""
    opt_header = getattr(pe, "OPTIONAL_HEADER", None)
    data_dirs = getattr(opt_header, "DATA_DIRECTORY", None) if opt_header else None
    if not data_dirs:
        return 0
    sec_idx = pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_SECURITY"]
    if len(data_dirs) <= sec_idx:
        return 0
    sec_dir = data_dirs[sec_idx]
    if sec_dir.VirtualAddress > 0 and sec_dir.Size > 0:
        if sec_dir.VirtualAddress + sec_dir.Size <= total_bytes:
            return 1
    return 0


def _extract_imports(pe: pefile.PE) -> tuple[int, list[str], list[str]]:
    """Extract total import count, unique DLL names, and flagged Windows APIs."""
    total_imports = 0
    dll_names: list[str] = []
    seen_dlls: set[str] = set()
    suspicious_apis: list[str] = []
    seen_apis: set[str] = set()

    for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []) or []:
        raw_dll = getattr(entry, "dll", None)
        if raw_dll:
            dll_str = raw_dll.decode("ascii", errors="ignore").strip()
            if dll_str and dll_str.lower() not in seen_dlls:
                seen_dlls.add(dll_str.lower())
                dll_names.append(dll_str)

        for imp in getattr(entry, "imports", []) or []:
            total_imports += 1
            if not imp.name:
                continue
            api_name = imp.name.decode("ascii", errors="ignore").strip()
            if api_name.lower() in SUSPICIOUS_WINDOWS_APIS and api_name.lower() not in seen_apis:
                seen_apis.add(api_name.lower())
                suspicious_apis.append(api_name)

    return total_imports, dll_names, suspicious_apis


def _extract_string_stats(file_bytes: bytes, sample_limit: int = 12) -> tuple[int, int, list[str]]:
    """Count printable ASCII strings (>=4 chars) and static triage indicator hits."""
    matches = _ASCII_STRING_RE.findall(file_bytes)
    ascii_count = len(matches)
    suspicious_count = 0
    samples: list[str] = []

    for raw in matches:
        text = raw.decode("ascii", errors="ignore")
        if any(pattern.search(text) for pattern in _SUSPICIOUS_STRING_PATTERNS):
            suspicious_count += 1
            if len(samples) < sample_limit:
                samples.append(text[:120])

    return ascii_count, suspicious_count, samples


def _unsupported_pe_result(
    warning: str,
    file_name: str | None,
    file_bytes: bytes,
) -> PEFeatureExtractionResult:
    """Build a consistent unsupported-analysis response without guessing features."""
    hashes = compute_file_hashes(file_bytes) if file_bytes else {"sha256": "", "md5": ""}
    return PEFeatureExtractionResult(
        supported=False,
        status="unsupported",
        warning=warning,
        file_name=file_name,
        sha256=hashes["sha256"],
        md5=hashes["md5"],
        file_size=len(file_bytes),
    )


def extract_pe_features(
    file_bytes: bytes,
    file_name: str | None = None,
    expected_features: Sequence[str] | None = None,
    max_file_size_bytes: int = DEFAULT_MAX_PE_BYTES,
) -> PEFeatureExtractionResult:
    """Safely extract static PE features from raw bytes without executing anything.

    Strictly checks structural completeness and schema compatibility with
    ``expected_features``. Returns an explicit unsupported-analysis warning
    whenever the file is malformed, non-PE, incomplete, oversized, or schema-
    incompatible.
    """
    if not isinstance(file_bytes, (bytes, bytearray)):
        return _unsupported_pe_result(
            "Unsupported input type: PE feature extraction requires raw bytes.",
            file_name,
            b"",
        )

    raw_bytes = bytes(file_bytes)
    if not raw_bytes:
        return _unsupported_pe_result(
            "Unsupported input: file buffer is empty (0 bytes).",
            file_name,
            raw_bytes,
        )

    if len(raw_bytes) > max_file_size_bytes:
        return _unsupported_pe_result(
            f"Unsupported input: file size ({len(raw_bytes):,} bytes) exceeds defensive static limit ({max_file_size_bytes:,} bytes).",
            file_name,
            raw_bytes,
        )

    if raw_bytes[:2] != b"MZ":
        return _unsupported_pe_result(
            "Unsupported file format: missing DOS 'MZ' signature; only Windows PE binaries (.exe/.dll/.sys/.scr) are supported for PE scoring.",
            file_name,
            raw_bytes,
        )

    hashes = compute_file_hashes(raw_bytes)
    file_size = len(raw_bytes)
    file_entropy = compute_shannon_entropy(raw_bytes)

    try:
        pe = pefile.PE(data=raw_bytes, fast_load=False)
    except pefile.PEFormatError as exc:
        return _unsupported_pe_result(
            f"Unsupported or malformed PE binary: {exc}",
            file_name,
            raw_bytes,
        )
    except Exception as exc:  # Defensive boundary against unexpected parser errors
        return _unsupported_pe_result(
            f"Unsupported or unreadable PE structure ({type(exc).__name__}): {exc}",
            file_name,
            raw_bytes,
        )

    try:
        if not hasattr(pe, "OPTIONAL_HEADER") or not hasattr(pe, "FILE_HEADER"):
            return _unsupported_pe_result(
                "Incomplete PE binary: missing COFF FILE_HEADER or OPTIONAL_HEADER.",
                file_name,
                raw_bytes,
            )

        sections = getattr(pe, "sections", []) or []
        if not sections:
            return _unsupported_pe_result(
                "Incomplete PE binary: no PE sections found; cannot extract section-level static features reliably.",
                file_name,
                raw_bytes,
            )

        section_details: list[dict[str, Any]] = []
        section_entropies: list[float] = []
        suspicious_section_count = 0
        total_virtual = 0
        total_raw = 0

        for section in sections:
            sec_name = _decode_section_name(section.Name)
            is_suspicious = _is_suspicious_section_name(sec_name)
            if is_suspicious:
                suspicious_section_count += 1

            sec_entropy = compute_shannon_entropy(section.get_data())
            section_entropies.append(sec_entropy)

            virt_size = int(getattr(section, "Misc_VirtualSize", 0) or 0)
            raw_size = int(getattr(section, "SizeOfRawData", 0) or 0)
            total_virtual += virt_size
            total_raw += raw_size

            section_details.append(
                {
                    "name": sec_name or "<unnamed>",
                    "virtual_size": virt_size,
                    "raw_size": raw_size,
                    "entropy": sec_entropy,
                    "suspicious_name": is_suspicious,
                }
            )

        mean_section_entropy = round(float(np.mean(section_entropies)), 4)
        max_section_entropy = round(float(np.max(section_entropies)), 4)

        if total_raw > 0:
            virtual_to_raw_ratio = min(round(total_virtual / total_raw, 4), 100.0)
        elif total_virtual > 0:
            virtual_to_raw_ratio = min(round(total_virtual / max(file_size, 1), 4), 100.0)
        else:
            virtual_to_raw_ratio = 0.0

        import_count, imported_dlls, suspicious_apis = _extract_imports(pe)
        has_signature = _check_authenticode_present(pe, file_size)
    finally:
        pe.close()

    ascii_string_count, suspicious_string_count, suspicious_strings = _extract_string_stats(
        raw_bytes
    )

    extracted: dict[str, float] = {
        "file_size": float(file_size),
        "file_entropy": float(file_entropy),
        "section_count": float(len(section_details)),
        "mean_section_entropy": float(mean_section_entropy),
        "max_section_entropy": float(max_section_entropy),
        "suspicious_section_names": float(suspicious_section_count),
        "import_count": float(import_count),
        "imported_dll_count": float(len(imported_dlls)),
        "suspicious_api_count": float(len(suspicious_apis)),
        "has_authenticode_signature": float(has_signature),
        "ascii_string_count": float(ascii_string_count),
        "suspicious_string_count": float(suspicious_string_count),
        "virtual_to_raw_ratio": float(virtual_to_raw_ratio),
    }

    target_schema = tuple(expected_features) if expected_features is not None else PE_FEATURE_COLUMNS
    compat = validate_feature_compatibility(extracted, expected_features=target_schema)
    if not compat.compatible or compat.aligned_features is None:
        return PEFeatureExtractionResult(
            supported=False,
            status="unsupported",
            warning=compat.warning,
            file_name=file_name,
            sha256=hashes["sha256"],
            md5=hashes["md5"],
            file_size=file_size,
        )

    advisory_warning: str | None = None
    if import_count == 0:
        advisory_warning = (
            "Static note: PE header parsed cleanly, but zero imported APIs were found "
            "(common in resource-only DLLs, native stubs, or packed binaries)."
        )

    return PEFeatureExtractionResult(
        supported=True,
        status="ok",
        warning=advisory_warning,
        file_name=file_name,
        sha256=hashes["sha256"],
        md5=hashes["md5"],
        file_size=file_size,
        features=compat.aligned_features,
        section_details=section_details,
        imported_dlls=imported_dlls,
        suspicious_apis=suspicious_apis,
        suspicious_strings=suspicious_strings,
    )


def extract_pe_features_from_file(
    file_path: str | Path,
    expected_features: Sequence[str] | None = None,
    max_file_size_bytes: int = DEFAULT_MAX_PE_BYTES,
) -> PEFeatureExtractionResult:
    """Safely read raw bytes from disk (without executing) and extract PE features."""
    path = Path(file_path)
    if not path.is_file():
        return _unsupported_pe_result(
            f"Unsupported input: path '{path}' does not exist or is not a regular file.",
            path.name,
            b"",
        )

    size_on_disk = path.stat().st_size
    if size_on_disk > max_file_size_bytes:
        return _unsupported_pe_result(
            f"Unsupported input: file '{path.name}' ({size_on_disk:,} bytes) exceeds defensive static limit ({max_file_size_bytes:,} bytes).",
            path.name,
            b"",
        )

    data = path.read_bytes()
    return extract_pe_features(
        file_bytes=data,
        file_name=path.name,
        expected_features=expected_features,
        max_file_size_bytes=max_file_size_bytes,
    )
