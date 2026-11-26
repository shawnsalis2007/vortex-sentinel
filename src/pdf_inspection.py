"""Vortex Sentinel — Safe Zero-Execution PDF Static Inspection.

Performs bounded, read-only static inspection of PDF documents using PyMuPDF
(``pymupdf``) and ``pikepdf`` strictly in memory:
- Enforces file size, page count, attachment count/size, object graph, and
  wall-clock processing-time limits.
- Handles malformed, encrypted, and unsupported PDFs gracefully without crashing.
- Inspects metadata, page text/links/annotations, JavaScript/action dictionary
  indicators, and embedded attachments without executing scripts/attachments or
  auto-opening suspicious PDFs in a browser preview.
- Routes supported embedded PE attachments to ``feature_extraction.py`` and
  marks unsupported attachments for manual review.
- Never labels a PDF malicious from a single indicator alone, and never claims
  a file is safe merely because no static indicators were found.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import io
from pathlib import Path
import re
import time
from typing import Any, Sequence

import pikepdf

try:
    import pymupdf
except ImportError:  # Fallback for older PyMuPDF installations
    import fitz as pymupdf  # type: ignore[no-redef]

try:
    from src.feature_extraction import (
        PE_FEATURE_COLUMNS,
        PEFeatureExtractionResult,
        compute_file_hashes,
        extract_pe_features,
    )
except ImportError:  # Allow running directly from within src/
    from feature_extraction import (  # type: ignore[no-redef]
        PE_FEATURE_COLUMNS,
        PEFeatureExtractionResult,
        compute_file_hashes,
        extract_pe_features,
    )

PE_ATTACHMENT_EXTENSIONS: frozenset[str] = frozenset(
    {".exe", ".dll", ".sys", ".scr", ".ocx", ".cpl", ".drv", ".efi"}
)

# Structural PDF dictionary keys inspected statically (zero script execution).
ACTION_AND_SCRIPT_KEYS: dict[str, str] = {
    "/JS": "javascript",
    "/JavaScript": "javascript",
    "/OpenAction": "auto_action",
    "/AA": "auto_action",
    "/Launch": "launch_action",
    "/SubmitForm": "external_action",
    "/ImportData": "external_action",
    "/GoToR": "external_action",
    "/GoToE": "external_action",
    "/RichMedia": "rich_media",
}

_SUSPICIOUS_URI_SCHEMES = ("javascript:", "vbscript:", "file:", "data:", "ms-msdt:", "search-ms:")
_IP_URL_RE = re.compile(r"^https?://(?:\d{1,3}\.){3}\d{1,3}(?:[:/]|$)", re.IGNORECASE)
_SUSPICIOUS_TEXT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:powershell(?:\.exe)?|cmd\.exe|wscript|mshta|rundll32|certutil)\b", re.IGNORECASE),
    re.compile(r"(?:-encodedcommand|-enc\b)", re.IGNORECASE),
)

PDF_EPISTEMIC_LIMITATIONS: tuple[str, ...] = (
    "Static PDF inspection never executes embedded JavaScript, actions, or attachments.",
    "Browser and embedded PDF auto-preview are disabled for defensive safety.",
    "A single indicator (e.g., /OpenAction or /JS) never proves a PDF is malicious.",
    "Finding zero indicators does NOT prove a PDF is safe; obfuscated streams or novel parser exploits may evade static checks.",
)


@dataclass(frozen=True)
class PDFInspectionLimits:
    """Defensive resource bounds to protect against huge or malformed PDFs."""

    max_file_size_bytes: int = 25 * 1024 * 1024
    max_pages: int = 50
    max_processing_seconds: float = 10.0
    max_attachments: int = 20
    max_attachment_bytes: int = 15 * 1024 * 1024
    max_pdf_objects: int = 15_000


@dataclass(frozen=True)
class PDFAttachmentResult:
    """Static inspection and routing result for one embedded PDF attachment."""

    filename: str
    size_bytes: int
    sha256: str
    md5: str
    magic_preview: str
    routing_status: str  # "pe_extracted" | "manual_review_required"
    review_reason: str
    pe_extraction: PEFeatureExtractionResult | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass(frozen=True)
class PDFIndicator:
    """Individual static indicator observed in PDF structure, links, or text."""

    category: str
    severity_hint: str
    location: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PDFInspectionResult:
    """Complete safe static PDF inspection summary."""

    supported: bool
    status: str  # "ok" | "encrypted" | "malformed" | "oversized" | "unsupported" | "partial_timeout"
    file_name: str | None
    sha256: str
    md5: str
    file_size: int
    pdf_version: str | None = None
    is_encrypted: bool = False
    total_pages: int = 0
    pages_inspected: int = 0
    truncated_by_page_limit: bool = False
    timed_out: bool = False
    preview_allowed: bool = False  # Always False: never auto-open suspicious PDFs
    metadata: dict[str, str] = field(default_factory=dict)
    total_text_chars: int = 0
    pages_with_text: int = 0
    links: list[dict[str, Any]] = field(default_factory=list)
    annotation_summary: dict[str, int] = field(default_factory=dict)
    indicators: list[PDFIndicator] = field(default_factory=list)
    indicator_categories: list[str] = field(default_factory=list)
    attachments: list[PDFAttachmentResult] = field(default_factory=list)
    structural_notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    assessment_tier: str = "INCOMPLETE_INSPECTION"
    assessment_summary: str = ""
    limitations: tuple[str, ...] = PDF_EPISTEMIC_LIMITATIONS
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _deadline_exceeded(start_time: float, limits: PDFInspectionLimits) -> bool:
    return (time.perf_counter() - start_time) >= limits.max_processing_seconds


def _sanitize_text(value: Any, max_len: int = 240) -> str:
    """Convert arbitrary metadata/PDF values to bounded printable single-line text."""
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    return text[:max_len]


def _classify_uri(uri: str) -> str | None:
    """Return an indicator detail if a URI uses a risky scheme or raw IP host."""
    lower = uri.strip().lower()
    for scheme in _SUSPICIOUS_URI_SCHEMES:
        if lower.startswith(scheme):
            return f"Link uses non-standard or script URI scheme ({scheme})"
    if _IP_URL_RE.match(lower):
        return "Link targets a raw IPv4 address rather than a domain name"
    return None


def _inspect_with_pymupdf(
    pdf_bytes: bytes,
    limits: PDFInspectionLimits,
    start_time: float,
) -> dict[str, Any]:
    """Extract metadata, bounded page text/links/annotations, and embedded files via PyMuPDF."""
    out: dict[str, Any] = {
        "total_pages": 0,
        "pages_inspected": 0,
        "truncated_by_page_limit": False,
        "timed_out": False,
        "is_encrypted": False,
        "metadata": {},
        "total_text_chars": 0,
        "pages_with_text": 0,
        "links": [],
        "annotation_summary": {},
        "indicators": [],
        "attachments_raw": [],
        "warnings": [],
    }

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        if getattr(doc, "is_encrypted", False) or getattr(doc, "needs_pass", False):
            out["is_encrypted"] = True
            out["warnings"].append(
                "PyMuPDF detected encryption/password protection; page text and objects cannot be read statically."
            )
            return out

        raw_meta = getattr(doc, "metadata", None) or {}
        out["metadata"] = {
            str(k): _sanitize_text(v)
            for k, v in raw_meta.items()
            if v not in (None, "")
        }

        total_pages = len(doc)
        out["total_pages"] = total_pages
        if total_pages > limits.max_pages:
            out["truncated_by_page_limit"] = True
            out["warnings"].append(
                f"Page inspection capped at {limits.max_pages} of {total_pages} total pages."
            )

        pages_to_scan = min(total_pages, limits.max_pages)
        seen_uris: set[str] = set()
        annot_counts: dict[str, int] = {}

        for page_idx in range(pages_to_scan):
            if _deadline_exceeded(start_time, limits):
                out["timed_out"] = True
                out["warnings"].append(
                    f"Processing time limit ({limits.max_processing_seconds:.1f}s) reached during page {page_idx + 1} inspection."
                )
                break

            page = doc[page_idx]
            out["pages_inspected"] += 1

            text = page.get_text("text") or ""
            stripped = text.strip()
            if stripped:
                out["pages_with_text"] += 1
                out["total_text_chars"] += len(stripped)
                for pattern in _SUSPICIOUS_TEXT_PATTERNS:
                    match = pattern.search(stripped)
                    if match:
                        out["indicators"].append(
                            PDFIndicator(
                                category="suspicious_text_pattern",
                                severity_hint="review",
                                location=f"Page {page_idx + 1} text",
                                detail=f"Matched static command pattern '{_sanitize_text(match.group(0), 60)}'",
                            )
                        )
                        break

            for link in page.get_links() or []:
                uri = link.get("uri")
                kind = link.get("kind")
                if uri:
                    clean_uri = _sanitize_text(uri, 300)
                    if clean_uri not in seen_uris and len(out["links"]) < 200:
                        seen_uris.add(clean_uri)
                        out["links"].append(
                            {"page": page_idx + 1, "uri": clean_uri, "kind": str(kind)}
                        )
                    uri_issue = _classify_uri(clean_uri)
                    if uri_issue:
                        out["indicators"].append(
                            PDFIndicator(
                                category="suspicious_uri",
                                severity_hint="elevated",
                                location=f"Page {page_idx + 1} link",
                                detail=f"{uri_issue}: {clean_uri[:100]}",
                            )
                        )
                elif kind == pymupdf.LINK_LAUNCH:
                    target_file = _sanitize_text(link.get("file", "<unknown>"), 120)
                    out["indicators"].append(
                        PDFIndicator(
                            category="launch_action",
                            severity_hint="elevated",
                            location=f"Page {page_idx + 1} link",
                            detail=f"Page link defines a LINK_LAUNCH action targeting '{target_file}'.",
                        )
                    )

            for annot in page.annots() or []:
                annot_type = (
                    annot.type[1]
                    if isinstance(getattr(annot, "type", None), (tuple, list)) and len(annot.type) > 1
                    else str(getattr(annot, "type", "Unknown"))
                )
                annot_counts[annot_type] = annot_counts.get(annot_type, 0) + 1

        out["annotation_summary"] = annot_counts

        # Extract embedded files listed in PyMuPDF portfolio / EmbeddedFiles catalog
        try:
            emb_names = list(doc.embfile_names() or [])
        except Exception:
            emb_names = []

        for idx, name in enumerate(emb_names):
            if idx >= limits.max_attachments:
                out["warnings"].append(
                    f"Embedded attachment inspection capped at {limits.max_attachments} files."
                )
                break
            if _deadline_exceeded(start_time, limits):
                out["timed_out"] = True
                break
            try:
                info = doc.embfile_info(name) or {}
                declared_size = int(info.get("size", 0) or 0)
                if declared_size > limits.max_attachment_bytes:
                    out["attachments_raw"].append(
                        {
                            "filename": _sanitize_text(info.get("filename") or name, 120),
                            "data": b"",
                            "declared_size": declared_size,
                            "oversized": True,
                        }
                    )
                    continue
                blob = bytes(doc.embfile_get(name))
                out["attachments_raw"].append(
                    {
                        "filename": _sanitize_text(info.get("filename") or name, 120),
                        "data": blob,
                        "declared_size": len(blob),
                        "oversized": len(blob) > limits.max_attachment_bytes,
                    }
                )
            except Exception as exc:
                out["warnings"].append(
                    f"Could not extract embedded file '{_sanitize_text(name, 80)}': {exc}"
                )

    return out


def _walk_pikepdf_object(
    obj: Any,
    location: str,
    visited: set[tuple[int, int]],
    state: dict[str, Any],
    limits: PDFInspectionLimits,
    start_time: float,
) -> None:
    """Safely traverse PDF objects with cycle detection, object cap, and deadline."""
    if state["timed_out"] or state["object_cap_reached"]:
        return
    if _deadline_exceeded(start_time, limits):
        state["timed_out"] = True
        return

    if isinstance(obj, pikepdf.Object):
        objgen = getattr(obj, "objgen", (0, 0))
        if objgen != (0, 0):
            if objgen in visited:
                return
            visited.add(objgen)
            if len(visited) >= limits.max_pdf_objects:
                state["object_cap_reached"] = True
                return

    if isinstance(obj, pikepdf.Dictionary):
        for raw_key in list(obj.keys()):
            key_str = str(raw_key)
            if key_str in ACTION_AND_SCRIPT_KEYS:
                category = ACTION_AND_SCRIPT_KEYS[key_str]
                if len(state["indicators"]) < 100:
                    state["indicators"].append(
                        PDFIndicator(
                            category=category,
                            severity_hint="elevated",
                            location=location,
                            detail=f"PDF dictionary contains {key_str} key.",
                        )
                    )
            if key_str in ("/AcroForm", "/XFA", "/ObjStm"):
                note = f"Document uses {key_str} structural feature (context only; not inherently malicious)."
                if note not in state["structural_notes"]:
                    state["structural_notes"].append(note)

            if key_str == "/S":
                action_type = str(obj.get(raw_key, ""))
                if action_type in ("/JavaScript", "/Launch", "/SubmitForm", "/ImportData", "/GoToR", "/GoToE"):
                    cat = ACTION_AND_SCRIPT_KEYS.get(action_type, "action_trigger")
                    if len(state["indicators"]) < 100:
                        state["indicators"].append(
                            PDFIndicator(
                                category=cat,
                                severity_hint="elevated",
                                location=location,
                                detail=f"Action dictionary specifies subtype /S {action_type}.",
                            )
                        )
                elif action_type == "/URI" and "/URI" in obj:
                    uri_val = _sanitize_text(obj.get("/URI"), 300)
                    uri_issue = _classify_uri(uri_val)
                    if uri_issue and len(state["indicators"]) < 100:
                        state["indicators"].append(
                            PDFIndicator(
                                category="suspicious_uri",
                                severity_hint="elevated",
                                location=location,
                                detail=f"{uri_issue}: {uri_val[:100]}",
                            )
                        )

            try:
                child = obj.get(raw_key)
                _walk_pikepdf_object(child, f"{location}->{key_str}", visited, state, limits, start_time)
            except Exception:
                continue

    elif isinstance(obj, pikepdf.Array):
        for idx, item in enumerate(obj):
            if idx >= 500:
                break
            try:
                _walk_pikepdf_object(item, f"{location}[{idx}]", visited, state, limits, start_time)
            except Exception:
                continue


def _inspect_with_pikepdf(
    pdf_bytes: bytes,
    limits: PDFInspectionLimits,
    start_time: float,
) -> dict[str, Any]:
    """Inspect PDF object dictionaries, action triggers, and FileSpec attachments via pikepdf."""
    state: dict[str, Any] = {
        "pdf_version": None,
        "is_encrypted": False,
        "timed_out": False,
        "object_cap_reached": False,
        "indicators": [],
        "structural_notes": [],
        "attachments_raw": [],
        "warnings": [],
    }

    with pikepdf.open(io.BytesIO(pdf_bytes)) as pdf:
        state["pdf_version"] = str(pdf.pdf_version)
        state["is_encrypted"] = bool(pdf.is_encrypted)

        visited: set[tuple[int, int]] = set()
        _walk_pikepdf_object(pdf.Root, "Catalog", visited, state, limits, start_time)

        if state["object_cap_reached"]:
            state["warnings"].append(
                f"PDF object graph traversal capped at {limits.max_pdf_objects:,} unique indirect objects."
            )
        if state["timed_out"]:
            state["warnings"].append(
                f"Processing time limit ({limits.max_processing_seconds:.1f}s) reached during structural traversal."
            )

        # Inspect annotation FileAttachment FileSpecs in case not registered in /Names
        for page_idx, page in enumerate(pdf.pages):
            if page_idx >= limits.max_pages or _deadline_exceeded(start_time, limits):
                break
            annots = page.get("/Annots", None)
            if not isinstance(annots, pikepdf.Array):
                continue
            for annot in annots:
                try:
                    if str(annot.get("/Subtype", "")) != "/FileAttachment":
                        continue
                    fs = annot.get("/FS")
                    if not isinstance(fs, pikepdf.Dictionary):
                        continue
                    fname = _sanitize_text(fs.get("/UF") or fs.get("/F") or "attachment.bin", 120)
                    ef = fs.get("/EF")
                    if not isinstance(ef, pikepdf.Dictionary):
                        continue
                    stream = ef.get("/UF") or ef.get("/F")
                    if isinstance(stream, pikepdf.Stream):
                        raw_data = bytes(stream.read_bytes())
                        oversized = len(raw_data) > limits.max_attachment_bytes
                        state["attachments_raw"].append(
                            {
                                "filename": fname,
                                "data": b"" if oversized else raw_data,
                                "declared_size": len(raw_data),
                                "oversized": oversized,
                            }
                        )
                except Exception as exc:
                    state["warnings"].append(
                        f"Could not parse FileAttachment annotation on page {page_idx + 1}: {exc}"
                    )

    return state


def _route_attachments(
    raw_attachments: list[dict[str, Any]],
    limits: PDFInspectionLimits,
    expected_features: Sequence[str],
) -> tuple[list[PDFAttachmentResult], list[PDFIndicator]]:
    """Deduplicate embedded files, route PEs to feature_extraction.py, flag unsupported."""
    results: list[PDFAttachmentResult] = []
    indicators: list[PDFIndicator] = []
    seen_keys: set[tuple[str, str]] = set()

    for entry in raw_attachments:
        if len(results) >= limits.max_attachments:
            break

        filename = entry.get("filename") or "unnamed_attachment"
        data: bytes = entry.get("data") or b""
        declared_size = int(entry.get("declared_size", len(data)))
        oversized = bool(entry.get("oversized", False))

        if oversized:
            dedup_key = (filename.lower(), f"oversized:{declared_size}")
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            results.append(
                PDFAttachmentResult(
                    filename=filename,
                    size_bytes=declared_size,
                    sha256="",
                    md5="",
                    magic_preview="",
                    routing_status="manual_review_required",
                    review_reason=(
                        f"Attachment exceeds {limits.max_attachment_bytes:,}-byte safe extraction cap; "
                        "flagged for manual SOC review."
                    ),
                )
            )
            indicators.append(
                PDFIndicator(
                    category="embedded_unsupported_attachment",
                    severity_hint="elevated",
                    location=f"Attachment '{filename}'",
                    detail="Oversized embedded attachment requires manual isolation and review.",
                )
            )
            continue

        hashes = compute_file_hashes(data)
        dedup_key = (filename.lower(), hashes["sha256"])
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        magic_preview = data[:8].hex()
        ext = Path(filename).suffix.lower()
        is_pe_candidate = data[:2] == b"MZ" or ext in PE_ATTACHMENT_EXTENSIONS

        if is_pe_candidate:
            pe_result = extract_pe_features(
                file_bytes=data,
                file_name=filename,
                expected_features=expected_features,
            )
            if pe_result.supported:
                results.append(
                    PDFAttachmentResult(
                        filename=filename,
                        size_bytes=len(data),
                        sha256=hashes["sha256"],
                        md5=hashes["md5"],
                        magic_preview=magic_preview,
                        routing_status="pe_extracted",
                        review_reason=(
                            "Embedded Windows PE executable parsed statically via feature_extraction.py "
                            "without execution."
                        ),
                        pe_extraction=pe_result,
                    )
                )
                indicators.append(
                    PDFIndicator(
                        category="embedded_pe_attachment",
                        severity_hint="elevated",
                        location=f"Attachment '{filename}'",
                        detail="PDF embeds a Windows Portable Executable (PE) artifact.",
                    )
                )
            else:
                reason = pe_result.warning or (
                    "PE-like attachment could not be parsed cleanly; manual review required."
                )
                results.append(
                    PDFAttachmentResult(
                        filename=filename,
                        size_bytes=len(data),
                        sha256=hashes["sha256"],
                        md5=hashes["md5"],
                        magic_preview=magic_preview,
                        routing_status="manual_review_required",
                        review_reason=reason,
                        pe_extraction=pe_result,
                    )
                )
                indicators.append(
                    PDFIndicator(
                        category="embedded_unsupported_attachment",
                        severity_hint="elevated",
                        location=f"Attachment '{filename}'",
                        detail=f"Unsupported/malformed PE-like attachment: {reason}",
                    )
                )
        else:
            results.append(
                PDFAttachmentResult(
                    filename=filename,
                    size_bytes=len(data),
                    sha256=hashes["sha256"],
                    md5=hashes["md5"],
                    magic_preview=magic_preview,
                    routing_status="manual_review_required",
                    review_reason=(
                        "Unsupported embedded attachment format for automated PE static classification; "
                        "marked for manual SOC review without execution."
                    ),
                )
            )
            indicators.append(
                PDFIndicator(
                    category="embedded_unsupported_attachment",
                    severity_hint="review",
                    location=f"Attachment '{filename}'",
                    detail="Non-PE embedded file detected; manual inspection required.",
                )
            )

    return results, indicators


def _assess_pdf_findings(
    status: str,
    indicator_categories: list[str],
    attachment_count: int,
) -> tuple[str, str]:
    """Determine PDF triage assessment without single-indicator overclaiming or false-safety claims."""
    if status in ("encrypted", "malformed", "oversized", "unsupported", "partial_timeout"):
        return (
            "INCOMPLETE_INSPECTION",
            f"Inspection incomplete ({status}): static analysis could not examine the entire PDF. "
            "Treat contents as unverified and route to manual analyst handling if untrusted.",
        )

    cat_count = len(indicator_categories)
    if cat_count == 0 and attachment_count == 0:
        return (
            "NO_INDICATORS_OBSERVED",
            "No static script, risky action, suspicious URI, or embedded-payload indicators were observed. "
            "This does NOT prove the PDF is safe—novel obfuscation, encrypted payloads, or parser exploits "
            "may not trigger static indicators.",
        )

    if cat_count <= 1:
        only_cat = indicator_categories[0] if indicator_categories else "attachment"
        return (
            "SINGLE_INDICATOR_REVIEW",
            f"Single indicator category observed ({only_cat}). Per Vortex Sentinel policy, a PDF is "
            "never labeled malicious from one indicator alone (e.g., legitimate interactive forms use "
            "single actions/scripts). Manual review is recommended before opening in an isolated viewer.",
        )

    joined = ", ".join(indicator_categories)
    return (
        "MULTIPLE_INDICATORS_SUSPICIOUS",
        f"Multiple distinct static indicator categories observed ({joined}). Risk is elevated; "
        "do not preview in a browser or execute attachments. Escalate for isolated sandbox/manual review.",
    )


def inspect_pdf_bytes(
    pdf_bytes: bytes,
    file_name: str | None = None,
    limits: PDFInspectionLimits | None = None,
    expected_pe_features: Sequence[str] = PE_FEATURE_COLUMNS,
) -> PDFInspectionResult:
    """Safely inspect raw PDF bytes in memory under strict size, page, and time limits."""
    start_time = time.perf_counter()
    cfg = limits or PDFInspectionLimits()

    if not isinstance(pdf_bytes, (bytes, bytearray)):
        return PDFInspectionResult(
            supported=False,
            status="unsupported",
            file_name=file_name,
            sha256="",
            md5="",
            file_size=0,
            warnings=["Unsupported input type: PDF inspection requires raw bytes."],
            assessment_tier="INCOMPLETE_INSPECTION",
            assessment_summary="Unsupported input type; inspection could not proceed.",
        )

    raw = bytes(pdf_bytes)
    hashes = compute_file_hashes(raw) if raw else {"sha256": "", "md5": ""}

    if not raw:
        return PDFInspectionResult(
            supported=False,
            status="unsupported",
            file_name=file_name,
            sha256=hashes["sha256"],
            md5=hashes["md5"],
            file_size=0,
            warnings=["Empty PDF buffer (0 bytes)."],
            assessment_tier="INCOMPLETE_INSPECTION",
            assessment_summary="Empty input cannot be inspected.",
        )

    if len(raw) > cfg.max_file_size_bytes:
        return PDFInspectionResult(
            supported=False,
            status="oversized",
            file_name=file_name,
            sha256=hashes["sha256"],
            md5=hashes["md5"],
            file_size=len(raw),
            warnings=[
                f"PDF size ({len(raw):,} bytes) exceeds defensive limit ({cfg.max_file_size_bytes:,} bytes)."
            ],
            assessment_tier="INCOMPLETE_INSPECTION",
            assessment_summary=(
                "PDF exceeds maximum allowed static inspection size; handle manually in an isolated environment."
            ),
            elapsed_seconds=round(time.perf_counter() - start_time, 4),
        )

    if b"%PDF-" not in raw[:1024]:
        return PDFInspectionResult(
            supported=False,
            status="unsupported",
            file_name=file_name,
            sha256=hashes["sha256"],
            md5=hashes["md5"],
            file_size=len(raw),
            warnings=["Missing '%PDF-' header signature within the first 1,024 bytes."],
            assessment_tier="INCOMPLETE_INSPECTION",
            assessment_summary="Input does not appear to be a valid PDF document.",
            elapsed_seconds=round(time.perf_counter() - start_time, 4),
        )

    warnings: list[str] = []
    mupdf_data: dict[str, Any] | None = None
    pike_data: dict[str, Any] | None = None
    is_encrypted = False

    try:
        mupdf_data = _inspect_with_pymupdf(raw, cfg, start_time)
        warnings.extend(mupdf_data["warnings"])
        if mupdf_data["is_encrypted"]:
            is_encrypted = True
    except Exception as exc:
        warnings.append(f"PyMuPDF could not fully parse PDF ({type(exc).__name__}): {exc}")

    if not is_encrypted and not _deadline_exceeded(start_time, cfg):
        try:
            pike_data = _inspect_with_pikepdf(raw, cfg, start_time)
            warnings.extend(pike_data["warnings"])
            if pike_data["is_encrypted"]:
                is_encrypted = True
        except pikepdf.PasswordError:
            is_encrypted = True
            warnings.append("pikepdf detected password protection; document cannot be decrypted statically.")
        except Exception as exc:
            warnings.append(f"pikepdf could not fully parse PDF ({type(exc).__name__}): {exc}")
    elif not is_encrypted and _deadline_exceeded(start_time, cfg):
        warnings.append(
            f"Skipped deep PDF dictionary walk because processing time limit ({cfg.max_processing_seconds:.1f}s) was reached."
        )

    if is_encrypted:
        tier, summary = _assess_pdf_findings("encrypted", [], 0)
        return PDFInspectionResult(
            supported=False,
            status="encrypted",
            file_name=file_name,
            sha256=hashes["sha256"],
            md5=hashes["md5"],
            file_size=len(raw),
            pdf_version=pike_data.get("pdf_version") if pike_data else None,
            is_encrypted=True,
            warnings=warnings,
            assessment_tier=tier,
            assessment_summary=summary,
            elapsed_seconds=round(time.perf_counter() - start_time, 4),
        )

    if mupdf_data is None and pike_data is None:
        tier, summary = _assess_pdf_findings("malformed", [], 0)
        return PDFInspectionResult(
            supported=False,
            status="malformed",
            file_name=file_name,
            sha256=hashes["sha256"],
            md5=hashes["md5"],
            file_size=len(raw),
            warnings=warnings or ["Malformed PDF could not be parsed by PyMuPDF or pikepdf."],
            assessment_tier=tier,
            assessment_summary=summary,
            elapsed_seconds=round(time.perf_counter() - start_time, 4),
        )

    combined_indicators: list[PDFIndicator] = []
    if mupdf_data:
        combined_indicators.extend(mupdf_data["indicators"])
    if pike_data:
        combined_indicators.extend(pike_data["indicators"])

    raw_attachments: list[dict[str, Any]] = []
    if mupdf_data:
        raw_attachments.extend(mupdf_data["attachments_raw"])
    if pike_data:
        raw_attachments.extend(pike_data["attachments_raw"])

    routed_attachments, attachment_indicators = _route_attachments(
        raw_attachments,
        cfg,
        expected_features=expected_pe_features,
    )
    combined_indicators.extend(attachment_indicators)

    # Deduplicate indicators cleanly while preserving order
    unique_indicators: list[PDFIndicator] = []
    seen_ind: set[tuple[str, str, str]] = set()
    for ind in combined_indicators:
        key = (ind.category, ind.location, ind.detail)
        if key not in seen_ind:
            seen_ind.add(key)
            unique_indicators.append(ind)

    categories = list(dict.fromkeys(ind.category for ind in unique_indicators))
    timed_out = bool(
        (mupdf_data and mupdf_data["timed_out"])
        or (pike_data and pike_data["timed_out"])
        or _deadline_exceeded(start_time, cfg)
    )
    status = "partial_timeout" if timed_out else "ok"

    tier, summary = _assess_pdf_findings(status, categories, len(routed_attachments))

    return PDFInspectionResult(
        supported=True,
        status=status,
        file_name=file_name,
        sha256=hashes["sha256"],
        md5=hashes["md5"],
        file_size=len(raw),
        pdf_version=pike_data.get("pdf_version") if pike_data else None,
        is_encrypted=False,
        total_pages=mupdf_data["total_pages"] if mupdf_data else 0,
        pages_inspected=mupdf_data["pages_inspected"] if mupdf_data else 0,
        truncated_by_page_limit=bool(mupdf_data and mupdf_data["truncated_by_page_limit"]),
        timed_out=timed_out,
        preview_allowed=False,
        metadata=mupdf_data["metadata"] if mupdf_data else {},
        total_text_chars=mupdf_data["total_text_chars"] if mupdf_data else 0,
        pages_with_text=mupdf_data["pages_with_text"] if mupdf_data else 0,
        links=mupdf_data["links"] if mupdf_data else [],
        annotation_summary=mupdf_data["annotation_summary"] if mupdf_data else {},
        indicators=unique_indicators,
        indicator_categories=categories,
        attachments=routed_attachments,
        structural_notes=pike_data["structural_notes"] if pike_data else [],
        warnings=warnings,
        assessment_tier=tier,
        assessment_summary=summary,
        elapsed_seconds=round(time.perf_counter() - start_time, 4),
    )


def inspect_pdf_file(
    file_path: str | Path,
    limits: PDFInspectionLimits | None = None,
    expected_pe_features: Sequence[str] = PE_FEATURE_COLUMNS,
) -> PDFInspectionResult:
    """Safely read raw PDF bytes from disk (without executing) and inspect."""
    cfg = limits or PDFInspectionLimits()
    path = Path(file_path)
    if not path.is_file():
        return PDFInspectionResult(
            supported=False,
            status="unsupported",
            file_name=path.name,
            sha256="",
            md5="",
            file_size=0,
            warnings=[f"File '{path}' does not exist or is not a regular file."],
            assessment_tier="INCOMPLETE_INSPECTION",
            assessment_summary="File could not be read.",
        )

    size_on_disk = path.stat().st_size
    if size_on_disk > cfg.max_file_size_bytes:
        return PDFInspectionResult(
            supported=False,
            status="oversized",
            file_name=path.name,
            sha256="",
            md5="",
            file_size=size_on_disk,
            warnings=[
                f"PDF '{path.name}' ({size_on_disk:,} bytes) exceeds defensive limit ({cfg.max_file_size_bytes:,} bytes)."
            ],
            assessment_tier="INCOMPLETE_INSPECTION",
            assessment_summary="PDF exceeds maximum allowed static inspection size.",
        )

    return inspect_pdf_bytes(
        pdf_bytes=path.read_bytes(),
        file_name=path.name,
        limits=cfg,
        expected_pe_features=expected_pe_features,
    )
