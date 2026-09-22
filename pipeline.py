from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, TypedDict

from field_normalizer import (
    CANONICALIZATION_VERSION,
    canonical_key,
    representative_value,
    values_match,
)

STDIO_ENCODING = "utf-8"
JSON_SUFFIX = ".json"
PDF_SUFFIX = ".pdf"
CATEGORIES_SUFFIX = ".categories.json"
MATCHES_SUFFIX = ".matches.json"
VLM_SUFFIX = ".vlm.json"
RESULT_SUFFIX = ".result.json"
OCR_SCRIPT = "ocr_pdf.py"
CLASSIFIER_SCRIPT = "page_classifier.py"
VLM_SCRIPT = "vlm_extract.py"
MATCHER_SCRIPT = "page_matcher.py"
SCRIPT_DIR = Path(__file__).resolve().parent
OTHER_LABEL = "other"
STATUS_OK = "ok"
RESULT_SCHEMA_VERSION = "2.0"
VERIFICATION_STATUSES = frozenset({"ok", "mismatch", "missing"})
LOAN_FIELDS = ("borrower_name", "property_address", "loan_number")
STAGE_OCR = "ocr"
STAGE_CLASSIFICATION = "classification"
STAGE_VLM = "vlm"
STAGE_MATCHING = "matching"
TIMING_PRECISION = 3
CONFIDENCE_PRECISION = 4
VLM_MISMATCH_NOTE_PREFIX = "vlm mismatch: "
REVIEW_FLAG_PAGE_MIN = 2
REVIEW_FLAG_CLUSTER_MARGIN = 1


class PageLabel(TypedDict):
    page_number: int
    label: str
    confidence: float | None
    notes: str


class DocumentEntry(TypedDict):
    label: str
    pages: list[int]


class LoanFieldVariant(TypedDict):
    value: str
    pages: list[int]


class LoanFieldValue(TypedDict):
    value: str | None
    canonical_value: str | None
    source_pages: list[int]
    variants: list[LoanFieldVariant]
    needs_review: bool


class PageTiming(TypedDict):
    page_number: int
    ocr_seconds: float
    classification_seconds: float
    vlm_seconds: float
    total_seconds: float


class PipelineTimings(TypedDict):
    total_seconds: float
    stages_seconds: dict[str, float]
    pages: list[PageTiming]


class ResultReport(TypedDict):
    schema_version: str
    canonicalization_version: str
    page_labels: list[PageLabel]
    documents: list[DocumentEntry]
    loan_level_fields: dict[str, LoanFieldValue]
    timings: PipelineTimings


class FieldEntry(NamedTuple):
    page_number: int
    value: str
    similarity: float | None
    needs_review: bool


StageRunner = Callable[[Sequence[str]], None]


def log_message(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding=STDIO_ENCODING, errors="replace")


def default_ocr_path(pdf_path: Path) -> Path:
    return pdf_path.parent / f"{pdf_path.stem}{JSON_SUFFIX}"


def default_categories_path(ocr_path: Path) -> Path:
    return ocr_path.parent / f"{ocr_path.stem}{CATEGORIES_SUFFIX}"


def default_vlm_path(pdf_path: Path) -> Path:
    return pdf_path.parent / f"{pdf_path.stem}{VLM_SUFFIX}"


def default_matches_path(ocr_path: Path) -> Path:
    return ocr_path.parent / f"{ocr_path.stem}{MATCHES_SUFFIX}"


def default_result_path(pdf_path: Path) -> Path:
    return pdf_path.parent / f"{pdf_path.stem}{RESULT_SUFFIX}"


def stage_command(script: str, *arguments: object) -> list[str]:
    return [sys.executable, str(SCRIPT_DIR / script), *(str(value) for value in arguments)]


def run_stage(command: Sequence[str]) -> None:
    completed = subprocess.run(list(command), check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Pipeline stage failed with exit code {completed.returncode}: "
            f"{' '.join(command)}"
        )


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding=STDIO_ENCODING))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON ({path}): {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Failed to read {label} ({path}): {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be an object: {path}")
    return payload


def require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    return float(value)


def require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def require_page_records(payload: dict[str, Any], label: str) -> list[dict[str, Any]]:
    pages = payload.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError(f"{label} has no 'pages' entries")
    records: list[dict[str, Any]] = []
    previous_number = 0
    for index, page in enumerate(pages):
        if not isinstance(page, dict):
            raise ValueError(f"{label} page {index} is not an object")
        page_number = page.get("page_number")
        if isinstance(page_number, bool) or not isinstance(page_number, int):
            raise ValueError(f"{label} page {index} has a non-integer page_number")
        if page_number < 1 or page_number <= previous_number:
            raise ValueError(f"{label} page {index} has a non-increasing page_number")
        previous_number = page_number
        records.append(page)
    return records


def index_pages(records: Sequence[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(record["page_number"]): record for record in records}


def require_page_coverage(
    expected: Sequence[int], index: Mapping[int, Any], label: str
) -> None:
    expected_set = set(expected)
    provided = set(index)
    missing = sorted(expected_set - provided)
    extra = sorted(provided - expected_set)
    if missing or extra:
        raise ValueError(
            f"{label} pages do not match the OCR pages "
            f"(missing={missing[:5]} extra={extra[:5]})"
        )


def require_category_labels(payload: dict[str, Any]) -> dict[str, str]:
    categories = payload.get("categories")
    if not isinstance(categories, dict) or not categories:
        raise ValueError("Categories JSON has no 'categories' mapping")
    labels: dict[str, str] = {}
    for key, value in categories.items():
        if not isinstance(key, str) or not key or not isinstance(value, str) or not value:
            raise ValueError("Categories JSON has an invalid 'categories' entry")
        labels[key] = value
    if OTHER_LABEL not in labels:
        raise ValueError(f"Categories JSON is missing the {OTHER_LABEL!r} label")
    return labels


def build_page_labels(
    ocr_numbers: Sequence[int],
    category_index: Mapping[int, dict[str, Any]],
    vlm_index: Mapping[int, dict[str, Any]],
    allowed_labels: Mapping[str, str],
) -> list[PageLabel]:
    labels: list[PageLabel] = []
    for page_number in ocr_numbers:
        record = category_index[page_number]
        category = record.get("category")
        confidence: float | None
        if category is None:
            label = OTHER_LABEL
            confidence = None
        elif isinstance(category, str) and category in allowed_labels:
            label = category
            raw_confidence = record.get("confidence")
            if raw_confidence is None:
                confidence = None
            else:
                confidence = require_number(
                    raw_confidence, f"Categories JSON page {page_number} confidence"
                )
                if not 0.0 <= confidence <= 1.0:
                    raise ValueError(
                        f"Categories JSON page {page_number} has a confidence "
                        f"outside [0, 1]: {confidence}"
                    )
                confidence = round(confidence, CONFIDENCE_PRECISION)
        else:
            raise ValueError(
                f"Categories JSON page {page_number} has an unknown category: {category!r}"
            )
        notes: list[str] = []
        review_reason = record.get("review_reason")
        if isinstance(review_reason, str) and review_reason:
            notes.append(review_reason)
        mismatches = vlm_index[page_number].get("mismatches", [])
        if not isinstance(mismatches, list) or not all(
            isinstance(item, str) for item in mismatches
        ):
            raise ValueError(f"VLM JSON page {page_number} has invalid 'mismatches'")
        if mismatches:
            notes.append(f"{VLM_MISMATCH_NOTE_PREFIX}{', '.join(mismatches)}")
        labels.append(
            {
                "page_number": page_number,
                "label": label,
                "confidence": confidence,
                "notes": "; ".join(notes),
            }
        )
    return labels


def require_member_page_number(entry: Any, label: str, index: int) -> int:
    if not isinstance(entry, dict):
        raise ValueError(f"{label} entry {index} is not an object")
    page_number = entry.get("page_number")
    if isinstance(page_number, bool) or not isinstance(page_number, int) or page_number < 1:
        raise ValueError(f"{label} entry {index} has an invalid page_number")
    return page_number


def collect_document_pages(matches_payload: dict[str, Any]) -> list[list[int]]:
    for name in ("groups", "unique_pages", "skipped_pages"):
        if not isinstance(matches_payload.get(name), list):
            raise ValueError(f"Matching JSON has no '{name}' list")
    documents: list[list[int]] = []
    for group_index, group in enumerate(matches_payload["groups"]):
        if not isinstance(group, dict):
            raise ValueError(f"Matching JSON group {group_index} is not an object")
        members = group.get("members")
        if not isinstance(members, list) or not members:
            raise ValueError(f"Matching JSON group {group_index} has no 'members'")
        documents.append(
            sorted(
                require_member_page_number(
                    member, f"Matching JSON group {group_index}", member_index
                )
                for member_index, member in enumerate(members)
            )
        )
    for entry_index, entry in enumerate(matches_payload["unique_pages"]):
        documents.append(
            [require_member_page_number(entry, "Matching JSON unique_pages", entry_index)]
        )
    for entry_index, entry in enumerate(matches_payload["skipped_pages"]):
        documents.append(
            [require_member_page_number(entry, "Matching JSON skipped_pages", entry_index)]
        )
    return documents


def majority_label(labels: Sequence[str]) -> str:
    counts = Counter(labels)
    best_label = labels[0]
    best_count = counts[best_label]
    for label in labels:
        if counts[label] > best_count:
            best_label = label
            best_count = counts[label]
    return best_label


def build_documents(
    matches_payload: dict[str, Any],
    page_labels: Sequence[PageLabel],
    ocr_numbers: Sequence[int],
) -> list[DocumentEntry]:
    label_by_number = {entry["page_number"]: entry["label"] for entry in page_labels}
    documents: list[DocumentEntry] = []
    for pages in collect_document_pages(matches_payload):
        unknown = [page_number for page_number in pages if page_number not in label_by_number]
        if unknown:
            raise ValueError(f"Matching JSON references unknown pages: {unknown[:5]}")
        documents.append(
            {
                "label": majority_label([label_by_number[page_number] for page_number in pages]),
                "pages": pages,
            }
        )
    documents.sort(key=lambda entry: entry["pages"][0])
    seen = [page_number for entry in documents for page_number in entry["pages"]]
    if len(seen) != len(set(seen)):
        raise ValueError("Matching JSON assigns a page to more than one document")
    missing = sorted(set(ocr_numbers) - set(seen))
    extra = sorted(set(seen) - set(ocr_numbers))
    if missing or extra:
        raise ValueError(
            f"Matching JSON does not cover all OCR pages "
            f"(missing={missing[:5]} extra={extra[:5]})"
        )
    return documents


def optional_similarity(value: Any, page_number: int, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"VLM JSON page {page_number} verification {field} has a non-numeric "
            f"similarity: {value!r}"
        )
    similarity = float(value)
    if not 0.0 <= similarity <= 1.0:
        raise ValueError(
            f"VLM JSON page {page_number} verification {field} has a similarity "
            f"outside [0, 1]: {similarity}"
        )
    return similarity


def require_review_flag(value: Any, page_number: int, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(
            f"VLM JSON page {page_number} verification {field} has a non-boolean "
            "'needs_review'; re-run vlm_extract.py to regenerate the report"
        )
    return value


def collect_field_entries(
    vlm_pages: Sequence[dict[str, Any]],
) -> dict[str, list[FieldEntry]]:
    entries: dict[str, list[FieldEntry]] = {field: [] for field in LOAN_FIELDS}
    for page in vlm_pages:
        page_number = int(page["page_number"])
        extracted = require_object(
            page.get("extracted"), f"VLM JSON page {page_number} extracted"
        )
        verification = require_object(
            page.get("verification"), f"VLM JSON page {page_number} verification"
        )
        for field in LOAN_FIELDS:
            check = require_object(
                verification.get(field),
                f"VLM JSON page {page_number} verification {field}",
            )
            status = check.get("status")
            if status not in VERIFICATION_STATUSES:
                raise ValueError(
                    f"VLM JSON page {page_number} has an unknown verification "
                    f"status for {field}: {status!r}"
                )
            value = extracted.get(field)
            if value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"VLM JSON page {page_number} has an invalid {field}: {value!r}"
                )
            if status != STATUS_OK:
                continue
            entries[field].append(
                FieldEntry(
                    page_number=page_number,
                    value=value,
                    similarity=optional_similarity(
                        check.get("similarity"), page_number, field
                    ),
                    needs_review=require_review_flag(
                        check.get("needs_review"), page_number, field
                    ),
                )
            )
    return entries


def cluster_entries(
    field: str, entries: Sequence[FieldEntry]
) -> list[list[FieldEntry]]:
    clusters: list[list[FieldEntry]] = []
    for entry in entries:
        for cluster in clusters:
            if any(
                values_match(field, entry.value, member.value) for member in cluster
            ):
                cluster.append(entry)
                break
        else:
            clusters.append([entry])
    return clusters


def cluster_pages(cluster: Sequence[FieldEntry]) -> list[int]:
    return sorted({entry.page_number for entry in cluster})


def cluster_similarity(cluster: Sequence[FieldEntry]) -> float:
    similarities = [
        entry.similarity for entry in cluster if entry.similarity is not None
    ]
    if not similarities:
        return 0.0
    return sum(similarities) / len(similarities)


def build_variants(cluster: Sequence[FieldEntry]) -> list[LoanFieldVariant]:
    pages_by_value: dict[str, list[int]] = {}
    for entry in cluster:
        pages_by_value.setdefault(entry.value, []).append(entry.page_number)
    return [
        {"value": value, "pages": sorted(pages)}
        for value, pages in sorted(
            pages_by_value.items(), key=lambda item: (min(item[1]), item[0])
        )
    ]


def rank_clusters(
    clusters: Sequence[Sequence[FieldEntry]],
) -> list[Sequence[FieldEntry]]:
    return sorted(
        clusters,
        key=lambda cluster: (
            -len(cluster_pages(cluster)),
            -cluster_similarity(cluster),
            min(entry.page_number for entry in cluster),
        ),
    )


def build_loan_level_fields(
    vlm_pages: Sequence[dict[str, Any]],
) -> dict[str, LoanFieldValue]:
    entries = collect_field_entries(vlm_pages)
    fields: dict[str, LoanFieldValue] = {}
    for field in LOAN_FIELDS:
        clusters = cluster_entries(field, entries[field])
        if not clusters:
            fields[field] = {
                "value": None,
                "canonical_value": None,
                "source_pages": [],
                "variants": [],
                "needs_review": False,
            }
            continue
        ordered = rank_clusters(clusters)
        winner = ordered[0]
        winner_pages = cluster_pages(winner)
        runner_up_pages = len(cluster_pages(ordered[1])) if len(ordered) > 1 else 0
        variants = build_variants(winner)
        representative = representative_value(
            field, [(variant["value"], variant["pages"]) for variant in variants]
        )
        needs_review = (
            any(entry.needs_review for entry in winner)
            or len(winner_pages) < REVIEW_FLAG_PAGE_MIN
            or (
                runner_up_pages > 0
                and len(winner_pages) - runner_up_pages <= REVIEW_FLAG_CLUSTER_MARGIN
            )
        )
        fields[field] = {
            "value": representative,
            "canonical_value": canonical_key(field, representative),
            "source_pages": winner_pages,
            "variants": variants,
            "needs_review": needs_review,
        }
    return fields


def require_stage_total(payload: dict[str, Any], label: str) -> float:
    timings = require_object(payload.get("timings"), f"{label} timings")
    return require_number(timings.get("total_seconds"), f"{label} timings total_seconds")


def build_timings(
    ocr_payload: dict[str, Any],
    categories_payload: dict[str, Any],
    vlm_payload: dict[str, Any],
    matches_payload: dict[str, Any],
    ocr_pages: Sequence[dict[str, Any]],
    category_index: Mapping[int, dict[str, Any]],
    vlm_index: Mapping[int, dict[str, Any]],
    total_seconds: float,
) -> PipelineTimings:
    stages_seconds = {
        STAGE_OCR: require_number(
            ocr_payload.get("total_duration_seconds"),
            "OCR JSON total_duration_seconds",
        ),
        STAGE_CLASSIFICATION: require_stage_total(categories_payload, "Categories JSON"),
        STAGE_VLM: require_stage_total(vlm_payload, "VLM JSON"),
        STAGE_MATCHING: require_stage_total(matches_payload, "Matching JSON"),
    }
    pages: list[PageTiming] = []
    for page in ocr_pages:
        page_number = int(page["page_number"])
        category_timings = require_object(
            category_index[page_number].get("timings_seconds"),
            f"Categories JSON page {page_number} timings_seconds",
        )
        vlm_timings = require_object(
            vlm_index[page_number].get("timings_seconds"),
            f"VLM JSON page {page_number} timings_seconds",
        )
        ocr_seconds = require_number(
            page.get("duration_seconds"),
            f"OCR JSON page {page_number} duration_seconds",
        )
        classification_seconds = require_number(
            category_timings.get("total_seconds"),
            f"Categories JSON page {page_number} total_seconds",
        )
        vlm_seconds = require_number(
            vlm_timings.get("total_seconds"),
            f"VLM JSON page {page_number} total_seconds",
        )
        pages.append(
            {
                "page_number": page_number,
                "ocr_seconds": round(ocr_seconds, TIMING_PRECISION),
                "classification_seconds": round(
                    classification_seconds, TIMING_PRECISION
                ),
                "vlm_seconds": round(vlm_seconds, TIMING_PRECISION),
                "total_seconds": round(
                    ocr_seconds + classification_seconds + vlm_seconds,
                    TIMING_PRECISION,
                ),
            }
        )
    return {
        "total_seconds": round(total_seconds, TIMING_PRECISION),
        "stages_seconds": {
            name: round(value, TIMING_PRECISION)
            for name, value in stages_seconds.items()
        },
        "pages": pages,
    }


def build_result(
    ocr_payload: dict[str, Any],
    categories_payload: dict[str, Any],
    vlm_payload: dict[str, Any],
    matches_payload: dict[str, Any],
    total_seconds: float,
) -> ResultReport:
    ocr_pages = require_page_records(ocr_payload, "OCR JSON")
    ocr_numbers = [int(page["page_number"]) for page in ocr_pages]
    category_index = index_pages(require_page_records(categories_payload, "Categories JSON"))
    vlm_index = index_pages(require_page_records(vlm_payload, "VLM JSON"))
    require_page_coverage(ocr_numbers, category_index, "Categories JSON")
    require_page_coverage(ocr_numbers, vlm_index, "VLM JSON")
    allowed_labels = require_category_labels(categories_payload)
    page_labels = build_page_labels(
        ocr_numbers, category_index, vlm_index, allowed_labels
    )
    documents = build_documents(matches_payload, page_labels, ocr_numbers)
    loan_level_fields = build_loan_level_fields(
        [vlm_index[page_number] for page_number in ocr_numbers]
    )
    timings = build_timings(
        ocr_payload,
        categories_payload,
        vlm_payload,
        matches_payload,
        ocr_pages,
        category_index,
        vlm_index,
        total_seconds,
    )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "page_labels": page_labels,
        "documents": documents,
        "loan_level_fields": loan_level_fields,
        "timings": timings,
    }


def run_pipeline(pdf_path: Path, runner: StageRunner = run_stage) -> ResultReport:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if pdf_path.suffix.lower() != PDF_SUFFIX:
        raise ValueError(f"File does not have a {PDF_SUFFIX} suffix: {pdf_path}")
    ocr_path = default_ocr_path(pdf_path)
    stages = (
        (STAGE_OCR, stage_command(OCR_SCRIPT, pdf_path)),
        (STAGE_CLASSIFICATION, stage_command(CLASSIFIER_SCRIPT, ocr_path)),
        (STAGE_VLM, stage_command(VLM_SCRIPT, pdf_path, "--ocr", ocr_path)),
        (STAGE_MATCHING, stage_command(MATCHER_SCRIPT, ocr_path)),
    )
    log_message(
        f"[pipeline] pdf={pdf_path} stages={','.join(name for name, _ in stages)}"
    )
    started = time.perf_counter()
    for name, command in stages:
        log_message(f"[pipeline] stage {name} start")
        runner(command)
        log_message(f"[pipeline] stage {name} done")
    total_seconds = time.perf_counter() - started
    report = build_result(
        load_json_object(ocr_path, "OCR JSON"),
        load_json_object(default_categories_path(ocr_path), "Categories JSON"),
        load_json_object(default_vlm_path(pdf_path), "VLM JSON"),
        load_json_object(default_matches_path(ocr_path), "Matching JSON"),
        total_seconds,
    )
    log_message(
        f"[pipeline] pages={len(report['page_labels'])} "
        f"documents={len(report['documents'])} total={total_seconds:.2f}s"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run OCR, page classification, VLM extraction, and page matching for a "
            "mortgage PDF and combine their reports into one result JSON with page "
            "labels, document groups, loan-level fields, and timings."
        )
    )
    parser.add_argument("pdf", type=Path, help="Mortgage PDF file to process")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"JSON output file (default: PDF path with {RESULT_SUFFIX} suffix)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print JSON to stdout instead of writing a file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_stdio()
    args = build_parser().parse_args(argv)
    try:
        report = run_pipeline(args.pdf)
    except Exception as exc:
        log_message(f"ERROR: {type(exc).__name__}: {exc}")
        return 1
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.stdout:
        print(payload)
        return 0
    out_path = args.out if args.out is not None else default_result_path(args.pdf)
    try:
        out_path.write_text(payload, encoding=STDIO_ENCODING)
    except OSError as exc:
        log_message(f"ERROR: failed to write JSON file ({out_path}): {exc}")
        return 1
    log_message(f"[pipeline] JSON written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
