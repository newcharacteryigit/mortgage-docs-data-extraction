from __future__ import annotations

import argparse
import base64
import difflib
import json
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, TypedDict

import pymupdf

DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_MODEL = "qwen/qwen3.5-9b"
DEFAULT_DPI = 200
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_TOKENS = 512
DEFAULT_SEED = 0
DEFAULT_MISMATCH_THRESHOLD = 0.75
SCHEMA_VERSION = "1.0"
PROMPT_VERSION = "1.1"
TEMPERATURE = 0.0
TOP_P = 1.0
REASONING_EFFORT_DISABLED = "none"
CHAT_COMPLETIONS_PATH = "/chat/completions"
PNG_DATA_URL_PREFIX = "data:image/png;base64,"
JSON_SUFFIX = ".json"
PDF_SUFFIX = ".pdf"
DEFAULT_OUTPUT_SUFFIX = ".vlm.json"
POINTS_PER_INCH = 72.0
STDIO_ENCODING = "utf-8"
TEXT_NORMALIZATION = "nfkd_casefold_alnum"
SIMILARITY_PRECISION = 4
EXTRACTION_SCHEMA_NAME = "mortgage_page_fields"
RAW_TEXT_PREVIEW_CHARS = 200
STATUS_OK = "ok"
STATUS_MISMATCH = "mismatch"
STATUS_MISSING = "missing"

EXTRACTION_FIELDS = (
    "borrower_name",
    "property_address",
    "loan_number",
    "page_number",
)

SYSTEM_PROMPT = """You extract structured fields from a single page of a mortgage document.

Return one JSON object with exactly these keys:
- borrower_name
- property_address
- loan_number
- page_number

Rules:
1. Copy values exactly as printed on the page. Do not summarize or translate.
2. page_number is only the printed page number, for example "2" from "Page 2 of 2".
   Do not include words like "Page" or "of" in the value.
3. Use null when the field does not appear on the page.
4. Never guess, infer, or fill from memory.
5. Output JSON only. No markdown, no explanations."""

USER_PROMPT = "Extract the fields from this page image."

EXTRACTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {field: {"type": ["string", "null"]} for field in EXTRACTION_FIELDS},
    "required": list(EXTRACTION_FIELDS),
    "additionalProperties": False,
}

RequestJson = Callable[[str, dict[str, Any], float], dict[str, Any]]


class ExtractedFields(TypedDict):
    borrower_name: str | None
    property_address: str | None
    loan_number: str | None
    page_number: str | None


class FieldVerification(TypedDict):
    status: str
    found_in_ocr: bool
    similarity: float | None


class OcrReference(TypedDict):
    line_count: int | None
    mean_score: float | None
    duration_seconds: float | None


class PageTimings(TypedDict):
    render_seconds: float
    vlm_seconds: float
    compare_seconds: float
    total_seconds: float


class PageExtraction(TypedDict):
    page_number: int
    page_ref: str
    extracted: ExtractedFields
    verification: dict[str, FieldVerification]
    mismatches: list[str]
    ocr_reference: OcrReference
    timings_seconds: PageTimings


class FieldStatusCounts(TypedDict):
    ok: int
    mismatch: int
    missing: int


class ExtractionSummary(TypedDict):
    page_count: int
    field_status_counts: dict[str, FieldStatusCounts]
    field_mismatch_count: int
    field_missing_count: int
    pages_with_mismatch: list[int]
    needs_review_page_count: int


class ExtractionTimings(TypedDict):
    total_seconds: float
    render_seconds: float
    vlm_seconds: float
    compare_seconds: float
    vlm_requests: int
    avg_vlm_seconds_per_page: float | None


class ModelInfo(TypedDict):
    id: str
    base_url: str
    thinking: bool
    prompt_version: str


class ExtractionParameters(TypedDict):
    dpi: int
    temperature: float
    top_p: float
    seed: int
    max_tokens: int
    reasoning_effort: str | None
    mismatch_similarity_threshold: float
    text_normalization: str


class ExtractionReport(TypedDict):
    schema_version: str
    input_files: dict[str, str]
    model: ModelInfo
    parameters: ExtractionParameters
    summary: ExtractionSummary
    pages: list[PageExtraction]
    timings: ExtractionTimings


@dataclass(frozen=True)
class OcrPage:
    source: str
    page_number: int
    text: str
    line_count: int | None
    mean_score: float | None
    duration_seconds: float | None


class OcrDocument(NamedTuple):
    source: str
    pages: list[OcrPage]


def log_message(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding=STDIO_ENCODING, errors="replace")


def default_ocr_path(pdf_path: Path) -> Path:
    return pdf_path.parent / f"{pdf_path.stem}{JSON_SUFFIX}"


def default_output_path(pdf_path: Path) -> Path:
    return pdf_path.parent / f"{pdf_path.stem}{DEFAULT_OUTPUT_SUFFIX}"


def make_page_ref(source: str, page_number: int) -> str:
    return f"{source}#page-{page_number}"


def open_pdf(pdf_path: Path) -> pymupdf.Document:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if pdf_path.suffix.lower() != PDF_SUFFIX:
        raise ValueError(f"File does not have a {PDF_SUFFIX} suffix: {pdf_path}")
    try:
        document = pymupdf.open(pdf_path)
    except Exception as exc:
        raise ValueError(f"Failed to open PDF ({pdf_path}): {exc}") from exc
    if document.page_count == 0:
        document.close()
        raise ValueError(f"PDF contains no pages: {pdf_path}")
    return document


def render_page_png(page: pymupdf.Page, dpi: int) -> bytes:
    if dpi <= 0:
        raise ValueError(f"DPI must be positive, got: {dpi}")
    scale = dpi / POINTS_PER_INCH
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    return pixmap.tobytes("png")


def optional_number(value: Any, field: str, page_index: int, path: Path) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"OCR JSON page {page_index} has a non-numeric {field}: {path}"
        )
    return float(value)


def optional_line_count(value: Any, page_index: int, path: Path) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"OCR JSON page {page_index} has a non-integer line_count: {path}"
        )
    return value


def load_ocr_document(path: Path) -> OcrDocument:
    if not path.is_file():
        raise FileNotFoundError(f"OCR JSON file not found: {path}")
    if path.suffix.lower() != JSON_SUFFIX:
        raise ValueError(f"File does not have a {JSON_SUFFIX} suffix: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON ({path}): {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Failed to read OCR JSON ({path}): {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"OCR JSON root must be an object: {path}")
    source = payload.get("source")
    if not isinstance(source, str) or not source:
        raise ValueError(f"OCR JSON is missing a non-empty 'source': {path}")
    pages = payload.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError(f"OCR JSON has no 'pages' entries: {path}")
    records: list[OcrPage] = []
    previous_number = 0
    for index, page in enumerate(pages):
        if not isinstance(page, dict):
            raise ValueError(f"OCR JSON page {index} is not an object: {path}")
        page_number = page.get("page_number")
        if isinstance(page_number, bool) or not isinstance(page_number, int):
            raise ValueError(f"OCR JSON page {index} has a non-integer page_number: {path}")
        if page_number < 1:
            raise ValueError(f"OCR JSON page {index} has a non-positive page_number: {path}")
        if page_number <= previous_number:
            raise ValueError(
                f"OCR JSON page {index} has a non-increasing page_number: {path}"
            )
        previous_number = page_number
        text = page.get("text")
        if not isinstance(text, str):
            raise ValueError(f"OCR JSON page {index} has a non-string text: {path}")
        records.append(
            OcrPage(
                source=source,
                page_number=page_number,
                text=text,
                line_count=optional_line_count(page.get("line_count"), index, path),
                mean_score=optional_number(page.get("mean_score"), "mean_score", index, path),
                duration_seconds=optional_number(
                    page.get("duration_seconds"), "duration_seconds", index, path
                ),
            )
        )
    return OcrDocument(source=source, pages=records)


def request_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode(STDIO_ENCODING)
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode(STDIO_ENCODING)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(STDIO_ENCODING, errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach LM Studio at {url}: {exc.reason}") from exc
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response from {url}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Expected a JSON object from {url}")
    return parsed


def preview_text(text: str) -> str:
    compact = " ".join(text.split())
    if len(compact) > RAW_TEXT_PREVIEW_CHARS:
        return f"{compact[:RAW_TEXT_PREVIEW_CHARS]}..."
    return compact


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_json_object(text: str, page_number: int) -> dict[str, Any]:
    candidate = strip_code_fences(text)
    parsed: Any = None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end <= start:
            raise RuntimeError(
                f"VLM response on page {page_number} contains no JSON object: "
                f"{preview_text(candidate)}"
            )
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"VLM response on page {page_number} is not valid JSON ({exc}): "
                f"{preview_text(candidate)}"
            ) from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"VLM response on page {page_number} is not a JSON object")
    return parsed


def validate_extracted_fields(payload: dict[str, Any], page_number: int) -> ExtractedFields:
    keys = set(payload.keys())
    expected = set(EXTRACTION_FIELDS)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise RuntimeError(
            f"VLM response on page {page_number} has wrong fields "
            f"(missing={missing}, extra={extra})"
        )
    fields: dict[str, str | None] = {}
    for field in EXTRACTION_FIELDS:
        value = payload[field]
        if value is None:
            fields[field] = None
            continue
        if not isinstance(value, str):
            raise RuntimeError(
                f"VLM response on page {page_number} has a non-string {field}"
            )
        if not value.strip():
            raise RuntimeError(
                f"VLM response on page {page_number} has an empty {field}; "
                "expected null for a missing value"
            )
        fields[field] = value
    return ExtractedFields(**fields)


def parse_extraction_response(payload: dict[str, Any], page_number: int) -> ExtractedFields:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(
            f"VLM response on page {page_number} has no 'choices' entries"
        )
    choice = choices[0]
    if not isinstance(choice, dict):
        raise RuntimeError(f"VLM response on page {page_number} has a malformed choice")
    finish_reason = choice.get("finish_reason")
    if finish_reason == "length":
        raise RuntimeError(
            f"VLM response on page {page_number} was truncated "
            "(increase --max-tokens or reduce --dpi)"
        )
    message = choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError(
            f"VLM response on page {page_number} has no message object"
        )
    content = message.get("content")
    reasoning = message.get("reasoning_content")
    text: str | None = None
    if isinstance(content, str) and content.strip():
        text = content
    elif isinstance(reasoning, str) and reasoning.strip():
        text = reasoning
    if text is None:
        raise RuntimeError(f"VLM response on page {page_number} is empty")
    parsed = parse_json_object(text, page_number)
    return validate_extracted_fields(parsed, page_number)


class VlmClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float,
        max_tokens: int,
        seed: int,
        thinking: bool,
        request: RequestJson = request_json,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.seed = seed
        self.thinking = thinking
        self._request = request

    def build_payload(self, png_bytes: bytes) -> dict[str, Any]:
        encoded = base64.b64encode(png_bytes).decode("ascii")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": USER_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"{PNG_DATA_URL_PREFIX}{encoded}"},
                        },
                    ],
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": EXTRACTION_SCHEMA_NAME,
                    "strict": True,
                    "schema": EXTRACTION_JSON_SCHEMA,
                },
            },
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": self.thinking},
        }
        if not self.thinking:
            payload["reasoning_effort"] = REASONING_EFFORT_DISABLED
        return payload

    def extract_page(self, png_bytes: bytes, page_number: int) -> ExtractedFields:
        response = self._request(
            f"{self.base_url}{CHAT_COMPLETIONS_PATH}",
            self.build_payload(png_bytes),
            self.timeout,
        )
        return parse_extraction_response(response, page_number)


def normalize_for_match(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    characters: list[str] = []
    for character in without_marks.casefold():
        if unicodedata.category(character) == "Pd":
            characters.append(" ")
        elif character.isalnum():
            characters.append(character)
        else:
            characters.append(" ")
    return " ".join("".join(characters).split())


def best_window_similarity(value_norm: str, ocr_norm: str) -> float:
    if not value_norm or not ocr_norm:
        return 0.0
    if value_norm in ocr_norm:
        return 1.0
    ocr_tokens = ocr_norm.split()
    value_tokens = value_norm.split()
    token_count = len(value_tokens)
    window_sizes = {max(1, token_count - 1), token_count, token_count + 1}
    best = 0.0
    for size in sorted(window_sizes):
        if size > len(ocr_tokens):
            continue
        for start in range(len(ocr_tokens) - size + 1):
            window = " ".join(ocr_tokens[start : start + size])
            ratio = difflib.SequenceMatcher(None, value_norm, window).ratio()
            if ratio > best:
                best = ratio
                if best == 1.0:
                    return 1.0
    return best


def compare_field(
    value: str | None, ocr_norm: str, threshold: float
) -> FieldVerification:
    if value is None:
        return {
            "status": STATUS_MISSING,
            "found_in_ocr": False,
            "similarity": None,
        }
    similarity = best_window_similarity(normalize_for_match(value), ocr_norm)
    status = STATUS_OK if similarity >= threshold else STATUS_MISMATCH
    return {
        "status": status,
        "found_in_ocr": status == STATUS_OK,
        "similarity": round(similarity, SIMILARITY_PRECISION),
    }


def verify_extracted_fields(
    fields: ExtractedFields, ocr_text: str, threshold: float
) -> tuple[dict[str, FieldVerification], list[str]]:
    ocr_norm = normalize_for_match(ocr_text)
    verification: dict[str, FieldVerification] = {}
    mismatches: list[str] = []
    for field in EXTRACTION_FIELDS:
        result = compare_field(fields[field], ocr_norm, threshold)
        verification[field] = result
        if result["status"] == STATUS_MISMATCH:
            mismatches.append(field)
    return verification, mismatches


def build_summary(pages: Sequence[PageExtraction]) -> ExtractionSummary:
    status_counts: dict[str, FieldStatusCounts] = {
        field: {"ok": 0, "mismatch": 0, "missing": 0} for field in EXTRACTION_FIELDS
    }
    pages_with_mismatch: list[int] = []
    for page in pages:
        for field in EXTRACTION_FIELDS:
            status = page["verification"][field]["status"]
            status_counts[field][status] += 1
        if page["mismatches"]:
            pages_with_mismatch.append(page["page_number"])
    mismatch_count = sum(counts["mismatch"] for counts in status_counts.values())
    missing_count = sum(counts["missing"] for counts in status_counts.values())
    return {
        "page_count": len(pages),
        "field_status_counts": status_counts,
        "field_mismatch_count": mismatch_count,
        "field_missing_count": missing_count,
        "pages_with_mismatch": pages_with_mismatch,
        "needs_review_page_count": len(pages_with_mismatch),
    }


def run_extraction(
    pdf_path: Path,
    ocr_path: Path,
    client: VlmClient,
    dpi: int = DEFAULT_DPI,
    mismatch_threshold: float = DEFAULT_MISMATCH_THRESHOLD,
) -> ExtractionReport:
    if dpi <= 0:
        raise ValueError(f"DPI must be positive, got: {dpi}")
    if not 0.0 <= mismatch_threshold <= 1.0:
        raise ValueError(
            f"Mismatch threshold must be within [0, 1], got: {mismatch_threshold}"
        )
    ocr_document = load_ocr_document(ocr_path)
    if Path(ocr_document.source).name != pdf_path.name:
        raise ValueError(
            f"OCR JSON source {ocr_document.source!r} does not match PDF {pdf_path.name!r}"
        )
    document = open_pdf(pdf_path)
    pdf_page_count = document.page_count
    if pdf_page_count != len(ocr_document.pages):
        document.close()
        raise ValueError(
            f"Page count mismatch: PDF has {pdf_page_count} pages but OCR JSON "
            f"has {len(ocr_document.pages)}"
        )
    log_message(
        f"[vlm] pdf={pdf_path} ocr={ocr_path} pages={pdf_page_count} dpi={dpi} "
        f"model={client.model} base_url={client.base_url} "
        f"threshold={mismatch_threshold} thinking={client.thinking}"
    )
    started = time.perf_counter()
    pages: list[PageExtraction] = []
    render_seconds = 0.0
    vlm_seconds = 0.0
    compare_seconds = 0.0
    try:
        for index, record in enumerate(ocr_document.pages):
            if record.page_number != index + 1:
                raise ValueError(
                    f"OCR JSON page {index} has page_number {record.page_number}; "
                    "pages must be contiguous starting at 1"
                )
            page_started = time.perf_counter()
            render_started = time.perf_counter()
            page = document.load_page(index)
            png_bytes = render_page_png(page, dpi)
            render_duration = time.perf_counter() - render_started
            vlm_started = time.perf_counter()
            try:
                fields = client.extract_page(png_bytes, record.page_number)
            except Exception as exc:
                raise RuntimeError(
                    f"VLM extraction failed on page {record.page_number}: {exc}"
                ) from exc
            vlm_duration = time.perf_counter() - vlm_started
            compare_started = time.perf_counter()
            verification, mismatches = verify_extracted_fields(
                fields, record.text, mismatch_threshold
            )
            compare_duration = time.perf_counter() - compare_started
            page_duration = time.perf_counter() - page_started
            render_seconds += render_duration
            vlm_seconds += vlm_duration
            compare_seconds += compare_duration
            pages.append(
                {
                    "page_number": record.page_number,
                    "page_ref": make_page_ref(record.source, record.page_number),
                    "extracted": fields,
                    "verification": verification,
                    "mismatches": mismatches,
                    "ocr_reference": {
                        "line_count": record.line_count,
                        "mean_score": record.mean_score,
                        "duration_seconds": record.duration_seconds,
                    },
                    "timings_seconds": {
                        "render_seconds": round(render_duration, 3),
                        "vlm_seconds": round(vlm_duration, 3),
                        "compare_seconds": round(compare_duration, 3),
                        "total_seconds": round(page_duration, 3),
                    },
                }
            )
            log_message(
                f"[vlm] page {record.page_number} done "
                f"render={render_duration:.2f}s vlm={vlm_duration:.2f}s "
                f"compare={compare_duration:.3f}s "
                f"mismatches={','.join(mismatches) if mismatches else '-'}"
            )
    finally:
        document.close()
    total_duration = time.perf_counter() - started
    summary = build_summary(pages)
    report: ExtractionReport = {
        "schema_version": SCHEMA_VERSION,
        "input_files": {"pdf": str(pdf_path), "ocr_json": str(ocr_path)},
        "model": {
            "id": client.model,
            "base_url": client.base_url,
            "thinking": client.thinking,
            "prompt_version": PROMPT_VERSION,
        },
        "parameters": {
            "dpi": dpi,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "seed": client.seed,
            "max_tokens": client.max_tokens,
            "reasoning_effort": None if client.thinking else REASONING_EFFORT_DISABLED,
            "mismatch_similarity_threshold": mismatch_threshold,
            "text_normalization": TEXT_NORMALIZATION,
        },
        "summary": summary,
        "pages": pages,
        "timings": {
            "total_seconds": round(total_duration, 3),
            "render_seconds": round(render_seconds, 3),
            "vlm_seconds": round(vlm_seconds, 3),
            "compare_seconds": round(compare_seconds, 3),
            "vlm_requests": len(pages),
            "avg_vlm_seconds_per_page": round(vlm_seconds / len(pages), 3)
            if pages
            else None,
        },
    }
    average = vlm_seconds / len(pages) if pages else 0.0
    log_message(
        f"[vlm] total duration={total_duration:.2f}s pages={len(pages)} "
        f"vlm={vlm_seconds:.2f}s avg_per_page={average:.2f}s"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract borrower_name, property_address, loan_number, and page_number "
            "from each PDF page with a vision model served by LM Studio, verify the "
            "values against OCR JSON text, and report mismatches and per-page timings."
        )
    )
    parser.add_argument("pdf", type=Path, help="PDF file to process")
    parser.add_argument(
        "--ocr",
        type=Path,
        default=None,
        help=f"OCR JSON produced by ocr_pdf.py (default: PDF path with {JSON_SUFFIX} suffix)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"JSON output file (default: PDF path with {DEFAULT_OUTPUT_SUFFIX} suffix)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print JSON to stdout instead of writing a file",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"LM Studio OpenAI-compatible base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LM Studio model id (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"Page render resolution (default: {DEFAULT_DPI})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"HTTP timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum response tokens per page (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Sampling seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--mismatch-threshold",
        type=float,
        default=DEFAULT_MISMATCH_THRESHOLD,
        help=(
            "Similarity below this value is reported as a mismatch "
            f"(default: {DEFAULT_MISMATCH_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable model thinking mode (disabled by default for deterministic output)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_stdio()
    args = build_parser().parse_args(argv)
    try:
        if not args.base_url:
            raise ValueError("--base-url must not be empty")
        if not args.model:
            raise ValueError("--model must not be empty")
        if args.dpi <= 0:
            raise ValueError(f"--dpi must be positive, got: {args.dpi}")
        if args.timeout <= 0:
            raise ValueError(f"--timeout must be positive, got: {args.timeout}")
        if args.max_tokens <= 0:
            raise ValueError(f"--max-tokens must be positive, got: {args.max_tokens}")
        if not 0.0 <= args.mismatch_threshold <= 1.0:
            raise ValueError(
                f"--mismatch-threshold must be within [0, 1], got: "
                f"{args.mismatch_threshold}"
            )
        ocr_path = args.ocr if args.ocr is not None else default_ocr_path(args.pdf)
        client = VlmClient(
            base_url=args.base_url,
            model=args.model,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            seed=args.seed,
            thinking=args.thinking,
            request=request_json,
        )
        report = run_extraction(
            args.pdf,
            ocr_path,
            client,
            dpi=args.dpi,
            mismatch_threshold=args.mismatch_threshold,
        )
    except Exception as exc:
        log_message(f"ERROR: {type(exc).__name__}: {exc}")
        return 1
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.stdout:
        print(payload)
        return 0
    out_path = args.out if args.out is not None else default_output_path(args.pdf)
    try:
        out_path.write_text(payload, encoding="utf-8")
    except OSError as exc:
        log_message(f"ERROR: failed to write JSON file ({out_path}): {exc}")
        return 1
    summary = report["summary"]
    log_message(
        f"[vlm] mismatched fields={summary['field_mismatch_count']} "
        f"missing fields={summary['field_missing_count']} "
        f"pages_with_mismatch={summary['pages_with_mismatch']}"
    )
    log_message(f"[vlm] JSON written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
