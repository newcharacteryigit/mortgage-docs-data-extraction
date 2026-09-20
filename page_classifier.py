from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, TypedDict

DEFAULT_BASE_URL = "https://api.typesafe.ai/v1"
DEFAULT_MODEL = "jev-latest"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_MAX_STATE_CHARS = 30000
DEFAULT_ENV_FILE = ".env"
SYSTEM_ONE_PATH = "/systemone"
QUESTION_ID = "page_category"
SCHEMA_VERSION = "1.0"
PROMPT_VERSION = "1.0"
JSON_SUFFIX = ".json"
DEFAULT_OUTPUT_SUFFIX = ".categories.json"
STDIO_ENCODING = "utf-8"
API_KEY_ENV_VAR = "TYPESAFE_API_KEY"
PROBABILITY_SUM_TOLERANCE = 0.001
CONFIDENCE_PRECISION = 4
PROBABILITY_PRECISION = 4
MAX_PROBABILITY = 1.0
MIN_PROBABILITY = 0.0

REVIEW_REASON_EMPTY_TEXT = "empty text"
REVIEW_REASON_LOW_CONFIDENCE = "low confidence"
REVIEW_REASON_OTHER_CATEGORY = "other category"

CATEGORY_LABELS: dict[str, str] = {
    "mortgage_closing_disclosure_seller": "Mortgage - Closing Disclosure - Seller",
    "lender_rate_note": "Lender - Rate Note",
    "title_rider": "Title - Rider",
    "property_tax_record_information_sheet": "Property - Tax Record Information Sheet",
    "title_signature_name_affidavit_ack": "Title - Signature / Name Affidavit (Ack)",
    "other": "Other / Unclassified",
}

CATEGORY_CRITERIA: dict[str, str] = {
    "mortgage_closing_disclosure_seller": (
        "Closing Disclosure for the seller: closing information, transaction "
        "information, loan costs, and seller-paid closing cost details."
    ),
    "lender_rate_note": (
        "Lender's promissory note or rate note: NOTE or UNIFORM SECURED NOTE "
        "headings, loan amount, interest rate, and payment terms."
    ),
    "title_rider": (
        "Title rider attached to a security instrument, such as a CONDOMINIUM "
        "RIDER, with numbered covenant paragraphs."
    ),
    "property_tax_record_information_sheet": (
        "Property tax record information sheet: TAX RECORD INFORMATION SHEET "
        "heading, type of tax, amounts paid, and next due date."
    ),
    "title_signature_name_affidavit_ack": (
        "Signature or name affidavit (acknowledgment): SIGNATURE/NAME AFFIDAVIT "
        "heading, borrower names, and notary or acknowledgment language."
    ),
    "other": "A page that does not match any category above.",
}

QUESTION_INSTRUCTIONS = (
    "Which document type is this page from a mortgage loan package? "
    "Classify by the page's printed title, headings, form text, and loan number. "
    "A continuation page without its own title belongs to the document type it "
    "continues. Use 'other' only when the page does not match any category."
)

RequestJson = Callable[[str, dict[str, Any], float, str], dict[str, Any]]


class OcrReference(TypedDict):
    line_count: int | None
    mean_score: float | None
    duration_seconds: float | None


class PageTimings(TypedDict):
    classify_seconds: float
    total_seconds: float


class TokenUsage(TypedDict):
    input_tokens: int
    output_tokens: int


class PageClassification(TypedDict):
    page_number: int
    page_ref: str
    category: str | None
    category_label: str | None
    confidence: float | None
    probabilities: dict[str, float] | None
    needs_review: bool
    review_reason: str | None
    ocr_reference: OcrReference
    timings_seconds: PageTimings
    usage: TokenUsage | None


class SkippedPage(TypedDict):
    page_number: int
    page_ref: str
    reason: str


class ClassificationSummary(TypedDict):
    page_count: int
    classified_page_count: int
    skipped_page_count: int
    needs_review_page_count: int
    needs_review_pages: list[int]
    category_counts: dict[str, int]


class ClassificationTimings(TypedDict):
    total_seconds: float
    classify_seconds: float
    requests: int
    avg_classify_seconds_per_page: float | None


class TotalUsage(TypedDict):
    input_tokens: int
    output_tokens: int
    requests: int


class ModelInfo(TypedDict):
    id: str
    base_url: str
    prompt_version: str
    response_model: str | None


class ClassificationParameters(TypedDict):
    confidence_threshold: float
    max_state_chars: int
    timeout_seconds: float


class ClassificationReport(TypedDict):
    schema_version: str
    input_files: dict[str, str]
    model: ModelInfo
    categories: dict[str, str]
    parameters: ClassificationParameters
    summary: ClassificationSummary
    pages: list[PageClassification]
    skipped_pages: list[SkippedPage]
    timings: ClassificationTimings
    usage: TotalUsage


@dataclass(frozen=True)
class OcrPage:
    page_number: int
    text: str
    line_count: int | None
    mean_score: float | None
    duration_seconds: float | None


class OcrDocument(NamedTuple):
    source: str
    pages: list[OcrPage]


@dataclass(frozen=True)
class ClassificationResult:
    category: str
    confidence: float
    probabilities: dict[str, float]
    response_model: str
    usage: TokenUsage


def log_message(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding=STDIO_ENCODING, errors="replace")


def default_output_path(ocr_path: Path) -> Path:
    return ocr_path.parent / f"{ocr_path.stem}{DEFAULT_OUTPUT_SUFFIX}"


def make_page_ref(source: str, page_number: int) -> str:
    return f"{source}#page-{page_number}"


def parse_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    entries: dict[str, str] = {}
    for raw_line in path.read_text(encoding=STDIO_ENCODING).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        entries[key] = value
    return entries


def resolve_api_key(env_file: Path) -> str:
    file_entries = parse_env_file(env_file)
    api_key = os.environ.get(API_KEY_ENV_VAR, "").strip()
    if not api_key:
        api_key = file_entries.get(API_KEY_ENV_VAR, "").strip()
    if not api_key:
        raise ValueError(
            f"{API_KEY_ENV_VAR} is not set; add it to {env_file} or the environment"
        )
    return api_key


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
        payload = json.loads(path.read_text(encoding=STDIO_ENCODING))
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
            raise ValueError(
                f"OCR JSON page {index} has a non-integer page_number: {path}"
            )
        if page_number < 1:
            raise ValueError(
                f"OCR JSON page {index} has a non-positive page_number: {path}"
            )
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
                page_number=page_number,
                text=text,
                line_count=optional_line_count(page.get("line_count"), index, path),
                mean_score=optional_number(
                    page.get("mean_score"), "mean_score", index, path
                ),
                duration_seconds=optional_number(
                    page.get("duration_seconds"), "duration_seconds", index, path
                ),
            )
        )
    return OcrDocument(source=source, pages=records)


def request_json(
    url: str, payload: dict[str, Any], timeout: float, api_key: str
) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode(STDIO_ENCODING)
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode(STDIO_ENCODING)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(STDIO_ENCODING, errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach TypeSafe at {url}: {exc.reason}") from exc
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response from {url}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Expected a JSON object from {url}")
    return parsed


def require_number(value: Any, field: str, page_number: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(
            f"TypeSafe response on page {page_number} has a non-numeric {field}"
        )
    return float(value)


def parse_usage(payload: dict[str, Any], page_number: int) -> TokenUsage:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        raise RuntimeError(
            f"TypeSafe response on page {page_number} has no usage object"
        )
    counts: dict[str, int] = {}
    for field in ("input_tokens", "output_tokens"):
        value = usage.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError(
                f"TypeSafe response on page {page_number} has a non-integer {field}"
            )
        counts[field] = value
    return TokenUsage(
        input_tokens=counts["input_tokens"], output_tokens=counts["output_tokens"]
    )


def parse_probabilities(
    value: Any, choice: str, page_number: int
) -> dict[str, float]:
    if not isinstance(value, dict):
        raise RuntimeError(
            f"TypeSafe response on page {page_number} has non-object probabilities"
        )
    keys = set(value.keys())
    expected = set(CATEGORY_LABELS)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise RuntimeError(
            f"TypeSafe response on page {page_number} has wrong probability keys "
            f"(missing={missing}, extra={extra})"
        )
    probabilities: dict[str, float] = {}
    for option, probability in value.items():
        number = require_number(
            probability, f"probability for {option!r}", page_number
        )
        if not MIN_PROBABILITY <= number <= MAX_PROBABILITY:
            raise RuntimeError(
                f"TypeSafe response on page {page_number} has an out-of-range "
                f"probability for {option!r}: {number}"
            )
        probabilities[option] = number
    total = sum(probabilities.values())
    if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
        raise RuntimeError(
            f"TypeSafe response on page {page_number} has probabilities summing "
            f"to {total}"
        )
    if probabilities[choice] != max(probabilities.values()):
        raise RuntimeError(
            f"TypeSafe response on page {page_number} picked {choice!r} without "
            "the highest probability"
        )
    return probabilities


def parse_classification_response(
    payload: dict[str, Any], page_number: int
) -> ClassificationResult:
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        raise RuntimeError(
            f"TypeSafe response on page {page_number} has no answers object"
        )
    answer = answers.get(QUESTION_ID)
    if not isinstance(answer, dict):
        raise RuntimeError(
            f"TypeSafe response on page {page_number} is missing answer {QUESTION_ID!r}"
        )
    if answer.get("type") != "choice":
        raise RuntimeError(
            f"TypeSafe response on page {page_number} has a non-choice answer"
        )
    choice = answer.get("choice")
    if not isinstance(choice, str) or choice not in CATEGORY_LABELS:
        raise RuntimeError(
            f"TypeSafe response on page {page_number} has an unknown category: "
            f"{choice!r}"
        )
    confidence = require_number(answer.get("confidence"), "confidence", page_number)
    if not MIN_PROBABILITY <= confidence <= MAX_PROBABILITY:
        raise RuntimeError(
            f"TypeSafe response on page {page_number} has an out-of-range "
            f"confidence: {confidence}"
        )
    probabilities = parse_probabilities(answer.get("probabilities"), choice, page_number)
    response_model = payload.get("model")
    if not isinstance(response_model, str) or not response_model:
        raise RuntimeError(
            f"TypeSafe response on page {page_number} has no model id"
        )
    return ClassificationResult(
        category=choice,
        confidence=confidence,
        probabilities=probabilities,
        response_model=response_model,
        usage=parse_usage(payload, page_number),
    )


class TypesafeClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        timeout: float,
        request: RequestJson = request_json,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._request = request

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}{SYSTEM_ONE_PATH}"

    def build_payload(self, state: str) -> dict[str, Any]:
        return {
            "state": state,
            "model": self.model,
            "questions": {
                QUESTION_ID: {
                    "type": "choice",
                    "instructions": QUESTION_INSTRUCTIONS,
                    "criteria": CATEGORY_CRITERIA,
                }
            },
        }

    def classify_page(self, text: str, page_number: int) -> ClassificationResult:
        response = self._request(
            self.endpoint, self.build_payload(text), self.timeout, self.api_key
        )
        return parse_classification_response(response, page_number)


def review_decision(
    category: str, confidence: float, confidence_threshold: float
) -> tuple[bool, str | None]:
    if confidence < confidence_threshold:
        return True, REVIEW_REASON_LOW_CONFIDENCE
    if category == "other":
        return True, REVIEW_REASON_OTHER_CATEGORY
    return False, None


def build_ocr_reference(record: OcrPage) -> OcrReference:
    return {
        "line_count": record.line_count,
        "mean_score": record.mean_score,
        "duration_seconds": record.duration_seconds,
    }


def skipped_page_record(record: OcrPage) -> PageClassification:
    return {
        "page_number": record.page_number,
        "page_ref": "",
        "category": None,
        "category_label": None,
        "confidence": None,
        "probabilities": None,
        "needs_review": True,
        "review_reason": REVIEW_REASON_EMPTY_TEXT,
        "ocr_reference": build_ocr_reference(record),
        "timings_seconds": {"classify_seconds": 0.0, "total_seconds": 0.0},
        "usage": None,
    }


def validate_page_text(record: OcrPage, max_state_chars: int) -> None:
    if record.text.strip() and len(record.text) > max_state_chars:
        raise ValueError(
            f"Page {record.page_number} text has {len(record.text)} characters, "
            f"exceeding the {max_state_chars} character state limit"
        )


def run_classification(
    ocr_path: Path,
    client: TypesafeClient,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    max_state_chars: int = DEFAULT_MAX_STATE_CHARS,
) -> ClassificationReport:
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError(
            f"Confidence threshold must be within [0, 1], got: {confidence_threshold}"
        )
    if max_state_chars <= 0:
        raise ValueError(f"Max state chars must be positive, got: {max_state_chars}")
    ocr_document = load_ocr_document(ocr_path)
    for record in ocr_document.pages:
        validate_page_text(record, max_state_chars)
    log_message(
        f"[classify] ocr={ocr_path} pages={len(ocr_document.pages)} "
        f"model={client.model} base_url={client.base_url} "
        f"threshold={confidence_threshold} max_state_chars={max_state_chars}"
    )
    started = time.perf_counter()
    pages: list[PageClassification] = []
    skipped_pages: list[SkippedPage] = []
    classify_seconds = 0.0
    input_tokens = 0
    output_tokens = 0
    response_model: str | None = None
    for record in ocr_document.pages:
        page_ref = make_page_ref(ocr_document.source, record.page_number)
        page_started = time.perf_counter()
        if not record.text.strip():
            skipped = skipped_page_record(record)
            skipped["page_ref"] = page_ref
            pages.append(skipped)
            skipped_pages.append(
                {
                    "page_number": record.page_number,
                    "page_ref": page_ref,
                    "reason": REVIEW_REASON_EMPTY_TEXT,
                }
            )
            log_message(
                f"[classify] page {record.page_number} skipped reason=empty text"
            )
            continue
        classify_started = time.perf_counter()
        try:
            result = client.classify_page(record.text, record.page_number)
        except Exception as exc:
            raise RuntimeError(
                f"Classification failed on page {record.page_number}: {exc}"
            ) from exc
        duration = time.perf_counter() - classify_started
        page_duration = time.perf_counter() - page_started
        classify_seconds += duration
        input_tokens += result.usage["input_tokens"]
        output_tokens += result.usage["output_tokens"]
        if response_model is None:
            response_model = result.response_model
        needs_review, review_reason = review_decision(
            result.category, result.confidence, confidence_threshold
        )
        pages.append(
            {
                "page_number": record.page_number,
                "page_ref": page_ref,
                "category": result.category,
                "category_label": CATEGORY_LABELS[result.category],
                "confidence": round(result.confidence, CONFIDENCE_PRECISION),
                "probabilities": {
                    option: round(probability, PROBABILITY_PRECISION)
                    for option, probability in result.probabilities.items()
                },
                "needs_review": needs_review,
                "review_reason": review_reason,
                "ocr_reference": build_ocr_reference(record),
                "timings_seconds": {
                    "classify_seconds": round(duration, 3),
                    "total_seconds": round(page_duration, 3),
                },
                "usage": result.usage,
            }
        )
        log_message(
            f"[classify] page {record.page_number} done "
            f"category={result.category} confidence={result.confidence:.4f} "
            f"duration={duration:.2f}s "
            f"review={review_reason if review_reason else '-'}"
        )
    total_duration = time.perf_counter() - started
    classified_pages = [
        page for page in pages if page["review_reason"] != REVIEW_REASON_EMPTY_TEXT
    ]
    category_counts = {category: 0 for category in CATEGORY_LABELS}
    for page in classified_pages:
        category = page["category"]
        if category is not None:
            category_counts[category] += 1
    summary: ClassificationSummary = {
        "page_count": len(pages),
        "classified_page_count": len(classified_pages),
        "skipped_page_count": len(skipped_pages),
        "needs_review_page_count": sum(
            1 for page in pages if page["needs_review"]
        ),
        "needs_review_pages": [
            page["page_number"] for page in pages if page["needs_review"]
        ],
        "category_counts": category_counts,
    }
    report: ClassificationReport = {
        "schema_version": SCHEMA_VERSION,
        "input_files": {"ocr_json": str(ocr_path)},
        "model": {
            "id": client.model,
            "base_url": client.base_url,
            "prompt_version": PROMPT_VERSION,
            "response_model": response_model,
        },
        "categories": dict(CATEGORY_LABELS),
        "parameters": {
            "confidence_threshold": confidence_threshold,
            "max_state_chars": max_state_chars,
            "timeout_seconds": client.timeout,
        },
        "summary": summary,
        "pages": pages,
        "skipped_pages": skipped_pages,
        "timings": {
            "total_seconds": round(total_duration, 3),
            "classify_seconds": round(classify_seconds, 3),
            "requests": len(classified_pages),
            "avg_classify_seconds_per_page": round(
                classify_seconds / len(classified_pages), 3
            )
            if classified_pages
            else None,
        },
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "requests": len(classified_pages),
        },
    }
    average = classify_seconds / len(classified_pages) if classified_pages else 0.0
    log_message(
        f"[classify] total duration={total_duration:.2f}s pages={len(pages)} "
        f"requests={len(classified_pages)} avg_per_page={average:.2f}s"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Classify every OCR JSON page of a mortgage document into a fixed "
            "category set with the TypeSafe Jev Choice primitive, and report the "
            "chosen category, confidence, per-page timings, and token usage."
        )
    )
    parser.add_argument(
        "ocr_json", type=Path, help="OCR JSON produced by ocr_pdf.py"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            f"JSON output file (default: OCR path with {DEFAULT_OUTPUT_SUFFIX} suffix)"
        ),
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print JSON to stdout instead of writing a file",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"TypeSafe API base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"TypeSafe model id or alias (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"HTTP timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help=(
            "Answers below this confidence are flagged as needs_review "
            f"(default: {DEFAULT_CONFIDENCE_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--max-state-chars",
        type=int,
        default=DEFAULT_MAX_STATE_CHARS,
        help=(
            "Fail fast when a page text exceeds this many characters "
            f"(default: {DEFAULT_MAX_STATE_CHARS})"
        ),
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(DEFAULT_ENV_FILE),
        help=f"File providing {API_KEY_ENV_VAR} (default: {DEFAULT_ENV_FILE})",
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
        if args.timeout <= 0:
            raise ValueError(f"--timeout must be positive, got: {args.timeout}")
        if not 0.0 <= args.confidence_threshold <= 1.0:
            raise ValueError(
                f"--confidence-threshold must be within [0, 1], got: "
                f"{args.confidence_threshold}"
            )
        if args.max_state_chars <= 0:
            raise ValueError(
                f"--max-state-chars must be positive, got: {args.max_state_chars}"
            )
        api_key = resolve_api_key(args.env_file)
        client = TypesafeClient(
            base_url=args.base_url,
            model=args.model,
            api_key=api_key,
            timeout=args.timeout,
            request=request_json,
        )
        report = run_classification(
            args.ocr_json,
            client,
            confidence_threshold=args.confidence_threshold,
            max_state_chars=args.max_state_chars,
        )
    except Exception as exc:
        log_message(f"ERROR: {type(exc).__name__}: {exc}")
        return 1
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.stdout:
        print(payload)
        return 0
    out_path = args.out if args.out is not None else default_output_path(args.ocr_json)
    try:
        out_path.write_text(payload, encoding=STDIO_ENCODING)
    except OSError as exc:
        log_message(f"ERROR: failed to write JSON file ({out_path}): {exc}")
        return 1
    summary = report["summary"]
    log_message(
        f"[classify] needs_review pages={summary['needs_review_pages']} "
        f"categories={summary['category_counts']}"
    )
    log_message(f"[classify] JSON written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
