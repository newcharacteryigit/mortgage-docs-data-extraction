from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, TypedDict

from page_classifier import (
    CATEGORY_CRITERIA,
    CATEGORY_LABELS,
    DEFAULT_MAX_STATE_CHARS,
    QUESTION_INSTRUCTIONS,
    REVIEW_REASON_OTHER_CATEGORY,
    load_ocr_document,
    make_page_ref,
    validate_page_text,
)

DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_MODEL = "google/gemma-4-e2b"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_TOKENS = 128
DEFAULT_SEED = 0
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_MARGIN_THRESHOLD = 0.5
TEMPERATURE = 0.0
TOP_P = 1.0
REASONING_EFFORT_DISABLED = "none"
CHAT_COMPLETIONS_PATH = "/chat/completions"
SCHEMA_VERSION = "1.0"
PROMPT_VERSION = "1.0"
INPUT_MODE = "ocr_text"
RESPONSE_SCHEMA_NAME = "page_category_verification"
DEFAULT_OUTPUT_SUFFIX = ".category_review.json"
CATEGORIES_SUFFIX = ".categories.json"
JSON_SUFFIX = ".json"
STDIO_ENCODING = "utf-8"
MIN_PROBABILITY = 0.0
MAX_PROBABILITY = 1.0
PROBABILITY_SUM_TOLERANCE = 0.001
MARGIN_PRECISION = 4
SECONDS_PRECISION = 3
RAW_TEXT_PREVIEW_CHARS = 200

TRIGGER_LOW_CONFIDENCE = "low confidence"
TRIGGER_LOW_MARGIN = "low margin"

RESOLUTION_NOT_TRIGGERED = "not_triggered"
RESOLUTION_ACCEPTED = "accepted"
RESOLUTION_MUTUAL_OTHER = "mutual_other"
RESOLUTION_DISAGREEMENT = "disagreement"
RESOLUTION_VERIFIER_FAILED = "verifier_failed"

REVIEW_REASON_DISAGREEMENT = "verifier disagreement"
REVIEW_REASON_VERIFIER_FAILED = "verifier failed"

OTHER_CATEGORY = "other"

RequestJson = Callable[[str, dict[str, Any], float], dict[str, Any]]

CATEGORY_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"category": {"type": "string", "enum": sorted(CATEGORY_LABELS)}},
    "required": ["category"],
    "additionalProperties": False,
}


def build_system_prompt() -> str:
    criteria_lines = "\n".join(
        f"- {category}: {CATEGORY_CRITERIA[category]}" for category in CATEGORY_LABELS
    )
    return (
        "You verify the document category of a single page from a mortgage loan "
        "package. Choose exactly one category key from this list:\n"
        f"{criteria_lines}\n\n"
        f"{QUESTION_INSTRUCTIONS}\n"
        "Answer with the exact category key. Reply with JSON only."
    )


SYSTEM_PROMPT = build_system_prompt()


class VerifierUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int


class PageVerification(TypedDict):
    triggered: bool
    trigger_reasons: list[str]
    margin: float | None
    verifier_category: str | None
    resolution: str
    error: str | None
    verifier_seconds: float
    usage: VerifierUsage | None


class VerificationSummary(TypedDict):
    page_count: int
    triggered_page_count: int
    accepted_page_count: int
    mutual_other_page_count: int
    disagreement_page_count: int
    verifier_failed_page_count: int
    needs_review_before: int
    needs_review_after: int
    triggered_pages: list[int]


class VerifierModelInfo(TypedDict):
    id: str
    base_url: str
    prompt_version: str
    response_model: str | None


class VerifierParameters(TypedDict):
    input_mode: str
    confidence_threshold: float
    margin_threshold: float
    max_state_chars: int
    timeout_seconds: float
    temperature: float
    top_p: float
    seed: int
    max_tokens: int
    reasoning_effort: str


class VerifierTimings(TypedDict):
    total_seconds: float
    verify_seconds: float
    requests: int
    avg_verify_seconds_per_request: float | None


class VerifierTotalUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    requests: int


class VerificationReport(TypedDict):
    schema_version: str
    input_files: dict[str, str]
    model: VerifierModelInfo
    categories: dict[str, str]
    parameters: VerifierParameters
    summary: VerificationSummary
    pages: list[dict[str, Any]]
    timings: VerifierTimings
    usage: VerifierTotalUsage


@dataclass(frozen=True)
class CategoryRecord:
    page_number: int
    page_ref: str
    category: str | None
    confidence: float | None
    probabilities: dict[str, float] | None
    needs_review: bool
    review_reason: str | None
    raw: dict[str, Any]


class CategoryReport(NamedTuple):
    ocr_json_path: str
    categories: dict[str, str]
    records: list[CategoryRecord]


@dataclass(frozen=True)
class TriggerDecision:
    triggered: bool
    reasons: tuple[str, ...]
    margin: float | None


@dataclass(frozen=True)
class VerifierAnswer:
    category: str
    response_model: str
    usage: VerifierUsage


def log_message(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding=STDIO_ENCODING, errors="replace")


def default_output_path(categories_path: Path) -> Path:
    name = categories_path.name
    if name.endswith(CATEGORIES_SUFFIX):
        stem = name[: -len(CATEGORIES_SUFFIX)]
    else:
        stem = categories_path.stem
    return categories_path.parent / f"{stem}{DEFAULT_OUTPUT_SUFFIX}"


def require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    return float(value)


def parse_probabilities(
    value: Any, category: str, page_number: int, path: Path
) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError(
            f"Categories JSON page {page_number} has non-object probabilities: {path}"
        )
    keys = set(value.keys())
    expected = set(CATEGORY_LABELS)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise ValueError(
            f"Categories JSON page {page_number} has wrong probability keys "
            f"(missing={missing}, extra={extra}): {path}"
        )
    probabilities: dict[str, float] = {}
    for option, probability in value.items():
        number = require_number(
            probability, f"Categories JSON page {page_number} probability for {option!r}"
        )
        if not MIN_PROBABILITY <= number <= MAX_PROBABILITY:
            raise ValueError(
                f"Categories JSON page {page_number} has an out-of-range probability "
                f"for {option!r}: {number}"
            )
        probabilities[option] = number
    total = sum(probabilities.values())
    if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
        raise ValueError(
            f"Categories JSON page {page_number} has probabilities summing to {total}"
        )
    if probabilities[category] != max(probabilities.values()):
        raise ValueError(
            f"Categories JSON page {page_number} category {category!r} does not have "
            "the highest probability"
        )
    return probabilities


def load_categories_report(path: Path) -> CategoryReport:
    if not path.is_file():
        raise FileNotFoundError(f"Categories JSON file not found: {path}")
    if path.suffix.lower() != JSON_SUFFIX:
        raise ValueError(f"File does not have a {JSON_SUFFIX} suffix: {path}")
    try:
        payload = json.loads(path.read_text(encoding=STDIO_ENCODING))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON ({path}): {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Failed to read Categories JSON ({path}): {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Categories JSON root must be an object: {path}")
    input_files = payload.get("input_files")
    if not isinstance(input_files, dict):
        raise ValueError(f"Categories JSON is missing 'input_files': {path}")
    ocr_json_path = input_files.get("ocr_json")
    if not isinstance(ocr_json_path, str) or not ocr_json_path:
        raise ValueError(f"Categories JSON is missing a non-empty 'input_files.ocr_json': {path}")
    categories = payload.get("categories")
    if categories is None:
        categories = dict(CATEGORY_LABELS)
    if not isinstance(categories, dict) or set(categories) != set(CATEGORY_LABELS):
        raise ValueError(
            f"Categories JSON has an unexpected 'categories' mapping: {path}"
        )
    labels: dict[str, str] = {}
    for key, value in categories.items():
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"Categories JSON has an invalid label for category {key!r}: {path}"
            )
        labels[key] = value
    pages = payload.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError(f"Categories JSON has no 'pages' entries: {path}")
    records: list[CategoryRecord] = []
    previous_number = 0
    for index, page in enumerate(pages):
        if not isinstance(page, dict):
            raise ValueError(f"Categories JSON page {index} is not an object: {path}")
        page_number = page.get("page_number")
        if isinstance(page_number, bool) or not isinstance(page_number, int):
            raise ValueError(
                f"Categories JSON page {index} has a non-integer page_number: {path}"
            )
        if page_number < 1:
            raise ValueError(
                f"Categories JSON page {index} has a non-positive page_number: {path}"
            )
        if page_number <= previous_number:
            raise ValueError(
                f"Categories JSON page {index} has a non-increasing page_number: {path}"
            )
        previous_number = page_number
        page_ref = page.get("page_ref")
        if not isinstance(page_ref, str) or not page_ref:
            raise ValueError(
                f"Categories JSON page {index} has no 'page_ref': {path}"
            )
        category = page.get("category")
        if category is not None and (
            not isinstance(category, str) or category not in CATEGORY_LABELS
        ):
            raise ValueError(
                f"Categories JSON page {index} has an unknown category: {category!r}"
            )
        needs_review = page.get("needs_review")
        if not isinstance(needs_review, bool):
            raise ValueError(
                f"Categories JSON page {index} has a non-boolean 'needs_review': {path}"
            )
        review_reason = page.get("review_reason")
        if review_reason is not None and (
            not isinstance(review_reason, str) or not review_reason
        ):
            raise ValueError(
                f"Categories JSON page {index} has an invalid 'review_reason': {path}"
            )
        confidence: float | None = None
        probabilities: dict[str, float] | None = None
        if category is not None:
            raw_confidence = page.get("confidence")
            if raw_confidence is None:
                raise ValueError(
                    f"Categories JSON page {index} has a category but no confidence: "
                    f"{path}"
                )
            confidence = require_number(
                raw_confidence, f"Categories JSON page {index} confidence"
            )
            if not MIN_PROBABILITY <= confidence <= MAX_PROBABILITY:
                raise ValueError(
                    f"Categories JSON page {index} has a confidence outside [0, 1]: "
                    f"{confidence}"
                )
            probabilities = parse_probabilities(
                page.get("probabilities"), category, page_number, path
            )
        records.append(
            CategoryRecord(
                page_number=page_number,
                page_ref=page_ref,
                category=category,
                confidence=confidence,
                probabilities=probabilities,
                needs_review=needs_review,
                review_reason=review_reason,
                raw=page,
            )
        )
    return CategoryReport(
        ocr_json_path=ocr_json_path,
        categories=labels,
        records=records,
    )


def resolve_ocr_path(categories_path: Path, stored_path: str) -> Path:
    candidate = Path(stored_path)
    if candidate.is_file():
        return candidate
    sibling = categories_path.parent / stored_path
    if sibling.is_file():
        return sibling
    raise FileNotFoundError(
        f"OCR JSON file not found: {stored_path} (referenced by {categories_path})"
    )


def evaluate_trigger(
    record: CategoryRecord,
    confidence_threshold: float,
    margin_threshold: float,
) -> TriggerDecision:
    if record.category is None:
        return TriggerDecision(triggered=False, reasons=(), margin=None)
    if record.confidence is None or record.probabilities is None:
        raise ValueError(
            f"Categories JSON page {record.page_number} has a category but no "
            "confidence or probabilities"
        )
    values = sorted(record.probabilities.values(), reverse=True)
    margin = values[0] - values[1]
    reasons: list[str] = []
    if record.confidence < confidence_threshold:
        reasons.append(TRIGGER_LOW_CONFIDENCE)
    if margin < margin_threshold:
        reasons.append(TRIGGER_LOW_MARGIN)
    return TriggerDecision(
        triggered=bool(reasons),
        reasons=tuple(reasons),
        margin=margin,
    )


def resolve_decision(jeev_category: str, verifier_category: str) -> str:
    if jeev_category == OTHER_CATEGORY and verifier_category == OTHER_CATEGORY:
        return RESOLUTION_MUTUAL_OTHER
    if jeev_category == verifier_category:
        return RESOLUTION_ACCEPTED
    return RESOLUTION_DISAGREEMENT


def review_outcome(
    record: CategoryRecord, resolution: str
) -> tuple[bool, str | None]:
    if resolution == RESOLUTION_ACCEPTED:
        return False, None
    if resolution == RESOLUTION_MUTUAL_OTHER:
        return True, REVIEW_REASON_OTHER_CATEGORY
    if resolution == RESOLUTION_DISAGREEMENT:
        return True, REVIEW_REASON_DISAGREEMENT
    if resolution == RESOLUTION_VERIFIER_FAILED:
        return True, REVIEW_REASON_VERIFIER_FAILED
    return record.needs_review, record.review_reason


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
                f"Verifier response on page {page_number} contains no JSON object: "
                f"{preview_text(candidate)}"
            )
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Verifier response on page {page_number} is not valid JSON ({exc}): "
                f"{preview_text(candidate)}"
            ) from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"Verifier response on page {page_number} is not a JSON object"
        )
    return parsed


def parse_usage(payload: dict[str, Any], page_number: int) -> VerifierUsage:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        raise RuntimeError(
            f"Verifier response on page {page_number} has no usage object"
        )
    counts: dict[str, int] = {}
    for field in ("prompt_tokens", "completion_tokens"):
        value = usage.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError(
                f"Verifier response on page {page_number} has a non-integer {field}"
            )
        counts[field] = value
    return VerifierUsage(
        prompt_tokens=counts["prompt_tokens"],
        completion_tokens=counts["completion_tokens"],
    )


def parse_verifier_response(payload: dict[str, Any], page_number: int) -> VerifierAnswer:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(
            f"Verifier response on page {page_number} has no 'choices' entries"
        )
    choice = choices[0]
    if not isinstance(choice, dict):
        raise RuntimeError(
            f"Verifier response on page {page_number} has a malformed choice"
        )
    if choice.get("finish_reason") == "length":
        raise RuntimeError(
            f"Verifier response on page {page_number} was truncated "
            "(increase --max-tokens)"
        )
    message = choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError(
            f"Verifier response on page {page_number} has no message object"
        )
    content = message.get("content")
    reasoning = message.get("reasoning_content")
    text: str | None = None
    if isinstance(content, str) and content.strip():
        text = content
    elif isinstance(reasoning, str) and reasoning.strip():
        text = reasoning
    if text is None:
        raise RuntimeError(f"Verifier response on page {page_number} is empty")
    parsed = parse_json_object(text, page_number)
    if set(parsed.keys()) != {"category"}:
        raise RuntimeError(
            f"Verifier response on page {page_number} has wrong fields: "
            f"{sorted(parsed.keys())}"
        )
    category = parsed["category"]
    if not isinstance(category, str) or category not in CATEGORY_LABELS:
        raise RuntimeError(
            f"Verifier response on page {page_number} has an unknown category: "
            f"{category!r}"
        )
    response_model = payload.get("model")
    if not isinstance(response_model, str) or not response_model:
        raise RuntimeError(f"Verifier response on page {page_number} has no model id")
    return VerifierAnswer(
        category=category,
        response_model=response_model,
        usage=parse_usage(payload, page_number),
    )


class VerifierClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float,
        max_tokens: int,
        seed: int,
        request: RequestJson = request_json,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.seed = seed
        self._request = request

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}{CHAT_COMPLETIONS_PATH}"

    def build_payload(self, state: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": state},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": RESPONSE_SCHEMA_NAME,
                    "strict": True,
                    "schema": CATEGORY_RESPONSE_SCHEMA,
                },
            },
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "stream": False,
            "reasoning_effort": REASONING_EFFORT_DISABLED,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def verify_page(self, state: str, page_number: int) -> VerifierAnswer:
        response = self._request(
            self.endpoint, self.build_payload(state), self.timeout
        )
        return parse_verifier_response(response, page_number)


def run_verification(
    categories_path: Path,
    client: VerifierClient,
    ocr_path: Path | None = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
    max_state_chars: int = DEFAULT_MAX_STATE_CHARS,
) -> VerificationReport:
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError(
            f"Confidence threshold must be within [0, 1], got: {confidence_threshold}"
        )
    if not 0.0 <= margin_threshold <= 1.0:
        raise ValueError(
            f"Margin threshold must be within [0, 1], got: {margin_threshold}"
        )
    if max_state_chars <= 0:
        raise ValueError(f"Max state chars must be positive, got: {max_state_chars}")
    category_report = load_categories_report(categories_path)
    resolved_ocr_path = (
        ocr_path
        if ocr_path is not None
        else resolve_ocr_path(categories_path, category_report.ocr_json_path)
    )
    ocr_document = load_ocr_document(resolved_ocr_path)
    ocr_by_number = {page.page_number: page for page in ocr_document.pages}
    expected_numbers = [record.page_number for record in category_report.records]
    provided_numbers = [page.page_number for page in ocr_document.pages]
    if expected_numbers != provided_numbers:
        missing = sorted(set(provided_numbers) - set(expected_numbers))
        extra = sorted(set(expected_numbers) - set(provided_numbers))
        raise ValueError(
            "Categories JSON pages do not match the OCR pages "
            f"(missing={missing[:5]} extra={extra[:5]})"
        )
    for record in category_report.records:
        expected_ref = make_page_ref(ocr_document.source, record.page_number)
        if record.page_ref != expected_ref:
            raise ValueError(
                f"Categories JSON page {record.page_number} has page_ref "
                f"{record.page_ref!r}, expected {expected_ref!r} for "
                f"{resolved_ocr_path}"
            )
    log_message(
        f"[verify] categories={categories_path} ocr={resolved_ocr_path} "
        f"pages={len(category_report.records)} model={client.model} "
        f"base_url={client.base_url} confidence_threshold={confidence_threshold} "
        f"margin_threshold={margin_threshold} input={INPUT_MODE}"
    )
    started = time.perf_counter()
    pages: list[dict[str, Any]] = []
    triggered_pages: list[int] = []
    accepted_count = 0
    mutual_other_count = 0
    disagreement_count = 0
    verifier_failed_count = 0
    needs_review_before = 0
    needs_review_after = 0
    verify_seconds = 0.0
    requests = 0
    prompt_tokens = 0
    completion_tokens = 0
    response_model: str | None = None
    for record in category_report.records:
        ocr_page = ocr_by_number[record.page_number]
        decision = evaluate_trigger(
            record, confidence_threshold, margin_threshold
        )
        verifier_category: str | None = None
        error: str | None = None
        usage: VerifierUsage | None = None
        duration = 0.0
        if not decision.triggered:
            resolution = RESOLUTION_NOT_TRIGGERED
        else:
            triggered_pages.append(record.page_number)
            validate_page_text(ocr_page, max_state_chars)
            margin_text = (
                f"{decision.margin:.{MARGIN_PRECISION}f}"
                if decision.margin is not None
                else "-"
            )
            log_message(
                f"[verify] page {record.page_number} triggered "
                f"reasons={', '.join(decision.reasons)} margin={margin_text}"
            )
            verify_started = time.perf_counter()
            requests += 1
            try:
                answer = client.verify_page(ocr_page.text, record.page_number)
            except Exception as exc:
                resolution = RESOLUTION_VERIFIER_FAILED
                error = f"{type(exc).__name__}: {exc}"
                log_message(
                    f"[verify] page {record.page_number} failed error={error}"
                )
            else:
                verifier_category = answer.category
                usage = answer.usage
                prompt_tokens += answer.usage["prompt_tokens"]
                completion_tokens += answer.usage["completion_tokens"]
                if response_model is None:
                    response_model = answer.response_model
                resolution = resolve_decision(record.category, answer.category)
                log_message(
                    f"[verify] page {record.page_number} done "
                    f"verifier_category={answer.category} resolution={resolution}"
                )
            duration = time.perf_counter() - verify_started
            verify_seconds += duration
            if resolution == RESOLUTION_ACCEPTED:
                accepted_count += 1
            elif resolution == RESOLUTION_MUTUAL_OTHER:
                mutual_other_count += 1
            elif resolution == RESOLUTION_DISAGREEMENT:
                disagreement_count += 1
            else:
                verifier_failed_count += 1
        needs_review, review_reason = review_outcome(record, resolution)
        if record.needs_review:
            needs_review_before += 1
        if needs_review:
            needs_review_after += 1
        verification: PageVerification = {
            "triggered": decision.triggered,
            "trigger_reasons": list(decision.reasons),
            "margin": (
                round(decision.margin, MARGIN_PRECISION)
                if decision.margin is not None
                else None
            ),
            "verifier_category": verifier_category,
            "resolution": resolution,
            "error": error,
            "verifier_seconds": round(duration, SECONDS_PRECISION),
            "usage": usage,
        }
        output_page = dict(record.raw)
        output_page["needs_review"] = needs_review
        output_page["review_reason"] = review_reason
        output_page["verification"] = verification
        pages.append(output_page)
    total_seconds = time.perf_counter() - started
    summary: VerificationSummary = {
        "page_count": len(pages),
        "triggered_page_count": len(triggered_pages),
        "accepted_page_count": accepted_count,
        "mutual_other_page_count": mutual_other_count,
        "disagreement_page_count": disagreement_count,
        "verifier_failed_page_count": verifier_failed_count,
        "needs_review_before": needs_review_before,
        "needs_review_after": needs_review_after,
        "triggered_pages": triggered_pages,
    }
    report: VerificationReport = {
        "schema_version": SCHEMA_VERSION,
        "input_files": {
            "categories_json": str(categories_path),
            "ocr_json": str(resolved_ocr_path),
        },
        "model": {
            "id": client.model,
            "base_url": client.base_url,
            "prompt_version": PROMPT_VERSION,
            "response_model": response_model,
        },
        "categories": dict(category_report.categories),
        "parameters": {
            "input_mode": INPUT_MODE,
            "confidence_threshold": confidence_threshold,
            "margin_threshold": margin_threshold,
            "max_state_chars": max_state_chars,
            "timeout_seconds": client.timeout,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "seed": client.seed,
            "max_tokens": client.max_tokens,
            "reasoning_effort": REASONING_EFFORT_DISABLED,
        },
        "summary": summary,
        "pages": pages,
        "timings": {
            "total_seconds": round(total_seconds, SECONDS_PRECISION),
            "verify_seconds": round(verify_seconds, SECONDS_PRECISION),
            "requests": requests,
            "avg_verify_seconds_per_request": (
                round(verify_seconds / requests, SECONDS_PRECISION)
                if requests
                else None
            ),
        },
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "requests": requests,
        },
    }
    log_message(
        f"[verify] total duration={total_seconds:.2f}s pages={len(pages)} "
        f"triggered={len(triggered_pages)} accepted={accepted_count} "
        f"mutual_other={mutual_other_count} disagreement={disagreement_count} "
        f"failed={verifier_failed_count}"
    )
    log_message(
        f"[verify] needs_review before={needs_review_before} "
        f"after={needs_review_after}"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify low-confidence or low-margin page categories from "
            "page_classifier.py with a local LM Studio model over the OCR page "
            "text, and report agreement, disagreement, and review decisions."
        )
    )
    parser.add_argument(
        "categories_json",
        type=Path,
        help="Categories JSON produced by page_classifier.py",
    )
    parser.add_argument(
        "--ocr",
        type=Path,
        default=None,
        help=(
            "OCR JSON produced by ocr_pdf.py (default: the OCR path recorded in "
            "the Categories JSON)"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "JSON output file (default: categories path with "
            f"{DEFAULT_OUTPUT_SUFFIX} suffix)"
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
        help=f"LM Studio OpenAI-compatible base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LM Studio model id (default: {DEFAULT_MODEL})",
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
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help=(
            "Pages below this classifier confidence are verified "
            f"(default: {DEFAULT_CONFIDENCE_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=DEFAULT_MARGIN_THRESHOLD,
        help=(
            "Pages whose top two probabilities differ by less than this margin "
            f"are verified (default: {DEFAULT_MARGIN_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--max-state-chars",
        type=int,
        default=DEFAULT_MAX_STATE_CHARS,
        help=(
            "Fail fast when a verified page text exceeds this many characters "
            f"(default: {DEFAULT_MAX_STATE_CHARS})"
        ),
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
        if args.max_tokens <= 0:
            raise ValueError(f"--max-tokens must be positive, got: {args.max_tokens}")
        if not 0.0 <= args.confidence_threshold <= 1.0:
            raise ValueError(
                f"--confidence-threshold must be within [0, 1], got: "
                f"{args.confidence_threshold}"
            )
        if not 0.0 <= args.margin_threshold <= 1.0:
            raise ValueError(
                f"--margin-threshold must be within [0, 1], got: "
                f"{args.margin_threshold}"
            )
        if args.max_state_chars <= 0:
            raise ValueError(
                f"--max-state-chars must be positive, got: {args.max_state_chars}"
            )
        client = VerifierClient(
            base_url=args.base_url,
            model=args.model,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            seed=args.seed,
            request=request_json,
        )
        report = run_verification(
            args.categories_json,
            client,
            ocr_path=args.ocr,
            confidence_threshold=args.confidence_threshold,
            margin_threshold=args.margin_threshold,
            max_state_chars=args.max_state_chars,
        )
    except Exception as exc:
        log_message(f"ERROR: {type(exc).__name__}: {exc}")
        return 1
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.stdout:
        print(payload)
        return 0
    out_path = (
        args.out
        if args.out is not None
        else default_output_path(args.categories_json)
    )
    try:
        out_path.write_text(payload, encoding=STDIO_ENCODING)
    except OSError as exc:
        log_message(f"ERROR: failed to write JSON file ({out_path}): {exc}")
        return 1
    log_message(f"[verify] JSON written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
