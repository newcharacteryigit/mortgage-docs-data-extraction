from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, TypedDict

import numpy as np
from numpy.typing import NDArray

DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
MODEL_ID_HINT = "jina-embeddings-v5-text-small-text-matching"
DOCUMENT_PREFIX = "Document: "
DEFAULT_SIMILARITY_THRESHOLD = 0.78
DEFAULT_BATCH_SIZE = 8
DEFAULT_TIMEOUT_SECONDS = 120.0
EMBEDDINGS_PATH = "/embeddings"
MODELS_PATH = "/models"
JSON_SUFFIX = ".json"
DEFAULT_OUTPUT_SUFFIX = ".matches.json"
SCHEMA_VERSION = "1.1"
TEXT_NORMALIZATION = "whitespace_collapsed"
SCORE_PRECISION = 6
STDIO_ENCODING = "utf-8"
WHITESPACE_PATTERN = re.compile(r"\s+")
LMSTUDIO_DOMAIN_HINT = (
    "LM Studio does not expose this model as an embedding model; in LM Studio open "
    "My Models, select the model, set 'Override Domain Type' to 'Text Embedding', "
    "then reload the model"
)
EMBEDDING_DOMAIN_ERROR_MARKERS = ("No models loaded", "not embedding", "not an embedding")

RequestJson = Callable[[str, dict[str, Any] | None, float], dict[str, Any]]


@dataclass(frozen=True)
class PageRecord:
    page_ref: str
    source: str
    page_number: int
    text: str


class PageReference(TypedDict):
    page_ref: str
    source: str
    page_number: int


class GroupMember(PageReference):
    similarity_to_previous_page: float | None


class MatchGroup(TypedDict):
    group_id: int
    size: int
    page_refs: list[str]
    representative: str
    members: list[GroupMember]
    min_edge_similarity: float
    mean_edge_similarity: float


class SkippedPage(PageReference):
    reason: str


class ModelInfo(TypedDict):
    id: str
    base_url: str
    document_prefix: str
    embedding_dimension: int | None


class MatchParameters(TypedDict):
    similarity_threshold: float
    batch_size: int
    text_normalization: str


class MatchSummary(TypedDict):
    page_count: int
    embedded_page_count: int
    skipped_page_count: int
    duplicate_group_count: int
    duplicate_page_count: int
    unique_page_count: int


class MatchTimings(TypedDict):
    total_seconds: float
    embedding_seconds: float
    matching_seconds: float
    embedding_requests: int


class MatchReport(TypedDict):
    schema_version: str
    input_files: list[str]
    model: ModelInfo
    parameters: MatchParameters
    summary: MatchSummary
    groups: list[MatchGroup]
    unique_pages: list[PageReference]
    skipped_pages: list[SkippedPage]
    timings: MatchTimings


class EmbeddingOutcome(NamedTuple):
    pages: list[PageRecord]
    vectors: NDArray[np.float64]
    skipped: list[tuple[PageRecord, str]]
    duration_seconds: float
    request_count: int


class PageGroup(NamedTuple):
    member_indices: tuple[int, ...]
    edge_scores: tuple[float | None, ...]


def log_message(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding=STDIO_ENCODING, errors="replace")


def default_output_path(input_path: Path) -> Path:
    return input_path.parent / f"{input_path.stem}{DEFAULT_OUTPUT_SUFFIX}"


def normalize_text(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def make_page_ref(source: str, page_number: int) -> str:
    return f"{source}#page-{page_number}"


def load_ocr_document(path: Path) -> list[PageRecord]:
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
    records: list[PageRecord] = []
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
            PageRecord(
                page_ref=make_page_ref(source, page_number),
                source=source,
                page_number=page_number,
                text=text,
            )
        )
    return records


def collect_page_records(paths: Sequence[Path]) -> list[PageRecord]:
    records: list[PageRecord] = []
    origins: dict[str, Path] = {}
    for path in paths:
        for record in load_ocr_document(path):
            previous = origins.get(record.page_ref)
            if previous is not None:
                raise ValueError(
                    f"Duplicate page reference {record.page_ref!r} in {previous} and {path}"
                )
            origins[record.page_ref] = path
            records.append(record)
    return records


def chunked(items: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def http_error_message(status: int, url: str, detail: str) -> str:
    message = f"HTTP {status} from {url}: {detail}"
    if any(marker in detail for marker in EMBEDDING_DOMAIN_ERROR_MARKERS):
        message = f"{message} Hint: {LMSTUDIO_DOMAIN_HINT}"
    return message


def request_json(
    url: str, payload: dict[str, Any] | None, timeout: float
) -> dict[str, Any]:
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode(STDIO_ENCODING)
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode(STDIO_ENCODING)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(STDIO_ENCODING, errors="replace")
        raise RuntimeError(http_error_message(exc.code, url, detail)) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach LM Studio at {url}: {exc.reason}") from exc
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response from {url}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Expected a JSON object from {url}")
    return parsed


class EmbeddingClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        prefix: str,
        batch_size: int,
        timeout: float,
        request: RequestJson = request_json,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.prefix = prefix
        self.batch_size = batch_size
        self.timeout = timeout
        self._request = request

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in chunked(texts, self.batch_size):
            payload = {
                "model": self.model,
                "input": [f"{self.prefix}{text}" for text in batch],
            }
            response = self._request(
                f"{self.base_url}{EMBEDDINGS_PATH}", payload, self.timeout
            )
            vectors.extend(parse_embeddings_response(response, len(batch)))
        return vectors


def extract_model_ids(payload: dict[str, Any]) -> list[str]:
    data = payload.get("data")
    if not isinstance(data, list):
        raise RuntimeError("LM Studio /models response has no 'data' list")
    model_ids: list[str] = []
    for entry in data:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise RuntimeError("LM Studio /models response contains an entry without an 'id'")
        model_ids.append(entry["id"])
    return model_ids


def list_models(base_url: str, timeout: float, request: RequestJson) -> list[str]:
    payload = request(f"{base_url.rstrip('/')}{MODELS_PATH}", None, timeout)
    return extract_model_ids(payload)


def select_model_id(model_ids: Sequence[str], hint: str) -> str:
    if not model_ids:
        raise ValueError(
            "LM Studio reported no models; load the embedding model and retry"
        )
    if hint in model_ids:
        return hint
    matches = sorted(model_id for model_id in model_ids if hint.lower() in model_id.lower())
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Multiple LM Studio models match {hint!r}: {matches}; choose one with --model"
        )
    raise ValueError(
        f"No LM Studio model matches {hint!r}; available models: {sorted(model_ids)}; "
        "choose one with --model"
    )


def parse_embeddings_response(
    payload: dict[str, Any], expected_count: int
) -> list[list[float]]:
    data = payload.get("data")
    if not isinstance(data, list):
        raise RuntimeError("Embedding response has no 'data' list")
    if len(data) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} embeddings, received {len(data)}"
        )
    ordered: list[list[float] | None] = [None] * expected_count
    for entry in data:
        if not isinstance(entry, dict) or isinstance(entry.get("index"), bool):
            raise RuntimeError("Embedding response entry is malformed")
        index = entry.get("index")
        if not isinstance(index, int) or not 0 <= index < expected_count:
            raise RuntimeError(f"Embedding response has an invalid index: {index!r}")
        if ordered[index] is not None:
            raise RuntimeError(f"Embedding response has a duplicate index: {index}")
        embedding = entry.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError(f"Embedding response entry {index} has no embedding")
        vector: list[float] = []
        for value in embedding:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RuntimeError(
                    f"Embedding response entry {index} contains a non-numeric value"
                )
            vector.append(float(value))
        ordered[index] = vector
    return [vector for vector in ordered if vector is not None]


def embed_pages(client: EmbeddingClient, pages: Sequence[PageRecord]) -> EmbeddingOutcome:
    embedded: list[PageRecord] = []
    skipped: list[tuple[PageRecord, str]] = []
    texts: list[str] = []
    for page in pages:
        normalized = normalize_text(page.text)
        if not normalized:
            skipped.append((page, "empty text"))
            continue
        embedded.append(page)
        texts.append(normalized)
    started = time.perf_counter()
    vectors = client.embed_texts(texts) if texts else []
    duration = time.perf_counter() - started
    if vectors:
        matrix = np.asarray(vectors, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != len(embedded):
            raise RuntimeError(
                f"Embedding count mismatch: expected {len(embedded)}, got {matrix.shape[0]}"
            )
    else:
        matrix = np.empty((0, 0), dtype=np.float64)
    request_count = math.ceil(len(texts) / client.batch_size) if texts else 0
    return EmbeddingOutcome(embedded, matrix, skipped, duration, request_count)


def normalize_vectors(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    if vectors.size == 0:
        return np.empty((0, 0), dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("Cannot compare an embedding with zero norm")
    return vectors / norms


def cosine_similarity(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        raise ValueError("Cannot compare an embedding with zero norm")
    score = float(np.dot(left, right)) / denominator
    return max(-1.0, min(1.0, score))


def group_adjacent_pages(
    pages: Sequence[PageRecord],
    vectors: NDArray[np.float64],
    threshold: float,
) -> list[PageGroup]:
    if not -1.0 <= threshold <= 1.0:
        raise ValueError(f"Similarity threshold must be within [-1, 1], got: {threshold}")
    if vectors.shape[0] != len(pages):
        raise ValueError(
            f"Vector count mismatch: {vectors.shape[0]} vectors for {len(pages)} pages"
        )
    normalized = normalize_vectors(vectors)
    groups: list[PageGroup] = []
    member_indices: list[int] = []
    edge_scores: list[float | None] = []
    previous_index: int | None = None
    for index, page in enumerate(pages):
        score: float | None = None
        joins_group = False
        if previous_index is not None:
            previous_page = pages[previous_index]
            is_consecutive = (
                page.source == previous_page.source
                and page.page_number == previous_page.page_number + 1
            )
            if is_consecutive:
                score = cosine_similarity(normalized[previous_index], normalized[index])
                joins_group = score >= threshold
        if joins_group:
            member_indices.append(index)
            edge_scores.append(score)
        else:
            if member_indices:
                groups.append(PageGroup(tuple(member_indices), tuple(edge_scores)))
            member_indices = [index]
            edge_scores = [None]
        previous_index = index
    if member_indices:
        groups.append(PageGroup(tuple(member_indices), tuple(edge_scores)))
    return groups


def round_score(value: float) -> float:
    return round(float(value), SCORE_PRECISION)


def build_group_payload(
    group: PageGroup, pages: Sequence[PageRecord], group_id: int
) -> MatchGroup:
    members: list[GroupMember] = []
    for index, score in zip(group.member_indices, group.edge_scores):
        page = pages[index]
        members.append(
            {
                "page_ref": page.page_ref,
                "source": page.source,
                "page_number": page.page_number,
                "similarity_to_previous_page": round_score(score)
                if score is not None
                else None,
            }
        )
    scores = [score for score in group.edge_scores if score is not None]
    return {
        "group_id": group_id,
        "size": len(group.member_indices),
        "page_refs": [pages[index].page_ref for index in group.member_indices],
        "representative": pages[group.member_indices[0]].page_ref,
        "members": members,
        "min_edge_similarity": round_score(min(scores)) if scores else 1.0,
        "mean_edge_similarity": round_score(float(np.mean(scores))) if scores else 1.0,
    }


def match_pages(
    records: Sequence[PageRecord],
    client: EmbeddingClient,
    threshold: float,
) -> tuple[MatchReport, EmbeddingOutcome]:
    if not -1.0 <= threshold <= 1.0:
        raise ValueError(f"Similarity threshold must be within [-1, 1], got: {threshold}")
    started = time.perf_counter()
    outcome = embed_pages(client, records)
    matching_started = time.perf_counter()
    groups = group_adjacent_pages(outcome.pages, outcome.vectors, threshold)
    matching_duration = time.perf_counter() - matching_started
    duplicate_groups = [group for group in groups if len(group.member_indices) > 1]
    unique_groups = [group for group in groups if len(group.member_indices) == 1]
    total_duration = time.perf_counter() - started
    report: MatchReport = {
        "schema_version": SCHEMA_VERSION,
        "input_files": [],
        "model": {
            "id": client.model,
            "base_url": client.base_url,
            "document_prefix": client.prefix,
            "embedding_dimension": int(outcome.vectors.shape[1])
            if outcome.vectors.size
            else None,
        },
        "parameters": {
            "similarity_threshold": threshold,
            "batch_size": client.batch_size,
            "text_normalization": TEXT_NORMALIZATION,
        },
        "summary": {
            "page_count": len(records),
            "embedded_page_count": len(outcome.pages),
            "skipped_page_count": len(outcome.skipped),
            "duplicate_group_count": len(duplicate_groups),
            "duplicate_page_count": sum(
                len(group.member_indices) for group in duplicate_groups
            ),
            "unique_page_count": len(unique_groups),
        },
        "groups": [
            build_group_payload(group, outcome.pages, group_id)
            for group_id, group in enumerate(duplicate_groups, start=1)
        ],
        "unique_pages": [
            {
                "page_ref": outcome.pages[group.member_indices[0]].page_ref,
                "source": outcome.pages[group.member_indices[0]].source,
                "page_number": outcome.pages[group.member_indices[0]].page_number,
            }
            for group in unique_groups
        ],
        "skipped_pages": [
            {
                "page_ref": page.page_ref,
                "source": page.source,
                "page_number": page.page_number,
                "reason": reason,
            }
            for page, reason in outcome.skipped
        ],
        "timings": {
            "total_seconds": round(total_duration, 3),
            "embedding_seconds": round(outcome.duration_seconds, 3),
            "matching_seconds": round(matching_duration, 3),
            "embedding_requests": outcome.request_count,
        },
    }
    return report, outcome


def run_matching(
    input_paths: Sequence[Path],
    client: EmbeddingClient,
    threshold: float,
) -> tuple[MatchReport, EmbeddingOutcome]:
    records = collect_page_records(input_paths)
    log_message(
        f"[match] files={len(input_paths)} pages={len(records)} model={client.model} "
        f"base_url={client.base_url} threshold={threshold}"
    )
    report, outcome = match_pages(records, client, threshold)
    log_message(
        f"[match] embedded {len(outcome.pages)}/{len(records)} pages in "
        f"{outcome.request_count} requests duration={report['timings']['embedding_seconds']:.2f}s "
        f"skipped={len(outcome.skipped)}"
    )
    report["input_files"] = [str(path) for path in input_paths]
    return report, outcome


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Group duplicate adjacent pages (i and i+1) within each OCR JSON document "
            "using local embeddings served by LM Studio. Pages from different input "
            "documents are never compared."
        )
    )
    parser.add_argument(
        "ocr_json",
        type=Path,
        nargs="+",
        help="OCR JSON file(s) produced by ocr_pdf.py",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"JSON output file (default: first input with {DEFAULT_OUTPUT_SUFFIX} suffix)",
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
        default=None,
        help=(
            "LM Studio model id; when omitted the id is resolved via GET /models "
            f"using the hint {MODEL_ID_HINT!r}"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help=f"Cosine similarity threshold (default: {DEFAULT_SIMILARITY_THRESHOLD})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Pages per embedding request (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"HTTP timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--prefix",
        default=DOCUMENT_PREFIX,
        help=(
            "Prefix applied to every page before embedding; the text-matching adapter "
            f"was trained with {DOCUMENT_PREFIX!r}"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_stdio()
    args = build_parser().parse_args(argv)
    try:
        if not args.base_url:
            raise ValueError("--base-url must not be empty")
        if args.batch_size < 1:
            raise ValueError(f"--batch-size must be positive, got: {args.batch_size}")
        if args.timeout <= 0:
            raise ValueError(f"--timeout must be positive, got: {args.timeout}")
        if not -1.0 <= args.threshold <= 1.0:
            raise ValueError(
                f"--threshold must be within [-1, 1], got: {args.threshold}"
            )
        model_id = args.model
        if model_id is None:
            model_id = select_model_id(
                list_models(args.base_url, args.timeout, request_json), MODEL_ID_HINT
            )
        client = EmbeddingClient(
            base_url=args.base_url,
            model=model_id,
            prefix=args.prefix,
            batch_size=args.batch_size,
            timeout=args.timeout,
            request=request_json,
        )
        report, _ = run_matching(args.ocr_json, client, args.threshold)
    except Exception as exc:
        log_message(f"ERROR: {type(exc).__name__}: {exc}")
        return 1
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.stdout:
        print(payload)
        return 0
    out_path = args.out if args.out is not None else default_output_path(args.ocr_json[0])
    try:
        out_path.write_text(payload, encoding="utf-8")
    except OSError as exc:
        log_message(f"ERROR: failed to write JSON file ({out_path}): {exc}")
        return 1
    summary = report["summary"]
    log_message(
        f"[match] duplicate groups={summary['duplicate_group_count']} "
        f"duplicate pages={summary['duplicate_page_count']} "
        f"unique pages={summary['unique_page_count']} skipped={summary['skipped_page_count']}"
    )
    log_message(f"[match] total duration={report['timings']['total_seconds']:.2f}s")
    log_message(f"[match] JSON written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
