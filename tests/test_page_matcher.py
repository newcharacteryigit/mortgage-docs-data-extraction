from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from page_matcher import (
    EmbeddingClient,
    PageRecord,
    collect_page_records,
    cosine_similarity,
    default_output_path,
    embed_pages,
    group_adjacent_pages,
    http_error_message,
    load_ocr_document,
    normalize_text,
    normalize_vectors,
    parse_embeddings_response,
    run_matching,
    select_model_id,
)


class FakeTransport:
    def __init__(
        self,
        vectors: dict[str, list[float]],
        model_ids: list[str] | None = None,
    ) -> None:
        self.vectors = vectors
        self.model_ids = model_ids or []
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, url: str, payload: dict[str, Any] | None, timeout: float
    ) -> dict[str, Any]:
        self.calls.append({"url": url, "payload": payload, "timeout": timeout})
        if url.endswith("/models"):
            return {"data": [{"id": model_id} for model_id in self.model_ids]}
        if url.endswith("/embeddings"):
            assert payload is not None
            return {
                "data": [
                    {"index": index, "embedding": self.vectors[text]}
                    for index, text in enumerate(payload["input"])
                ]
            }
        raise AssertionError(f"Unexpected URL: {url}")


def make_client(
    transport: FakeTransport,
    batch_size: int = 8,
    prefix: str = "Document: ",
) -> EmbeddingClient:
    return EmbeddingClient(
        base_url="http://127.0.0.1:1234/v1",
        model="test-model",
        prefix=prefix,
        batch_size=batch_size,
        timeout=30.0,
        request=transport,
    )


def write_ocr_json(path: Path, source: str, pages: list[tuple[int, str]]) -> Path:
    payload = {
        "source": source,
        "engine": "paddleocr",
        "pages": [
            {
                "page_number": number,
                "text": text,
                "line_count": 1,
                "mean_score": 0.9,
                "duration_seconds": 0.1,
            }
            for number, text in pages
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text("LOAN #: 20414784\n\nPage 1  of 2\r\n") == (
        "LOAN #: 20414784 Page 1 of 2"
    )
    assert normalize_text("   \n\t ") == ""


def test_load_ocr_document_builds_page_records(tmp_path: Path) -> None:
    path = write_ocr_json(tmp_path / "a.json", "a.pdf", [(1, "first"), (2, "second")])
    records = load_ocr_document(path)
    assert [record.page_ref for record in records] == [
        "a.pdf#page-1",
        "a.pdf#page-2",
    ]
    assert records[1].source == "a.pdf"
    assert records[1].page_number == 2
    assert records[1].text == "second"


def test_load_ocr_document_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_ocr_document(tmp_path / "missing.json")


def test_load_ocr_document_non_json_suffix_raises(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match=r"\.json suffix"):
        load_ocr_document(path)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("not json", "Failed to parse JSON"),
        ("[]", "root must be an object"),
        (json.dumps({"pages": [{"page_number": 1, "text": "x"}]}), "source"),
        (json.dumps({"source": "a.pdf"}), "no 'pages'"),
        (json.dumps({"source": "a.pdf", "pages": []}), "no 'pages'"),
        (json.dumps({"source": "a.pdf", "pages": ["x"]}), "not an object"),
        (
            json.dumps({"source": "a.pdf", "pages": [{"page_number": "1", "text": "x"}]}),
            "non-integer page_number",
        ),
        (
            json.dumps({"source": "a.pdf", "pages": [{"page_number": 0, "text": "x"}]}),
            "non-positive page_number",
        ),
        (
            json.dumps({"source": "a.pdf", "pages": [{"page_number": 1, "text": 5}]}),
            "non-string text",
        ),
        (
            json.dumps(
                {
                    "source": "a.pdf",
                    "pages": [
                        {"page_number": 2, "text": "x"},
                        {"page_number": 1, "text": "y"},
                    ],
                }
            ),
            "non-increasing page_number",
        ),
    ],
)
def test_load_ocr_document_invalid_payloads_raise(
    tmp_path: Path, payload: str, message: str
) -> None:
    path = tmp_path / "broken.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_ocr_document(path)


def test_collect_page_records_rejects_duplicate_page_refs(tmp_path: Path) -> None:
    first = write_ocr_json(tmp_path / "first.json", "same.pdf", [(1, "alpha")])
    second = write_ocr_json(tmp_path / "second.json", "same.pdf", [(1, "alpha")])
    with pytest.raises(ValueError, match="Duplicate page reference"):
        collect_page_records([first, second])


def test_select_model_id_prefers_exact_match() -> None:
    model_ids = ["jinaai/model", "text-embedding-jinaai/model"]
    assert select_model_id(model_ids, "jinaai/model") == "jinaai/model"


def test_select_model_id_accepts_single_substring_match() -> None:
    model_ids = ["prism-ml/bonsai-27b", "text-embedding-jinaai/model"]
    assert select_model_id(model_ids, "jinaai/model") == "text-embedding-jinaai/model"


def test_select_model_id_rejects_ambiguous_and_missing() -> None:
    with pytest.raises(ValueError, match="Multiple"):
        select_model_id(["a/model", "b/model"], "model")
    with pytest.raises(ValueError, match="available models"):
        select_model_id(["only-chat"], "jinaai/model")
    with pytest.raises(ValueError, match="no models"):
        select_model_id([], "jinaai/model")


def test_parse_embeddings_response_orders_by_index() -> None:
    payload = {
        "data": [
            {"index": 1, "embedding": [0.0, 1.0]},
            {"index": 0, "embedding": [1.0, 0.0]},
        ]
    }
    assert parse_embeddings_response(payload, 2) == [[1.0, 0.0], [0.0, 1.0]]


@pytest.mark.parametrize(
    ("payload", "expected_count", "message"),
    [
        ({"data": [{"index": 0, "embedding": [1.0]}]}, 2, "Expected 2 embeddings"),
        ({"data": [{"index": 1, "embedding": [1.0]}]}, 1, "invalid index"),
        (
            {
                "data": [
                    {"index": 0, "embedding": [1.0]},
                    {"index": 0, "embedding": [1.0]},
                ]
            },
            2,
            "duplicate index",
        ),
        ({"data": [{"index": 0, "embedding": ["x"]}]}, 1, "non-numeric"),
        ({"data": [{"index": 0}]}, 1, "no embedding"),
        ({"data": "x"}, 1, "no 'data' list"),
    ],
)
def test_parse_embeddings_response_rejects_malformed(
    payload: dict[str, Any], expected_count: int, message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        parse_embeddings_response(payload, expected_count)


def test_cosine_similarity() -> None:
    vector = np.array([1.0, 0.0])
    assert cosine_similarity(vector, vector) == pytest.approx(1.0)
    assert cosine_similarity(vector, np.array([0.0, 1.0])) == pytest.approx(0.0)
    assert cosine_similarity(vector, np.array([-1.0, 0.0])) == pytest.approx(-1.0)


def test_cosine_similarity_rejects_zero_norm() -> None:
    with pytest.raises(ValueError, match="zero norm"):
        cosine_similarity(np.array([0.0, 0.0]), np.array([1.0, 0.0]))


def test_normalize_vectors_rejects_zero_norm() -> None:
    with pytest.raises(ValueError, match="zero norm"):
        normalize_vectors(np.array([[0.0, 0.0]]))


def make_pages(*entries: tuple[int, str], source: str = "doc.pdf") -> list[PageRecord]:
    return [
        PageRecord(
            page_ref=f"{source}#page-{number}",
            source=source,
            page_number=number,
            text=text,
        )
        for number, text in entries
    ]


def test_group_adjacent_pages_groups_only_matching_neighbors() -> None:
    pages = make_pages((1, "alpha"), (2, "beta"), (3, "beta"), (4, "gamma"))
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    groups = group_adjacent_pages(pages, vectors, 0.9)
    assert [group.member_indices for group in groups] == [(0,), (1, 2), (3,)]
    assert groups[1].edge_scores == (None, pytest.approx(1.0))
    assert groups[0].edge_scores == (None,)


def test_group_adjacent_pages_does_not_merge_non_adjacent_pages() -> None:
    pages = make_pages((1, "alpha"), (2, "beta"), (3, "alpha"))
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )
    groups = group_adjacent_pages(pages, vectors, 0.9)
    assert [group.member_indices for group in groups] == [(0,), (1,), (2,)]


def test_group_adjacent_pages_does_not_cross_sources() -> None:
    first = make_pages((1, "alpha"), source="first.pdf")
    second = make_pages((1, "alpha"), source="second.pdf")
    pages = sorted(first + second, key=lambda page: (page.source, page.page_number))
    vectors = np.array([[1.0, 0.0], [1.0, 0.0]])
    groups = group_adjacent_pages(pages, vectors, 0.9)
    assert [group.member_indices for group in groups] == [(0,), (1,)]


def test_group_adjacent_pages_skips_gaps_from_skipped_pages() -> None:
    pages = make_pages((1, "alpha"), (3, "alpha"))
    vectors = np.array([[1.0, 0.0], [1.0, 0.0]])
    groups = group_adjacent_pages(pages, vectors, 0.9)
    assert [group.member_indices for group in groups] == [(0,), (1,)]


def test_group_adjacent_pages_threshold_is_inclusive() -> None:
    pages = make_pages((1, "alpha"), (2, "alpha"))
    vectors = np.array([[1.0, 0.0], [0.9, 0.4358898943540674]])
    groups = group_adjacent_pages(pages, vectors, 0.9)
    assert [group.member_indices for group in groups] == [(0, 1)]


def test_group_adjacent_pages_rejects_invalid_threshold() -> None:
    pages = make_pages((1, "alpha"))
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        group_adjacent_pages(pages, np.array([[1.0, 0.0]]), 1.5)


def test_embed_pages_prefixes_text_and_skips_empty() -> None:
    transport = FakeTransport({"Document: alpha": [1.0, 0.0]})
    client = make_client(transport)
    records = [
        PageRecord("a#page-1", "a", 1, "alpha\n"),
        PageRecord("a#page-2", "a", 2, " \n "),
    ]
    outcome = embed_pages(client, records)
    assert [page.page_ref for page in outcome.pages] == ["a#page-1"]
    assert outcome.skipped == [(records[1], "empty text")]
    assert outcome.request_count == 1
    assert transport.calls[0]["payload"]["input"] == ["Document: alpha"]


def test_run_matching_groups_adjacent_duplicate_pages(tmp_path: Path) -> None:
    first = write_ocr_json(
        tmp_path / "first.json",
        "first.pdf",
        [(1, "alpha"), (2, "  alpha  "), (3, "beta"), (4, "")],
    )
    second = write_ocr_json(
        tmp_path / "second.json",
        "second.pdf",
        [(1, "alpha"), (2, "gamma")],
    )
    transport = FakeTransport(
        {
            "Document: alpha": [1.0, 0.0, 0.0],
            "Document: beta": [0.0, 1.0, 0.0],
            "Document: gamma": [0.0, 0.0, 1.0],
        }
    )
    client = make_client(transport, batch_size=2)

    report, _ = run_matching([first, second], client, 0.95)

    assert report["input_files"] == [str(first), str(second)]
    assert report["summary"] == {
        "page_count": 6,
        "embedded_page_count": 5,
        "skipped_page_count": 1,
        "duplicate_group_count": 1,
        "duplicate_page_count": 2,
        "unique_page_count": 3,
    }
    group = report["groups"][0]
    assert group["page_refs"] == ["first.pdf#page-1", "first.pdf#page-2"]
    assert group["representative"] == "first.pdf#page-1"
    assert group["size"] == 2
    assert group["members"][0]["similarity_to_previous_page"] is None
    assert group["members"][1]["similarity_to_previous_page"] == 1.0
    assert group["min_edge_similarity"] == 1.0
    assert group["mean_edge_similarity"] == 1.0
    assert [page["page_ref"] for page in report["unique_pages"]] == [
        "first.pdf#page-3",
        "second.pdf#page-1",
        "second.pdf#page-2",
    ]
    assert report["skipped_pages"] == [
        {
            "page_ref": "first.pdf#page-4",
            "source": "first.pdf",
            "page_number": 4,
            "reason": "empty text",
        }
    ]
    assert report["model"]["embedding_dimension"] == 3
    assert report["timings"]["embedding_requests"] == 3
    assert report["timings"]["total_seconds"] >= 0.0
    json.dumps(report, ensure_ascii=False)


def test_run_matching_does_not_compare_separate_documents(tmp_path: Path) -> None:
    first = write_ocr_json(tmp_path / "first.json", "first.pdf", [(1, "alpha")])
    second = write_ocr_json(tmp_path / "second.json", "second.pdf", [(1, "alpha")])
    transport = FakeTransport({"Document: alpha": [1.0, 0.0]})
    client = make_client(transport)

    report, _ = run_matching([first, second], client, 0.9)

    assert report["summary"]["duplicate_group_count"] == 0
    assert len(report["unique_pages"]) == 2


def test_run_matching_respects_threshold(tmp_path: Path) -> None:
    path = write_ocr_json(tmp_path / "doc.json", "doc.pdf", [(1, "alpha"), (2, "beta")])
    transport = FakeTransport(
        {
            "Document: alpha": [1.0, 0.0],
            "Document: beta": [0.9, 0.4358898943540674],
        }
    )
    client = make_client(transport)

    strict_report, _ = run_matching([path], client, 0.95)
    assert strict_report["summary"]["duplicate_group_count"] == 0
    assert len(strict_report["unique_pages"]) == 2

    loose_report, _ = run_matching([path], client, 0.85)
    assert loose_report["summary"]["duplicate_group_count"] == 1
    members = loose_report["groups"][0]["members"]
    assert members[1]["similarity_to_previous_page"] == pytest.approx(0.9, abs=1e-6)


def test_http_error_message_adds_domain_hint_for_misclassified_model() -> None:
    detail = '{"error":"No models loaded. Please load a model in the developer page."}'
    message = http_error_message(400, "http://127.0.0.1:1234/v1/embeddings", detail)
    assert "HTTP 400" in message
    assert "Override Domain Type" in message
    assert "Override Domain Type" not in http_error_message(500, "http://x", "boom")


def test_default_output_path(tmp_path: Path) -> None:
    assert default_output_path(tmp_path / "AREAL_LOAN.json") == (
        tmp_path / "AREAL_LOAN.matches.json"
    )
