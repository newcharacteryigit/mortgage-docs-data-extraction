from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from page_matcher import (
    CategorySignal,
    EdgeEvaluation,
    EmbeddingClient,
    PageRecord,
    ScoreWeights,
    category_score,
    collect_page_records,
    combine_scores,
    cosine_similarity,
    default_output_path,
    embed_pages,
    evaluate_edges,
    group_adjacent_pages,
    http_error_message,
    load_categories_json,
    load_documents,
    load_ocr_document,
    load_vlm_json,
    normalize_text,
    page_number_score,
    parse_embeddings_response,
    resolve_auxiliary_signals,
    run_matching,
    select_model_id,
)

WEIGHTS = ScoreWeights(embedding=0.60, category=0.25, page_number=0.15)


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


def write_categories_json(
    path: Path, pages: list[tuple[str, str, float, bool]]
) -> Path:
    payload = {
        "pages": [
            {
                "page_ref": page_ref,
                "category": category,
                "confidence": confidence,
                "needs_review": needs_review,
            }
            for page_ref, category, confidence, needs_review in pages
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_vlm_json(path: Path, pages: list[tuple[str, str | None]]) -> Path:
    payload = {
        "pages": [
            {"page_ref": page_ref, "extracted": {"page_number": number}}
            for page_ref, number in pages
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


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


def make_evaluation(combined: float, blocked: bool = False) -> EdgeEvaluation:
    return EdgeEvaluation(
        combined=combined,
        embedding=combined,
        category=None,
        page_number=None,
        available_weight=0.6,
        blocked=blocked,
    )


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


def test_load_categories_json_marks_other_and_review_as_unavailable(
    tmp_path: Path,
) -> None:
    path = write_categories_json(
        tmp_path / "categories.json",
        [
            ("doc.pdf#page-1", "title_rider", 1.0, False),
            ("doc.pdf#page-2", "other", 0.9, False),
            ("doc.pdf#page-3", "title_rider", 0.8, True),
        ],
    )
    signals = load_categories_json(path)
    assert signals["doc.pdf#page-1"] == CategorySignal("title_rider", 1.0)
    assert signals["doc.pdf#page-2"] is None
    assert signals["doc.pdf#page-3"] is None


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"pages": []}, "no 'pages'"),
        (
            {"pages": [{"category": "title_rider", "confidence": 1.0, "needs_review": False}]},
            "no 'page_ref'",
        ),
        (
            {
                "pages": [
                    {
                        "page_ref": "doc.pdf#page-1",
                        "category": "",
                        "confidence": 1.0,
                        "needs_review": False,
                    }
                ]
            },
            "no 'category'",
        ),
        (
            {
                "pages": [
                    {
                        "page_ref": "doc.pdf#page-1",
                        "category": "title_rider",
                        "confidence": "high",
                        "needs_review": False,
                    }
                ]
            },
            "non-numeric 'confidence'",
        ),
        (
            {
                "pages": [
                    {
                        "page_ref": "doc.pdf#page-1",
                        "category": "title_rider",
                        "confidence": 1.5,
                        "needs_review": False,
                    }
                ]
            },
            "outside \\[0, 1\\]",
        ),
        (
            {
                "pages": [
                    {
                        "page_ref": "doc.pdf#page-1",
                        "category": "title_rider",
                        "confidence": 1.0,
                        "needs_review": "yes",
                    }
                ]
            },
            "non-boolean 'needs_review'",
        ),
        (
            {
                "pages": [
                    {
                        "page_ref": "doc.pdf#page-1",
                        "category": "title_rider",
                        "confidence": 1.0,
                        "needs_review": False,
                    },
                    {
                        "page_ref": "doc.pdf#page-1",
                        "category": "lender_rate_note",
                        "confidence": 1.0,
                        "needs_review": False,
                    },
                ]
            },
            "duplicate page_ref",
        ),
    ],
)
def test_load_categories_json_rejects_invalid_payloads(
    tmp_path: Path, payload: dict[str, Any], message: str
) -> None:
    path = tmp_path / "categories.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_categories_json(path)


def test_load_vlm_json_parses_printed_page_numbers(tmp_path: Path) -> None:
    path = write_vlm_json(
        tmp_path / "vlm.json",
        [
            ("doc.pdf#page-1", "1"),
            ("doc.pdf#page-2", " 12 "),
            ("doc.pdf#page-3", None),
        ],
    )
    assert load_vlm_json(path) == {
        "doc.pdf#page-1": 1,
        "doc.pdf#page-2": 12,
        "doc.pdf#page-3": None,
    }


def test_load_vlm_json_rejects_non_numeric_page_number(tmp_path: Path) -> None:
    path = write_vlm_json(tmp_path / "vlm.json", [("doc.pdf#page-1", "Page 1")])
    with pytest.raises(ValueError, match="non-numeric 'page_number'"):
        load_vlm_json(path)


def test_load_vlm_json_rejects_missing_extracted_object(tmp_path: Path) -> None:
    path = tmp_path / "vlm.json"
    path.write_text(
        json.dumps({"pages": [{"page_ref": "doc.pdf#page-1"}]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="no 'extracted' object"):
        load_vlm_json(path)


def test_resolve_auxiliary_signals_discovers_sibling_files(tmp_path: Path) -> None:
    ocr = write_ocr_json(tmp_path / "doc.json", "doc.pdf", [(1, "alpha"), (2, "beta")])
    categories = write_categories_json(
        tmp_path / "doc.categories.json",
        [
            ("doc.pdf#page-1", "title_rider", 1.0, False),
            ("doc.pdf#page-2", "title_rider", 1.0, False),
        ],
    )
    vlm = write_vlm_json(
        tmp_path / "doc.vlm.json",
        [("doc.pdf#page-1", "1"), ("doc.pdf#page-2", "2")],
    )
    signals = resolve_auxiliary_signals(load_documents([ocr]), None, None)
    assert signals.categories_json == str(categories)
    assert signals.vlm_json == str(vlm)
    assert signals.categories["doc.pdf#page-1"] == CategorySignal("title_rider", 1.0)
    assert signals.page_numbers["doc.pdf#page-2"] == 2


def test_resolve_auxiliary_signals_without_files_returns_empty(tmp_path: Path) -> None:
    ocr = write_ocr_json(tmp_path / "doc.json", "doc.pdf", [(1, "alpha")])
    signals = resolve_auxiliary_signals(load_documents([ocr]), None, None)
    assert signals.categories == {}
    assert signals.page_numbers == {}
    assert signals.categories_json is None
    assert signals.vlm_json is None


def test_resolve_auxiliary_signals_rejects_missing_pages(tmp_path: Path) -> None:
    ocr = write_ocr_json(tmp_path / "doc.json", "doc.pdf", [(1, "alpha"), (2, "beta")])
    write_categories_json(
        tmp_path / "doc.categories.json",
        [("doc.pdf#page-1", "title_rider", 1.0, False)],
    )
    with pytest.raises(ValueError, match="do not match the OCR document"):
        resolve_auxiliary_signals(load_documents([ocr]), None, None)


def test_resolve_auxiliary_signals_allows_missing_empty_pages(tmp_path: Path) -> None:
    ocr = write_ocr_json(tmp_path / "doc.json", "doc.pdf", [(1, "alpha"), (2, "")])
    write_categories_json(
        tmp_path / "doc.categories.json",
        [("doc.pdf#page-1", "title_rider", 1.0, False)],
    )
    write_vlm_json(
        tmp_path / "doc.vlm.json",
        [("doc.pdf#page-1", "1"), ("doc.pdf#page-2", None)],
    )
    signals = resolve_auxiliary_signals(load_documents([ocr]), None, None)
    assert signals.categories != {}
    assert signals.page_numbers["doc.pdf#page-2"] is None


def test_resolve_auxiliary_signals_rejects_explicit_path_with_multiple_inputs(
    tmp_path: Path,
) -> None:
    first = write_ocr_json(tmp_path / "first.json", "first.pdf", [(1, "alpha")])
    second = write_ocr_json(tmp_path / "second.json", "second.pdf", [(1, "alpha")])
    documents = load_documents([first, second])
    with pytest.raises(ValueError, match="single OCR JSON"):
        resolve_auxiliary_signals(documents, tmp_path / "missing.json", None)
    with pytest.raises(ValueError, match="single OCR JSON"):
        resolve_auxiliary_signals(documents, None, tmp_path / "missing.json")


def test_resolve_auxiliary_signals_raises_for_missing_explicit_path(
    tmp_path: Path,
) -> None:
    ocr = write_ocr_json(tmp_path / "doc.json", "doc.pdf", [(1, "alpha")])
    with pytest.raises(FileNotFoundError, match="not found"):
        resolve_auxiliary_signals(load_documents([ocr]), tmp_path / "missing.json", None)


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


def test_category_score_same_category_uses_min_confidence() -> None:
    left = CategorySignal("title_rider", 1.0)
    right = CategorySignal("title_rider", 0.8)
    assert category_score(left, right) == pytest.approx(0.8)


def test_category_score_different_categories_is_zero() -> None:
    left = CategorySignal("title_rider", 1.0)
    right = CategorySignal("lender_rate_note", 1.0)
    assert category_score(left, right) == 0.0


def test_category_score_unavailable_is_none() -> None:
    signal = CategorySignal("title_rider", 1.0)
    assert category_score(None, signal) is None
    assert category_score(signal, None) is None


def test_page_number_score_rules() -> None:
    assert page_number_score(1, 2) == 1.0
    assert page_number_score(2, 2) == 0.5
    assert page_number_score(2, 4) is None
    assert page_number_score(4, 3) is None
    assert page_number_score(None, 2) is None
    assert page_number_score(2, None) is None


def test_combine_scores_with_all_signals() -> None:
    combined, available = combine_scores(0.8, 0.5, 1.0, WEIGHTS)
    assert available == pytest.approx(1.0)
    assert combined == pytest.approx(0.6 * 0.8 + 0.25 * 0.5 + 0.15 * 1.0)


def test_combine_scores_renormalizes_missing_signals() -> None:
    combined, available = combine_scores(0.8, None, None, WEIGHTS)
    assert available == pytest.approx(0.6)
    assert combined == pytest.approx(0.8)


def test_combine_scores_rejects_negative_and_zero_weights() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        combine_scores(0.8, None, None, ScoreWeights(-1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="must not all be zero"):
        combine_scores(0.8, None, None, ScoreWeights(0.0, 0.0, 0.0))


def test_evaluate_edges_only_connects_consecutive_same_source_pages() -> None:
    pages = make_pages((1, "alpha"), (3, "alpha"), (4, "alpha"))
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    evaluations = evaluate_edges(pages, vectors)
    assert evaluations[0] is None
    assert evaluations[1] is None
    assert evaluations[2] is not None


def test_evaluate_edges_does_not_cross_sources() -> None:
    pages = [
        PageRecord("first.pdf#page-1", "first.pdf", 1, "alpha"),
        PageRecord("second.pdf#page-2", "second.pdf", 2, "alpha"),
    ]
    vectors = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert evaluate_edges(pages, vectors) == [None, None]


def test_evaluate_edges_combines_signals() -> None:
    pages = make_pages((1, "alpha"), (2, "beta"))
    vectors = np.array([[1.0, 0.0], [0.8, 0.6]])
    categories = {
        "doc.pdf#page-1": CategorySignal("title_rider", 1.0),
        "doc.pdf#page-2": CategorySignal("title_rider", 0.8),
    }
    page_numbers = {"doc.pdf#page-1": 1, "doc.pdf#page-2": 2}
    evaluation = evaluate_edges(
        pages, vectors, categories, page_numbers, WEIGHTS, 0.5
    )[1]
    assert evaluation is not None
    assert evaluation.embedding == pytest.approx(0.8)
    assert evaluation.category == pytest.approx(0.8)
    assert evaluation.page_number == 1.0
    assert evaluation.available_weight == pytest.approx(1.0)
    assert evaluation.combined == pytest.approx(0.6 * 0.8 + 0.25 * 0.8 + 0.15)
    assert not evaluation.blocked


def test_evaluate_edges_blocks_below_min_embedding_similarity() -> None:
    pages = make_pages((1, "alpha"), (2, "beta"))
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    categories = {
        "doc.pdf#page-1": CategorySignal("title_rider", 1.0),
        "doc.pdf#page-2": CategorySignal("title_rider", 1.0),
    }
    page_numbers = {"doc.pdf#page-1": 1, "doc.pdf#page-2": 2}
    evaluation = evaluate_edges(
        pages, vectors, categories, page_numbers, WEIGHTS, 0.6
    )[1]
    assert evaluation is not None
    assert evaluation.embedding == 0.0
    assert evaluation.combined == pytest.approx(0.25 + 0.15)
    assert evaluation.blocked


def test_evaluate_edges_clamps_negative_cosine_to_zero() -> None:
    pages = make_pages((1, "alpha"), (2, "beta"))
    vectors = np.array([[1.0, 0.0], [-1.0, 0.0]])
    evaluation = evaluate_edges(pages, vectors)[1]
    assert evaluation is not None
    assert evaluation.embedding == 0.0


def test_group_adjacent_pages_uses_combined_scores() -> None:
    evaluations = [
        None,
        make_evaluation(0.9),
        make_evaluation(0.5),
        make_evaluation(0.95),
        None,
    ]
    groups = group_adjacent_pages(evaluations, 0.8)
    assert [group.member_indices for group in groups] == [(0, 1), (2, 3), (4,)]
    assert groups[0].edge_scores[0] is None
    assert groups[0].edge_scores[1] is not None
    assert groups[0].edge_scores[1].combined == pytest.approx(0.9)
    assert groups[1].edge_scores[0] is None
    assert groups[1].edge_scores[1] is not None
    assert groups[1].edge_scores[1].combined == pytest.approx(0.95)
    assert groups[2].edge_scores == (None,)


def test_group_adjacent_pages_ignores_blocked_edges() -> None:
    evaluations = [None, make_evaluation(0.99, blocked=True)]
    groups = group_adjacent_pages(evaluations, 0.8)
    assert [group.member_indices for group in groups] == [(0,), (1,)]


def test_group_adjacent_pages_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        group_adjacent_pages([None], 1.5)


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

    assert report["schema_version"] == "1.2"
    assert report["scoring_version"] == "1.0"
    assert report["input_files"] == [str(first), str(second)]
    assert report["auxiliary"] == {"categories_json": None, "vlm_json": None}
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
    assert group["members"][0]["edge"] is None
    edge = group["members"][1]["edge"]
    assert edge is not None
    assert edge["combined_similarity"] == 1.0
    assert edge["available_weight"] == pytest.approx(0.6)
    assert edge["signals"] == {
        "embedding_similarity": 1.0,
        "category_match": None,
        "page_number_sequence": None,
    }
    assert group["min_combined_similarity"] == 1.0
    assert group["mean_combined_similarity"] == 1.0
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


def test_run_matching_uses_auxiliary_signals(tmp_path: Path) -> None:
    ocr = write_ocr_json(
        tmp_path / "doc.json",
        "doc.pdf",
        [
            (1, "alpha"),
            (2, "alpha"),
            (3, "beta"),
            (4, "beta"),
            (5, "beta"),
            (6, "gamma"),
            (7, "gamma"),
        ],
    )
    categories = write_categories_json(
        tmp_path / "doc.categories.json",
        [
            ("doc.pdf#page-1", "title_rider", 1.0, False),
            ("doc.pdf#page-2", "title_rider", 1.0, False),
            ("doc.pdf#page-3", "lender_rate_note", 0.9, False),
            ("doc.pdf#page-4", "lender_rate_note", 0.9, False),
            ("doc.pdf#page-5", "other", 0.5, False),
            ("doc.pdf#page-6", "title_rider", 0.8, False),
            ("doc.pdf#page-7", "title_rider", 0.8, False),
        ],
    )
    vlm = write_vlm_json(
        tmp_path / "doc.vlm.json",
        [
            ("doc.pdf#page-1", "1"),
            ("doc.pdf#page-2", "2"),
            ("doc.pdf#page-3", "1"),
            ("doc.pdf#page-4", "2"),
            ("doc.pdf#page-5", None),
            ("doc.pdf#page-6", "1"),
            ("doc.pdf#page-7", "2"),
        ],
    )
    transport = FakeTransport(
        {
            "Document: alpha": [1.0, 0.0, 0.0],
            "Document: beta": [1.0, 0.0, 0.0],
            "Document: gamma": [0.0, 1.0, 0.0],
        }
    )
    client = make_client(transport)

    report, _ = run_matching([ocr], client, 0.78)

    assert report["auxiliary"] == {
        "categories_json": str(categories),
        "vlm_json": str(vlm),
    }
    assert report["parameters"]["weights"] == {
        "embedding": 0.6,
        "category": 0.25,
        "page_number": 0.15,
    }
    assert report["parameters"]["min_embedding_similarity"] == 0.6
    assert [group["page_refs"] for group in report["groups"]] == [
        ["doc.pdf#page-1", "doc.pdf#page-2"],
        ["doc.pdf#page-3", "doc.pdf#page-4", "doc.pdf#page-5"],
        ["doc.pdf#page-6", "doc.pdf#page-7"],
    ]
    second_group = report["groups"][1]
    page_four_edge = second_group["members"][1]["edge"]
    assert page_four_edge is not None
    assert page_four_edge["combined_similarity"] == pytest.approx(
        0.6 + 0.25 * 0.9 + 0.15, abs=1e-6
    )
    assert page_four_edge["signals"] == {
        "embedding_similarity": 1.0,
        "category_match": 0.9,
        "page_number_sequence": 1.0,
    }
    page_five_edge = second_group["members"][2]["edge"]
    assert page_five_edge is not None
    assert page_five_edge["combined_similarity"] == pytest.approx(1.0)
    assert page_five_edge["signals"]["category_match"] is None
    assert page_five_edge["signals"]["page_number_sequence"] is None
    assert page_five_edge["available_weight"] == pytest.approx(0.6)
    third_group = report["groups"][2]
    page_seven_edge = third_group["members"][1]["edge"]
    assert page_seven_edge is not None
    assert page_seven_edge["combined_similarity"] == pytest.approx(
        0.6 + 0.25 * 0.8 + 0.15, abs=1e-6
    )
    assert third_group["min_combined_similarity"] == pytest.approx(
        page_seven_edge["combined_similarity"]
    )
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
    assert members[1]["edge"]["combined_similarity"] == pytest.approx(0.9, abs=1e-6)


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
