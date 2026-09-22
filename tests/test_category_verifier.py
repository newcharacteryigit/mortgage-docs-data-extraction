from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path
from typing import Any

import pytest

import category_verifier
from category_verifier import (
    CATEGORY_RESPONSE_SCHEMA,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MARGIN_THRESHOLD,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    INPUT_MODE,
    RESOLUTION_ACCEPTED,
    RESOLUTION_DISAGREEMENT,
    RESOLUTION_MUTUAL_OTHER,
    RESOLUTION_NOT_TRIGGERED,
    RESOLUTION_VERIFIER_FAILED,
    REVIEW_REASON_DISAGREEMENT,
    REVIEW_REASON_VERIFIER_FAILED,
    SYSTEM_PROMPT,
    TRIGGER_LOW_CONFIDENCE,
    TRIGGER_LOW_MARGIN,
    CategoryRecord,
    VerifierClient,
    build_parser,
    default_output_path,
    evaluate_trigger,
    load_categories_report,
    main,
    parse_verifier_response,
    request_json,
    resolve_decision,
    resolve_ocr_path,
    run_verification,
)
from page_classifier import (
    CATEGORY_LABELS,
    REVIEW_REASON_LOW_CONFIDENCE,
    REVIEW_REASON_OTHER_CATEGORY,
)
from page_matcher import load_categories_json

OCR_NAME = "document.json"
CATEGORIES_NAME = "document.categories.json"
SOURCE = "document.pdf"
PDF_REF_PREFIX = f"{SOURCE}#page-"
TAX_CATEGORY = "property_tax_record_information_sheet"
NOTE_CATEGORY = "lender_rate_note"
CLOSING_CATEGORY = "mortgage_closing_disclosure_seller"
RIDER_CATEGORY = "title_rider"
AFFIDAVIT_CATEGORY = "title_signature_name_affidavit_ack"
OTHER_CATEGORY = "other"
FAKE_RESPONSE_MODEL = "fake-verifier-model"
PAGE_ONE_TEXT = "Closing Disclosure\nCLOSING DISCLOSURE PAGE 2a OF 2"
PAGE_TWO_TEXT = "NOTE\nUNIFORM SECURED NOTE"


class FakeTransport:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, url: str, payload: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        self.calls.append({"url": url, "payload": payload, "timeout": timeout})
        index = len(self.calls) - 1
        if index >= len(self.responses):
            raise AssertionError(f"Unexpected request {index + 1}: {url}")
        return self.responses[index]


class FlakyTransport(FakeTransport):
    def __init__(self, responses: list[dict[str, Any]], failures: set[int]) -> None:
        super().__init__(responses)
        self.failures = failures

    def __call__(
        self, url: str, payload: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        index = len(self.calls)
        self.calls.append({"url": url, "payload": payload, "timeout": timeout})
        if index in self.failures:
            raise RuntimeError("boom")
        return self.responses.pop(0)


class FakeHttpResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> FakeHttpResponse:
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def probabilities_for(choice: str, choice_probability: float = 0.9) -> dict[str, float]:
    others = [category for category in CATEGORY_LABELS if category != choice]
    remaining = (1.0 - choice_probability) / len(others)
    probabilities = {category: remaining for category in others}
    probabilities[choice] = choice_probability
    return probabilities


def chat_response(
    category: str = CLOSING_CATEGORY,
    model: str = FAKE_RESPONSE_MODEL,
    finish_reason: str = "stop",
    content: str | None = None,
    reasoning_content: str | None = None,
    include_usage: bool = True,
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": (
            content if content is not None else json.dumps({"category": category})
        ),
    }
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    payload: dict[str, Any] = {
        "model": model,
        "choices": [
            {"index": 0, "message": message, "finish_reason": finish_reason}
        ],
    }
    if include_usage:
        payload["usage"] = {"prompt_tokens": 120, "completion_tokens": 8}
    return payload


def make_client(transport: FakeTransport) -> VerifierClient:
    return VerifierClient(
        base_url="http://127.0.0.1:1234/v1",
        model=DEFAULT_MODEL,
        timeout=30.0,
        max_tokens=DEFAULT_MAX_TOKENS,
        seed=0,
        request=transport,
    )


def write_ocr_json(
    path: Path, pages: list[tuple[int, str]], source: str = SOURCE
) -> Path:
    payload = {
        "source": source,
        "engine": "paddleocr",
        "pages": [
            {
                "page_number": number,
                "text": text,
                "line_count": 3,
                "mean_score": 0.9,
                "duration_seconds": 0.1,
            }
            for number, text in pages
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def category_page(
    number: int,
    category: str | None,
    confidence: float | None = 0.99,
    probabilities: dict[str, float] | None = None,
    needs_review: bool = False,
    review_reason: str | None = None,
    page_ref: str | None = None,
) -> dict[str, Any]:
    ref = page_ref if page_ref is not None else f"{PDF_REF_PREFIX}{number}"
    if category is None:
        return {
            "page_number": number,
            "page_ref": ref,
            "category": None,
            "category_label": None,
            "confidence": None,
            "probabilities": None,
            "needs_review": True,
            "review_reason": "empty text",
            "ocr_reference": {"line_count": 0, "mean_score": None, "duration_seconds": 0.0},
            "timings_seconds": {"classify_seconds": 0.0, "total_seconds": 0.0},
            "usage": None,
        }
    return {
        "page_number": number,
        "page_ref": ref,
        "category": category,
        "category_label": CATEGORY_LABELS[category],
        "confidence": confidence,
        "probabilities": (
            probabilities
            if probabilities is not None
            else probabilities_for(category)
        ),
        "needs_review": needs_review,
        "review_reason": review_reason,
        "ocr_reference": {"line_count": 3, "mean_score": 0.9, "duration_seconds": 0.1},
        "timings_seconds": {"classify_seconds": 0.5, "total_seconds": 0.5},
        "usage": {"input_tokens": 100, "output_tokens": 10},
    }


def write_categories_json(
    path: Path, ocr_path: Path, pages: list[dict[str, Any]]
) -> Path:
    payload = {
        "schema_version": "1.0",
        "input_files": {"ocr_json": str(ocr_path)},
        "model": {
            "id": "jev-latest",
            "base_url": "https://api.typesafe.ai/v1",
            "prompt_version": "1.0",
            "response_model": "jev-1.13.0",
        },
        "categories": dict(CATEGORY_LABELS),
        "parameters": {
            "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
            "max_state_chars": 30000,
            "timeout_seconds": 60.0,
        },
        "summary": {},
        "pages": pages,
        "skipped_pages": [],
        "timings": {
            "total_seconds": 1.0,
            "classify_seconds": 1.0,
            "requests": len(pages),
            "avg_classify_seconds_per_page": 0.1,
        },
        "usage": {"input_tokens": 10, "output_tokens": 5, "requests": len(pages)},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def sample_inputs(
    tmp_path: Path,
    ocr_pages: list[tuple[int, str]],
    category_pages: list[dict[str, Any]],
) -> tuple[Path, Path]:
    ocr_path = write_ocr_json(tmp_path / OCR_NAME, ocr_pages)
    categories_path = write_categories_json(
        tmp_path / CATEGORIES_NAME, ocr_path, category_pages
    )
    return categories_path, ocr_path


def make_record(
    category: str | None = CLOSING_CATEGORY,
    confidence: float | None = 0.99,
    probabilities: dict[str, float] | None = None,
    needs_review: bool = False,
    review_reason: str | None = None,
    page_number: int = 1,
) -> CategoryRecord:
    return CategoryRecord(
        page_number=page_number,
        page_ref=f"{PDF_REF_PREFIX}{page_number}",
        category=category,
        confidence=confidence,
        probabilities=(
            probabilities
            if probabilities is not None
            else (probabilities_for(category) if category is not None else None)
        ),
        needs_review=needs_review,
        review_reason=review_reason,
        raw={},
    )


def test_default_output_path(tmp_path: Path) -> None:
    assert default_output_path(tmp_path / CATEGORIES_NAME) == (
        tmp_path / "document.category_review.json"
    )
    assert default_output_path(tmp_path / "other.json") == (
        tmp_path / "other.category_review.json"
    )


def test_build_parser_defaults() -> None:
    args = build_parser().parse_args(["document.categories.json"])
    assert args.model == DEFAULT_MODEL
    assert args.confidence_threshold == DEFAULT_CONFIDENCE_THRESHOLD
    assert args.margin_threshold == DEFAULT_MARGIN_THRESHOLD
    assert args.max_tokens == DEFAULT_MAX_TOKENS
    assert args.ocr is None


def test_load_categories_report_requires_ocr_reference(tmp_path: Path) -> None:
    path = tmp_path / CATEGORIES_NAME
    path.write_text(json.dumps({"input_files": {}, "pages": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="input_files.ocr_json"):
        load_categories_report(path)


def test_load_categories_report_rejects_unknown_category(tmp_path: Path) -> None:
    ocr_path = write_ocr_json(tmp_path / OCR_NAME, [(1, PAGE_ONE_TEXT)])
    page = category_page(1, CLOSING_CATEGORY)
    page["category"] = "not_a_category"
    categories_path = write_categories_json(
        tmp_path / CATEGORIES_NAME, ocr_path, [page]
    )
    with pytest.raises(ValueError, match="unknown category"):
        load_categories_report(categories_path)


def test_load_categories_report_rejects_bad_probability_keys(tmp_path: Path) -> None:
    ocr_path = write_ocr_json(tmp_path / OCR_NAME, [(1, PAGE_ONE_TEXT)])
    probabilities = probabilities_for(CLOSING_CATEGORY)
    del probabilities[OTHER_CATEGORY]
    page = category_page(1, CLOSING_CATEGORY, probabilities=probabilities)
    categories_path = write_categories_json(
        tmp_path / CATEGORIES_NAME, ocr_path, [page]
    )
    with pytest.raises(ValueError, match="probability keys"):
        load_categories_report(categories_path)


def test_load_categories_report_rejects_probability_sum(tmp_path: Path) -> None:
    ocr_path = write_ocr_json(tmp_path / OCR_NAME, [(1, PAGE_ONE_TEXT)])
    probabilities = {category: 0.1 for category in CATEGORY_LABELS}
    page = category_page(1, CLOSING_CATEGORY, probabilities=probabilities)
    categories_path = write_categories_json(
        tmp_path / CATEGORIES_NAME, ocr_path, [page]
    )
    with pytest.raises(ValueError, match="summing"):
        load_categories_report(categories_path)


def test_resolve_ocr_path_prefers_existing_and_falls_back_to_sibling(
    tmp_path: Path,
) -> None:
    categories_path = tmp_path / CATEGORIES_NAME
    ocr_path = write_ocr_json(tmp_path / OCR_NAME, [(1, PAGE_ONE_TEXT)])
    assert resolve_ocr_path(categories_path, OCR_NAME) == ocr_path
    assert resolve_ocr_path(categories_path, str(ocr_path)) == ocr_path
    with pytest.raises(FileNotFoundError):
        resolve_ocr_path(categories_path, "missing.json")


def test_evaluate_trigger_flags_low_confidence() -> None:
    decision = evaluate_trigger(make_record(confidence=0.4), 0.5, 0.5)
    assert decision.triggered is True
    assert decision.reasons == (TRIGGER_LOW_CONFIDENCE,)
    assert decision.margin == pytest.approx(0.88)


def test_evaluate_trigger_flags_low_margin() -> None:
    probabilities = {
        CLOSING_CATEGORY: 0.55,
        NOTE_CATEGORY: 0.35,
        TAX_CATEGORY: 0.04,
        RIDER_CATEGORY: 0.02,
        AFFIDAVIT_CATEGORY: 0.02,
        OTHER_CATEGORY: 0.02,
    }
    decision = evaluate_trigger(
        make_record(confidence=0.9, probabilities=probabilities), 0.5, 0.5
    )
    assert decision.triggered is True
    assert decision.reasons == (TRIGGER_LOW_MARGIN,)
    assert decision.margin == pytest.approx(0.2)


def test_evaluate_trigger_combines_reasons() -> None:
    probabilities = {
        CLOSING_CATEGORY: 0.55,
        NOTE_CATEGORY: 0.35,
        TAX_CATEGORY: 0.04,
        RIDER_CATEGORY: 0.02,
        AFFIDAVIT_CATEGORY: 0.02,
        OTHER_CATEGORY: 0.02,
    }
    decision = evaluate_trigger(
        make_record(confidence=0.4, probabilities=probabilities), 0.5, 0.5
    )
    assert decision.reasons == (TRIGGER_LOW_CONFIDENCE, TRIGGER_LOW_MARGIN)


def test_evaluate_trigger_does_not_flag_high_confidence_page() -> None:
    decision = evaluate_trigger(make_record(), 0.5, 0.5)
    assert decision.triggered is False
    assert decision.reasons == ()
    assert decision.margin == pytest.approx(0.88)


def test_evaluate_trigger_ignores_page_without_category() -> None:
    decision = evaluate_trigger(make_record(category=None), 0.5, 0.5)
    assert decision.triggered is False
    assert decision.margin is None


def test_resolve_decision_rules() -> None:
    assert resolve_decision(CLOSING_CATEGORY, CLOSING_CATEGORY) == (
        RESOLUTION_ACCEPTED
    )
    assert resolve_decision(CLOSING_CATEGORY, NOTE_CATEGORY) == (
        RESOLUTION_DISAGREEMENT
    )
    assert resolve_decision(OTHER_CATEGORY, OTHER_CATEGORY) == (
        RESOLUTION_MUTUAL_OTHER
    )
    assert resolve_decision(OTHER_CATEGORY, CLOSING_CATEGORY) == (
        RESOLUTION_DISAGREEMENT
    )


def test_build_payload_is_text_only_and_deterministic() -> None:
    client = make_client(FakeTransport([]))
    payload = client.build_payload(PAGE_ONE_TEXT)
    assert payload["model"] == DEFAULT_MODEL
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1] == {"role": "user", "content": PAGE_ONE_TEXT}
    assert "image_url" not in json.dumps(payload)
    assert payload["temperature"] == 0.0
    assert payload["top_p"] == 1.0
    assert payload["seed"] == 0
    assert payload["stream"] is False
    assert payload["reasoning_effort"] == "none"
    response_format = payload["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == CATEGORY_RESPONSE_SCHEMA
    for category in CATEGORY_LABELS:
        assert category in SYSTEM_PROMPT
    assert "continuation page" in SYSTEM_PROMPT


def test_parse_verifier_response_plain_json() -> None:
    answer = parse_verifier_response(chat_response(NOTE_CATEGORY), 1)
    assert answer.category == NOTE_CATEGORY
    assert answer.response_model == FAKE_RESPONSE_MODEL
    assert answer.usage == {"prompt_tokens": 120, "completion_tokens": 8}


def test_parse_verifier_response_strips_code_fences() -> None:
    content = f"```json\n{json.dumps({'category': RIDER_CATEGORY})}\n```"
    answer = parse_verifier_response(chat_response(content=content), 1)
    assert answer.category == RIDER_CATEGORY


def test_parse_verifier_response_uses_reasoning_fallback() -> None:
    answer = parse_verifier_response(
        chat_response(
            content="",
            reasoning_content=json.dumps({"category": AFFIDAVIT_CATEGORY}),
        ),
        1,
    )
    assert answer.category == AFFIDAVIT_CATEGORY


def test_parse_verifier_response_rejects_unknown_category() -> None:
    with pytest.raises(RuntimeError, match="unknown category"):
        parse_verifier_response(chat_response(category="bogus"), 1)


def test_parse_verifier_response_rejects_wrong_fields() -> None:
    content = json.dumps({"page_category": CLOSING_CATEGORY})
    with pytest.raises(RuntimeError, match="wrong fields"):
        parse_verifier_response(chat_response(content=content), 1)


def test_parse_verifier_response_rejects_truncated_response() -> None:
    with pytest.raises(RuntimeError, match="truncated"):
        parse_verifier_response(chat_response(finish_reason="length"), 1)


def test_parse_verifier_response_requires_usage() -> None:
    with pytest.raises(RuntimeError, match="usage"):
        parse_verifier_response(chat_response(include_usage=False), 1)


def test_parse_verifier_response_rejects_non_json() -> None:
    with pytest.raises(RuntimeError, match="no JSON object"):
        parse_verifier_response(chat_response(content="not json"), 1)


def test_run_verification_default_thresholds_do_not_trigger(tmp_path: Path) -> None:
    categories_path, ocr_path = sample_inputs(
        tmp_path,
        [(1, PAGE_ONE_TEXT), (2, "   ")],
        [
            category_page(1, CLOSING_CATEGORY),
            category_page(2, None),
        ],
    )
    transport = FakeTransport([])
    report = run_verification(categories_path, make_client(transport))
    assert transport.calls == []
    assert report["summary"] == {
        "page_count": 2,
        "triggered_page_count": 0,
        "accepted_page_count": 0,
        "mutual_other_page_count": 0,
        "disagreement_page_count": 0,
        "verifier_failed_page_count": 0,
        "needs_review_before": 1,
        "needs_review_after": 1,
        "triggered_pages": [],
    }
    assert report["pages"][0]["category"] == CLOSING_CATEGORY
    assert report["pages"][0]["needs_review"] is False
    assert report["pages"][0]["verification"]["resolution"] == (
        RESOLUTION_NOT_TRIGGERED
    )
    assert report["pages"][0]["verification"]["margin"] == pytest.approx(0.88)
    assert report["pages"][1]["review_reason"] == "empty text"
    assert report["input_files"]["ocr_json"] == str(ocr_path)
    assert report["parameters"]["input_mode"] == INPUT_MODE
    assert report["model"]["response_model"] is None
    assert report["timings"]["requests"] == 0
    assert report["timings"]["avg_verify_seconds_per_request"] is None
    assert report["usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "requests": 0,
    }
    json.dumps(report)


def test_run_verification_accepts_agreement_and_clears_review(
    tmp_path: Path,
) -> None:
    categories_path, _ = sample_inputs(
        tmp_path,
        [(1, PAGE_ONE_TEXT)],
        [
            category_page(
                1,
                CLOSING_CATEGORY,
                confidence=0.4,
                needs_review=True,
                review_reason=REVIEW_REASON_LOW_CONFIDENCE,
            )
        ],
    )
    transport = FakeTransport([chat_response(CLOSING_CATEGORY)])
    report = run_verification(categories_path, make_client(transport))
    assert len(transport.calls) == 1
    assert transport.calls[0]["payload"]["messages"][1]["content"] == PAGE_ONE_TEXT
    page = report["pages"][0]
    assert page["verification"]["triggered"] is True
    assert page["verification"]["trigger_reasons"] == [TRIGGER_LOW_CONFIDENCE]
    assert page["verification"]["verifier_category"] == CLOSING_CATEGORY
    assert page["verification"]["resolution"] == RESOLUTION_ACCEPTED
    assert page["verification"]["error"] is None
    assert page["needs_review"] is False
    assert page["review_reason"] is None
    assert report["summary"]["accepted_page_count"] == 1
    assert report["summary"]["needs_review_before"] == 1
    assert report["summary"]["needs_review_after"] == 0
    assert report["summary"]["triggered_pages"] == [1]
    assert report["model"]["response_model"] == FAKE_RESPONSE_MODEL
    assert report["usage"] == {
        "prompt_tokens": 120,
        "completion_tokens": 8,
        "requests": 1,
    }
    assert report["timings"]["requests"] == 1


def test_run_verification_flags_disagreement(tmp_path: Path) -> None:
    categories_path, _ = sample_inputs(
        tmp_path,
        [(1, PAGE_ONE_TEXT)],
        [category_page(1, CLOSING_CATEGORY, confidence=0.4)],
    )
    transport = FakeTransport([chat_response(NOTE_CATEGORY)])
    report = run_verification(categories_path, make_client(transport))
    page = report["pages"][0]
    assert page["verification"]["resolution"] == RESOLUTION_DISAGREEMENT
    assert page["verification"]["verifier_category"] == NOTE_CATEGORY
    assert page["needs_review"] is True
    assert page["review_reason"] == REVIEW_REASON_DISAGREEMENT
    assert report["summary"]["disagreement_page_count"] == 1


def test_run_verification_mutual_other_is_not_accepted(tmp_path: Path) -> None:
    probabilities = {
        OTHER_CATEGORY: 0.55,
        CLOSING_CATEGORY: 0.35,
        TAX_CATEGORY: 0.04,
        RIDER_CATEGORY: 0.02,
        AFFIDAVIT_CATEGORY: 0.02,
        NOTE_CATEGORY: 0.02,
    }
    categories_path, _ = sample_inputs(
        tmp_path,
        [(1, PAGE_ONE_TEXT)],
        [
            category_page(
                1,
                OTHER_CATEGORY,
                confidence=0.6,
                probabilities=probabilities,
                needs_review=True,
                review_reason=REVIEW_REASON_OTHER_CATEGORY,
            )
        ],
    )
    transport = FakeTransport([chat_response(OTHER_CATEGORY)])
    report = run_verification(categories_path, make_client(transport))
    page = report["pages"][0]
    assert page["verification"]["trigger_reasons"] == [TRIGGER_LOW_MARGIN]
    assert page["verification"]["resolution"] == RESOLUTION_MUTUAL_OTHER
    assert page["needs_review"] is True
    assert page["review_reason"] == REVIEW_REASON_OTHER_CATEGORY
    assert report["summary"]["mutual_other_page_count"] == 1


def test_run_verification_failure_marks_page_and_continues(tmp_path: Path) -> None:
    categories_path, _ = sample_inputs(
        tmp_path,
        [(1, PAGE_ONE_TEXT), (2, PAGE_TWO_TEXT)],
        [
            category_page(1, CLOSING_CATEGORY, confidence=0.4),
            category_page(2, NOTE_CATEGORY, confidence=0.4),
        ],
    )
    transport = FlakyTransport([chat_response(NOTE_CATEGORY)], failures={0})
    report = run_verification(categories_path, make_client(transport))
    first = report["pages"][0]
    assert first["verification"]["resolution"] == RESOLUTION_VERIFIER_FAILED
    assert first["verification"]["error"] == "RuntimeError: boom"
    assert first["needs_review"] is True
    assert first["review_reason"] == REVIEW_REASON_VERIFIER_FAILED
    second = report["pages"][1]
    assert second["verification"]["resolution"] == RESOLUTION_ACCEPTED
    assert second["needs_review"] is False
    assert report["summary"]["verifier_failed_page_count"] == 1
    assert report["summary"]["accepted_page_count"] == 1
    assert report["timings"]["requests"] == 2


def test_run_verification_rejects_oversized_triggered_page(tmp_path: Path) -> None:
    categories_path, _ = sample_inputs(
        tmp_path,
        [(1, "abcdefgh")],
        [category_page(1, CLOSING_CATEGORY, confidence=0.4)],
    )
    transport = FakeTransport([])
    with pytest.raises(ValueError, match="state limit"):
        run_verification(
            categories_path, make_client(transport), max_state_chars=5
        )
    assert transport.calls == []


def test_run_verification_rejects_page_ref_mismatch(tmp_path: Path) -> None:
    categories_path, _ = sample_inputs(
        tmp_path,
        [(1, PAGE_ONE_TEXT)],
        [category_page(1, CLOSING_CATEGORY, page_ref="other.pdf#page-1")],
    )
    with pytest.raises(ValueError, match="page_ref"):
        run_verification(categories_path, make_client(FakeTransport([])))


def test_run_verification_rejects_page_coverage_mismatch(tmp_path: Path) -> None:
    categories_path, _ = sample_inputs(
        tmp_path,
        [(1, PAGE_ONE_TEXT)],
        [
            category_page(1, CLOSING_CATEGORY),
            category_page(2, NOTE_CATEGORY),
        ],
    )
    with pytest.raises(ValueError, match="do not match"):
        run_verification(categories_path, make_client(FakeTransport([])))


def test_run_verification_rejects_invalid_threshold(tmp_path: Path) -> None:
    categories_path, _ = sample_inputs(
        tmp_path,
        [(1, PAGE_ONE_TEXT)],
        [category_page(1, CLOSING_CATEGORY)],
    )
    with pytest.raises(ValueError, match="Margin threshold"):
        run_verification(
            categories_path, make_client(FakeTransport([])), margin_threshold=1.5
        )


def test_output_is_page_matcher_compatible(tmp_path: Path) -> None:
    categories_path, _ = sample_inputs(
        tmp_path,
        [(1, PAGE_ONE_TEXT), (2, PAGE_TWO_TEXT)],
        [
            category_page(1, CLOSING_CATEGORY, confidence=0.4),
            category_page(2, NOTE_CATEGORY, confidence=0.4),
        ],
    )
    transport = FakeTransport(
        [chat_response(CLOSING_CATEGORY), chat_response(CLOSING_CATEGORY)]
    )
    report = run_verification(categories_path, make_client(transport))
    report_path = tmp_path / "verified.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    signals = load_categories_json(report_path)
    assert signals[f"{PDF_REF_PREFIX}1"] is not None
    assert signals[f"{PDF_REF_PREFIX}2"] is None


def test_main_writes_default_output_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    categories_path, _ = sample_inputs(
        tmp_path,
        [(1, PAGE_ONE_TEXT)],
        [category_page(1, CLOSING_CATEGORY, confidence=0.4)],
    )
    transport = FakeTransport([chat_response(CLOSING_CATEGORY)])
    monkeypatch.setattr(category_verifier, "request_json", transport)
    rc = main([str(categories_path)])
    assert rc == 0
    out_path = default_output_path(categories_path)
    assert out_path.is_file()
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["pages"][0]["verification"]["resolution"] == RESOLUTION_ACCEPTED
    assert transport.calls[0]["url"].endswith("/chat/completions")


def test_main_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    categories_path, _ = sample_inputs(
        tmp_path,
        [(1, PAGE_ONE_TEXT)],
        [category_page(1, CLOSING_CATEGORY)],
    )
    monkeypatch.setattr(category_verifier, "request_json", FakeTransport([]))
    rc = main([str(categories_path), "--stdout"])
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["summary"]["triggered_page_count"] == 0


def test_main_rejects_invalid_arguments(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    categories_path, _ = sample_inputs(
        tmp_path,
        [(1, PAGE_ONE_TEXT)],
        [category_page(1, CLOSING_CATEGORY)],
    )
    rc = main([str(categories_path), "--margin-threshold", "2"])
    assert rc == 1
    assert "margin-threshold" in capsys.readouterr().err


def test_request_json_reports_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Any, timeout: float) -> FakeHttpResponse:
        raise urllib.error.HTTPError(
            request.full_url,
            500,
            "Server Error",
            {},
            io.BytesIO(b'{"error": "boom"}'),
        )

    monkeypatch.setattr(category_verifier.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="HTTP 500"):
        request_json("http://127.0.0.1:1234/v1/chat/completions", {}, 5.0)


def test_request_json_rejects_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Any, timeout: float) -> FakeHttpResponse:
        return FakeHttpResponse(b"not json")

    monkeypatch.setattr(category_verifier.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="Invalid JSON"):
        request_json("http://127.0.0.1:1234/v1/chat/completions", {}, 5.0)
