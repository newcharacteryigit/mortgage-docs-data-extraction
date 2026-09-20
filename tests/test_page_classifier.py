from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path
from typing import Any

import pytest

import page_classifier
from page_classifier import (
    API_KEY_ENV_VAR,
    CATEGORY_CRITERIA,
    CATEGORY_LABELS,
    DEFAULT_BASE_URL,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_SUFFIX,
    DEFAULT_TIMEOUT_SECONDS,
    QUESTION_ID,
    REVIEW_REASON_EMPTY_TEXT,
    REVIEW_REASON_LOW_CONFIDENCE,
    REVIEW_REASON_OTHER_CATEGORY,
    SYSTEM_ONE_PATH,
    TypesafeClient,
    default_output_path,
    load_ocr_document,
    main,
    parse_classification_response,
    parse_env_file,
    request_json,
    resolve_api_key,
    run_classification,
)

OCR_NAME = "document.json"
TAX_CATEGORY = "property_tax_record_information_sheet"
RIDER_CATEGORY = "title_rider"
NOTE_CATEGORY = "lender_rate_note"
CLOSING_CATEGORY = "mortgage_closing_disclosure_seller"
AFFIDAVIT_CATEGORY = "title_signature_name_affidavit_ack"
OTHER_CATEGORY = "other"
FAKE_API_KEY = "test-api-key"
FAKE_RESPONSE_MODEL = "jev-1.13.0"


class FakeTransport:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, url: str, payload: dict[str, Any], timeout: float, api_key: str
    ) -> dict[str, Any]:
        self.calls.append(
            {"url": url, "payload": payload, "timeout": timeout, "api_key": api_key}
        )
        index = len(self.calls) - 1
        if index >= len(self.responses):
            raise AssertionError(f"Unexpected request {index + 1}: {url}")
        return self.responses[index]


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


def classify_response(
    choice: str,
    confidence: float = 0.95,
    probabilities: dict[str, float] | None = None,
    model: str = FAKE_RESPONSE_MODEL,
    input_tokens: int = 100,
    output_tokens: int = 10,
) -> dict[str, Any]:
    return {
        "model": model,
        "answers": {
            QUESTION_ID: {
                "type": "choice",
                "choice": choice,
                "confidence": confidence,
                "probabilities": (
                    probabilities if probabilities is not None else probabilities_for(choice)
                ),
            }
        },
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def make_client(transport: FakeTransport) -> TypesafeClient:
    return TypesafeClient(
        base_url=DEFAULT_BASE_URL,
        model=DEFAULT_MODEL,
        api_key=FAKE_API_KEY,
        timeout=DEFAULT_TIMEOUT_SECONDS,
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
                "line_count": 3,
                "mean_score": 0.9,
                "duration_seconds": 0.1,
            }
            for number, text in pages
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_default_output_path(tmp_path: Path) -> None:
    assert default_output_path(tmp_path / "document.json") == (
        tmp_path / f"document{DEFAULT_OUTPUT_SUFFIX}"
    )


def test_category_metadata_is_consistent() -> None:
    assert set(CATEGORY_LABELS) == set(CATEGORY_CRITERIA)
    assert "other" in CATEGORY_LABELS
    assert len(CATEGORY_LABELS) == 6


def test_parse_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# comment",
                "",
                "export TYPESAFE_API_KEY=from-export",
                'QUOTED="quoted value"',
                "SINGLE='single value'",
                "WITHOUT_EQUALS",
                "  SPACED = spaced  ",
            ]
        ),
        encoding="utf-8",
    )
    entries = parse_env_file(env_file)
    assert entries["TYPESAFE_API_KEY"] == "from-export"
    assert entries["QUOTED"] == "quoted value"
    assert entries["SINGLE"] == "single value"
    assert entries["SPACED"] == "spaced"
    assert "WITHOUT_EQUALS" not in entries


def test_parse_env_file_missing_returns_empty(tmp_path: Path) -> None:
    assert parse_env_file(tmp_path / "missing.env") == {}


def test_resolve_api_key_prefers_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(f"{API_KEY_ENV_VAR}=file-key", encoding="utf-8")
    monkeypatch.setenv(API_KEY_ENV_VAR, "environment-key")
    assert resolve_api_key(env_file) == "environment-key"


def test_resolve_api_key_reads_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(f"{API_KEY_ENV_VAR}=file-key", encoding="utf-8")
    monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
    assert resolve_api_key(env_file) == "file-key"


def test_resolve_api_key_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
    with pytest.raises(ValueError, match=API_KEY_ENV_VAR):
        resolve_api_key(tmp_path / "missing.env")


def test_client_payload_and_endpoint() -> None:
    client = make_client(FakeTransport([]))
    payload = client.build_payload("MORTGAGE NOTE")
    assert client.endpoint == f"{DEFAULT_BASE_URL}{SYSTEM_ONE_PATH}"
    assert payload["state"] == "MORTGAGE NOTE"
    assert payload["model"] == DEFAULT_MODEL
    assert set(payload["questions"]) == {QUESTION_ID}
    question = payload["questions"][QUESTION_ID]
    assert question["type"] == "choice"
    assert question["criteria"] == CATEGORY_CRITERIA
    assert "continuation page" in question["instructions"]


def test_classify_page_sends_api_key_and_state() -> None:
    transport = FakeTransport([classify_response(TAX_CATEGORY)])
    client = make_client(transport)
    result = client.classify_page("TAX RECORD INFORMATION SHEET", 1)
    assert result.category == TAX_CATEGORY
    assert result.confidence == 0.95
    call = transport.calls[0]
    assert call["url"] == f"{DEFAULT_BASE_URL}{SYSTEM_ONE_PATH}"
    assert call["api_key"] == FAKE_API_KEY
    assert call["payload"]["state"] == "TAX RECORD INFORMATION SHEET"


def test_parse_classification_response_valid() -> None:
    result = parse_classification_response(classify_response(NOTE_CATEGORY), 2)
    assert result.category == NOTE_CATEGORY
    assert result.response_model == FAKE_RESPONSE_MODEL
    assert result.usage == {"input_tokens": 100, "output_tokens": 10}
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-9


def test_parse_response_rejects_unknown_category() -> None:
    with pytest.raises(RuntimeError, match="unknown category"):
        parse_classification_response(classify_response("unknown_category"), 1)


def test_parse_response_rejects_missing_answer() -> None:
    payload = classify_response(TAX_CATEGORY)
    payload["answers"] = {}
    with pytest.raises(RuntimeError, match=QUESTION_ID):
        parse_classification_response(payload, 1)


def test_parse_response_rejects_non_choice_answer() -> None:
    payload = classify_response(TAX_CATEGORY)
    payload["answers"][QUESTION_ID]["type"] = "noul"
    with pytest.raises(RuntimeError, match="non-choice"):
        parse_classification_response(payload, 1)


def test_parse_response_rejects_wrong_probability_keys() -> None:
    payload = classify_response(TAX_CATEGORY)
    payload["answers"][QUESTION_ID]["probabilities"] = {TAX_CATEGORY: 1.0}
    with pytest.raises(RuntimeError, match="wrong probability keys"):
        parse_classification_response(payload, 1)


def test_parse_response_rejects_unnormalized_probabilities() -> None:
    probabilities = probabilities_for(TAX_CATEGORY)
    probabilities[TAX_CATEGORY] = 0.5
    payload = classify_response(TAX_CATEGORY, probabilities=probabilities)
    with pytest.raises(RuntimeError, match="summing"):
        parse_classification_response(payload, 1)


def test_parse_response_rejects_out_of_range_confidence() -> None:
    payload = classify_response(TAX_CATEGORY, confidence=1.5)
    with pytest.raises(RuntimeError, match="out-of-range confidence"):
        parse_classification_response(payload, 1)


def test_parse_response_rejects_choice_without_highest_probability() -> None:
    probabilities = {category: 0.0 for category in CATEGORY_LABELS}
    probabilities[TAX_CATEGORY] = 0.3
    probabilities[RIDER_CATEGORY] = 0.7
    payload = classify_response(
        TAX_CATEGORY, confidence=0.4, probabilities=probabilities
    )
    with pytest.raises(RuntimeError, match="highest probability"):
        parse_classification_response(payload, 1)


def test_parse_response_rejects_non_integer_usage() -> None:
    payload = classify_response(TAX_CATEGORY)
    payload["usage"]["input_tokens"] = "100"
    with pytest.raises(RuntimeError, match="input_tokens"):
        parse_classification_response(payload, 1)


def test_run_classification_happy_path(tmp_path: Path) -> None:
    ocr_path = write_ocr_json(
        tmp_path / OCR_NAME,
        "AREAL_LOAN.pdf",
        [
            (1, "TAX RECORD INFORMATION SHEET\nTYPE OF TAX"),
            (2, "CONDOMINIUM RIDER\nTHIS CONDOMINIUM RIDER"),
            (3, "NOTE\nUNIFORM SECURED NOTE"),
        ],
    )
    transport = FakeTransport(
        [
            classify_response(TAX_CATEGORY),
            classify_response(RIDER_CATEGORY),
            classify_response(NOTE_CATEGORY),
        ]
    )
    report = run_classification(ocr_path, make_client(transport))
    assert [page["category"] for page in report["pages"]] == [
        TAX_CATEGORY,
        RIDER_CATEGORY,
        NOTE_CATEGORY,
    ]
    assert report["pages"][0]["category_label"] == CATEGORY_LABELS[TAX_CATEGORY]
    assert all(page["needs_review"] is False for page in report["pages"])
    assert report["summary"]["category_counts"][TAX_CATEGORY] == 1
    assert report["summary"]["classified_page_count"] == 3
    assert report["summary"]["skipped_page_count"] == 0
    assert report["summary"]["needs_review_pages"] == []
    assert report["timings"]["requests"] == 3
    assert report["timings"]["avg_classify_seconds_per_page"] is not None
    assert report["usage"] == {
        "input_tokens": 300,
        "output_tokens": 30,
        "requests": 3,
    }
    assert report["model"]["response_model"] == FAKE_RESPONSE_MODEL
    assert report["model"]["prompt_version"]
    assert report["schema_version"] == "1.0"
    assert report["pages"][0]["page_ref"] == "AREAL_LOAN.pdf#page-1"
    json.dumps(report)


def test_run_classification_skips_empty_page(tmp_path: Path) -> None:
    ocr_path = write_ocr_json(
        tmp_path / OCR_NAME,
        "AREAL_LOAN.pdf",
        [(1, "CLOSING DISCLOSURE"), (2, "   ")],
    )
    transport = FakeTransport([classify_response(CLOSING_CATEGORY)])
    report = run_classification(ocr_path, make_client(transport))
    assert len(transport.calls) == 1
    empty_page = report["pages"][1]
    assert empty_page["category"] is None
    assert empty_page["confidence"] is None
    assert empty_page["needs_review"] is True
    assert empty_page["review_reason"] == REVIEW_REASON_EMPTY_TEXT
    assert report["skipped_pages"] == [
        {
            "page_number": 2,
            "page_ref": "AREAL_LOAN.pdf#page-2",
            "reason": REVIEW_REASON_EMPTY_TEXT,
        }
    ]
    assert report["summary"]["skipped_page_count"] == 1
    assert report["summary"]["page_count"] == 2
    assert report["timings"]["requests"] == 1


def test_run_classification_flags_low_confidence(tmp_path: Path) -> None:
    ocr_path = write_ocr_json(
        tmp_path / OCR_NAME, "AREAL_LOAN.pdf", [(1, "SIGNATURE/NAME AFFIDAVIT")]
    )
    transport = FakeTransport(
        [classify_response(AFFIDAVIT_CATEGORY, confidence=DEFAULT_CONFIDENCE_THRESHOLD - 0.1)]
    )
    report = run_classification(ocr_path, make_client(transport))
    assert report["pages"][0]["needs_review"] is True
    assert report["pages"][0]["review_reason"] == REVIEW_REASON_LOW_CONFIDENCE


def test_run_classification_confidence_boundary_is_not_reviewed(tmp_path: Path) -> None:
    ocr_path = write_ocr_json(
        tmp_path / OCR_NAME, "AREAL_LOAN.pdf", [(1, "NOTE\nUNIFORM SECURED NOTE")]
    )
    transport = FakeTransport(
        [classify_response(NOTE_CATEGORY, confidence=DEFAULT_CONFIDENCE_THRESHOLD)]
    )
    report = run_classification(ocr_path, make_client(transport))
    assert report["pages"][0]["needs_review"] is False
    assert report["pages"][0]["review_reason"] is None


def test_run_classification_flags_other_category(tmp_path: Path) -> None:
    ocr_path = write_ocr_json(
        tmp_path / OCR_NAME, "AREAL_LOAN.pdf", [(1, "SOME OTHER FORM")]
    )
    transport = FakeTransport([classify_response(OTHER_CATEGORY, confidence=0.99)])
    report = run_classification(ocr_path, make_client(transport))
    assert report["pages"][0]["needs_review"] is True
    assert report["pages"][0]["review_reason"] == REVIEW_REASON_OTHER_CATEGORY
    assert report["summary"]["category_counts"][OTHER_CATEGORY] == 1


def test_run_classification_rejects_long_page_before_requests(tmp_path: Path) -> None:
    ocr_path = write_ocr_json(
        tmp_path / OCR_NAME, "AREAL_LOAN.pdf", [(1, "abcdefgh"), (2, "ijklmnop")]
    )
    transport = FakeTransport([])
    with pytest.raises(ValueError, match="state limit"):
        run_classification(ocr_path, make_client(transport), max_state_chars=5)
    assert transport.calls == []


def test_run_classification_rejects_invalid_threshold(tmp_path: Path) -> None:
    ocr_path = write_ocr_json(tmp_path / OCR_NAME, "AREAL_LOAN.pdf", [(1, "text")])
    with pytest.raises(ValueError, match="Confidence threshold"):
        run_classification(
            ocr_path, make_client(FakeTransport([])), confidence_threshold=1.5
        )


def test_load_ocr_document_rejects_non_increasing_pages(tmp_path: Path) -> None:
    ocr_path = write_ocr_json(
        tmp_path / OCR_NAME, "AREAL_LOAN.pdf", [(1, "one"), (1, "two")]
    )
    with pytest.raises(ValueError, match="non-increasing"):
        load_ocr_document(ocr_path)


def test_request_json_sends_bearer_header(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> FakeHttpResponse:
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeHttpResponse(json.dumps({"ok": True}).encode("utf-8"))

    monkeypatch.setattr(page_classifier.urllib.request, "urlopen", fake_urlopen)
    result = request_json("https://api.typesafe.ai/v1/systemone", {"a": 1}, 5.0, "secret")
    assert result == {"ok": True}
    assert captured["request"].get_header("Authorization") == "Bearer secret"
    assert captured["request"].get_header("Content-type") == "application/json"
    assert captured["timeout"] == 5.0
    assert json.loads(captured["request"].data.decode("utf-8")) == {"a": 1}


def test_request_json_reports_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Any, timeout: float) -> FakeHttpResponse:
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "Unauthorized",
            {},
            io.BytesIO(b'{"error": "invalid key"}'),
        )

    monkeypatch.setattr(page_classifier.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="HTTP 401"):
        request_json("https://api.typesafe.ai/v1/systemone", {}, 5.0, "bad")


def test_request_json_rejects_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Any, timeout: float) -> FakeHttpResponse:
        return FakeHttpResponse(b"not json")

    monkeypatch.setattr(page_classifier.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="Invalid JSON"):
        request_json("https://api.typesafe.ai/v1/systemone", {}, 5.0, "key")


def test_main_missing_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
    ocr_path = write_ocr_json(tmp_path / OCR_NAME, "AREAL_LOAN.pdf", [(1, "text")])
    rc = main([str(ocr_path), "--env-file", str(tmp_path / "missing.env")])
    assert rc == 1
    assert API_KEY_ENV_VAR in capsys.readouterr().err


def test_main_writes_output_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(API_KEY_ENV_VAR, FAKE_API_KEY)
    ocr_path = write_ocr_json(
        tmp_path / OCR_NAME, "AREAL_LOAN.pdf", [(1, "TAX RECORD INFORMATION SHEET")]
    )
    transport = FakeTransport([classify_response(TAX_CATEGORY)])
    monkeypatch.setattr(page_classifier, "request_json", transport)
    out_path = tmp_path / "result.json"
    rc = main([str(ocr_path), "--out", str(out_path)])
    assert rc == 0
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["pages"][0]["category"] == TAX_CATEGORY
    assert transport.calls[0]["api_key"] == FAKE_API_KEY


def test_main_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(API_KEY_ENV_VAR, FAKE_API_KEY)
    ocr_path = write_ocr_json(
        tmp_path / OCR_NAME, "AREAL_LOAN.pdf", [(1, "CONDOMINIUM RIDER")]
    )
    monkeypatch.setattr(
        page_classifier, "request_json", FakeTransport([classify_response(RIDER_CATEGORY)])
    )
    rc = main([str(ocr_path), "--stdout"])
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["pages"][0]["category"] == RIDER_CATEGORY


def test_main_rejects_invalid_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(API_KEY_ENV_VAR, FAKE_API_KEY)
    ocr_path = write_ocr_json(tmp_path / OCR_NAME, "AREAL_LOAN.pdf", [(1, "text")])
    rc = main([str(ocr_path), "--confidence-threshold", "2"])
    assert rc == 1
    assert "confidence-threshold" in capsys.readouterr().err


def test_main_default_output_path_is_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(API_KEY_ENV_VAR, FAKE_API_KEY)
    ocr_path = write_ocr_json(
        tmp_path / OCR_NAME, "AREAL_LOAN.pdf", [(1, "NOTE\nUNIFORM SECURED NOTE")]
    )
    monkeypatch.setattr(
        page_classifier, "request_json", FakeTransport([classify_response(NOTE_CATEGORY)])
    )
    rc = main([str(ocr_path)])
    assert rc == 0
    assert default_output_path(ocr_path).is_file()
