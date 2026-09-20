from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pymupdf
import pytest

from vlm_extract import (
    DEFAULT_DPI,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MISMATCH_THRESHOLD,
    DEFAULT_MODEL,
    DEFAULT_SEED,
    EXTRACTION_FIELDS,
    PNG_DATA_URL_PREFIX,
    VlmClient,
    best_window_similarity,
    build_parser,
    compare_field,
    default_ocr_path,
    default_output_path,
    load_ocr_document,
    normalize_for_match,
    parse_extraction_response,
    run_extraction,
    verify_extracted_fields,
)

PDF_NAME = "mortgage.pdf"
PAGE_ONE_OCR_TEXT = (
    "Borrower: Jane Doe\nLoan #: 12345\nPage 1 of 2\n"
    "Property Address: 1 Main St, Las Vegas, NV"
)
PAGE_TWO_OCR_TEXT = "UNRELATED PAGE CONTENT"


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


def chat_response(
    content: str | None,
    reasoning_content: str | None = None,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    return {"choices": [{"index": 0, "message": message, "finish_reason": finish_reason}]}


def fields_payload(**overrides: Any) -> str:
    fields: dict[str, Any] = {field: None for field in EXTRACTION_FIELDS}
    fields.update(overrides)
    return json.dumps(fields)


def make_client(transport: FakeTransport, thinking: bool = False) -> VlmClient:
    return VlmClient(
        base_url="http://127.0.0.1:1234/v1",
        model=DEFAULT_MODEL,
        timeout=30.0,
        max_tokens=DEFAULT_MAX_TOKENS,
        seed=DEFAULT_SEED,
        thinking=thinking,
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


@pytest.fixture(scope="module")
def sample_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("vlm") / PDF_NAME
    document = pymupdf.open()
    first = document.new_page()
    first.insert_text((72, 100), "Borrower: Jane Doe", fontsize=14)
    first.insert_text((72, 130), "Loan #: 12345", fontsize=14)
    second = document.new_page()
    second.insert_text((72, 100), PAGE_TWO_OCR_TEXT, fontsize=14)
    document.save(path)
    document.close()
    return path


def test_default_paths(tmp_path: Path) -> None:
    assert default_ocr_path(tmp_path / "document.pdf") == tmp_path / "document.json"
    assert default_output_path(tmp_path / "document.pdf") == tmp_path / "document.vlm.json"


def test_build_parser_defaults() -> None:
    args = build_parser().parse_args(["document.pdf"])
    assert args.model == DEFAULT_MODEL
    assert args.dpi == DEFAULT_DPI
    assert args.mismatch_threshold == DEFAULT_MISMATCH_THRESHOLD
    assert args.thinking is False
    assert args.ocr is None
    assert args.out is None


def test_build_payload_contains_image_schema_and_deterministic_params() -> None:
    transport = FakeTransport([])
    client = make_client(transport)
    payload = client.build_payload(b"\x89PNG-bytes")
    assert payload["model"] == DEFAULT_MODEL
    assert payload["temperature"] == 0.0
    assert payload["top_p"] == 1.0
    assert payload["seed"] == DEFAULT_SEED
    assert payload["max_tokens"] == DEFAULT_MAX_TOKENS
    assert payload["stream"] is False
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["reasoning_effort"] == "none"
    roles = [message["role"] for message in payload["messages"]]
    assert roles == ["system", "user"]
    assert isinstance(payload["messages"][0]["content"], str)
    user_content = payload["messages"][1]["content"]
    image_url = user_content[1]["image_url"]["url"]
    assert image_url.startswith(PNG_DATA_URL_PREFIX)
    response_format = payload["response_format"]
    assert response_format["type"] == "json_schema"
    schema = response_format["json_schema"]["schema"]
    assert schema["required"] == list(EXTRACTION_FIELDS)
    assert schema["additionalProperties"] is False
    assert schema["properties"]["borrower_name"]["type"] == ["string", "null"]


def test_build_payload_enables_thinking_when_requested() -> None:
    transport = FakeTransport([])
    client = make_client(transport, thinking=True)
    payload = client.build_payload(b"\x89PNG-bytes")
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}
    assert "reasoning_effort" not in payload


def test_extract_page_parses_plain_json() -> None:
    transport = FakeTransport(
        [chat_response(fields_payload(borrower_name="Jane Doe", loan_number="12345"))]
    )
    client = make_client(transport)
    fields = client.extract_page(b"\x89PNG-bytes", 1)
    assert fields["borrower_name"] == "Jane Doe"
    assert fields["loan_number"] == "12345"
    assert fields["property_address"] is None
    assert fields["page_number"] is None
    assert transport.calls[0]["url"] == "http://127.0.0.1:1234/v1/chat/completions"


def test_parse_response_strips_code_fences() -> None:
    response = chat_response(f"```json\n{fields_payload(loan_number='999')}\n```")
    fields = parse_extraction_response(response, 2)
    assert fields["loan_number"] == "999"


def test_parse_response_falls_back_to_reasoning_content() -> None:
    response = chat_response(
        content="",
        reasoning_content=f"Let me think... {fields_payload(borrower_name='Jane Doe')}",
    )
    fields = parse_extraction_response(response, 3)
    assert fields["borrower_name"] == "Jane Doe"


def test_parse_response_requires_choices() -> None:
    with pytest.raises(RuntimeError, match="no 'choices'"):
        parse_extraction_response({}, 1)


def test_parse_response_rejects_truncated_response() -> None:
    response = chat_response(fields_payload(), finish_reason="length")
    with pytest.raises(RuntimeError, match="truncated"):
        parse_extraction_response(response, 1)


def test_parse_response_rejects_non_json_content() -> None:
    response = chat_response("I could not find any fields.")
    with pytest.raises(RuntimeError, match="no JSON object"):
        parse_extraction_response(response, 1)


def test_parse_response_rejects_missing_fields() -> None:
    response = chat_response(json.dumps({"borrower_name": "Jane Doe"}))
    with pytest.raises(RuntimeError, match="wrong fields"):
        parse_extraction_response(response, 1)


def test_parse_response_rejects_non_string_value() -> None:
    response = chat_response(fields_payload(loan_number=12345))
    with pytest.raises(RuntimeError, match="non-string loan_number"):
        parse_extraction_response(response, 1)


def test_parse_response_rejects_empty_string() -> None:
    response = chat_response(fields_payload(borrower_name="   "))
    with pytest.raises(RuntimeError, match="empty borrower_name"):
        parse_extraction_response(response, 1)


def test_normalize_for_match_handles_diacritics_and_dashes() -> None:
    assert normalize_for_match("Alya Renard—Van Merçer") == "alya renard van mercer"
    assert normalize_for_match("LOAN #: 20-414-784\n") == "loan 20 414 784"


def test_best_window_similarity_exact_and_fuzzy() -> None:
    ocr_norm = normalize_for_match(
        "604 N Crest view Hill Dr Unit 1144, Las Vegas, NV 89139"
    )
    exact = best_window_similarity(
        normalize_for_match("Las Vegas, NV 89139"), ocr_norm
    )
    assert exact == 1.0
    fuzzy = best_window_similarity(
        normalize_for_match("604 N Crestview Hill Dr Unit 1144"), ocr_norm
    )
    assert fuzzy >= DEFAULT_MISMATCH_THRESHOLD
    assert (
        best_window_similarity(normalize_for_match("9999 Unknown Road"), ocr_norm)
        < DEFAULT_MISMATCH_THRESHOLD
    )


def test_compare_field_statuses() -> None:
    ocr_norm = normalize_for_match("Loan #: 20414784\nBorrower: Alya Renard-Van Mercer")
    assert compare_field(None, ocr_norm, 0.75) == {
        "status": "missing",
        "found_in_ocr": False,
        "similarity": None,
    }
    assert compare_field("20414784", ocr_norm, 0.75)["status"] == "ok"
    assert (
        compare_field("Alya Renard-Van Mercer", ocr_norm, 0.75)["similarity"] == 1.0
    )
    mismatch = compare_field("Someone Else", ocr_norm, 0.75)
    assert mismatch["status"] == "mismatch"
    assert mismatch["found_in_ocr"] is False


def test_verify_extracted_fields_lists_mismatches() -> None:
    fields = {
        "borrower_name": "Jane Doe",
        "property_address": None,
        "loan_number": "12345",
        "page_number": "1",
    }
    verification, mismatches = verify_extracted_fields(
        fields, PAGE_ONE_OCR_TEXT, DEFAULT_MISMATCH_THRESHOLD
    )
    assert mismatches == []
    assert verification["property_address"]["status"] == "missing"
    assert all(
        verification[field]["status"] == "ok"
        for field in ("borrower_name", "loan_number", "page_number")
    )


def test_load_ocr_document_reads_pages(tmp_path: Path) -> None:
    path = write_ocr_json(
        tmp_path / "a.json", PDF_NAME, [(1, "first"), (2, "second")]
    )
    document = load_ocr_document(path)
    assert document.source == PDF_NAME
    assert [page.page_number for page in document.pages] == [1, 2]
    assert document.pages[1].text == "second"
    assert document.pages[0].mean_score == 0.9


def test_load_ocr_document_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_ocr_document(tmp_path / "missing.json")


def test_load_ocr_document_non_json_suffix_raises(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match=r"\.json suffix"):
        load_ocr_document(path)


def test_run_extraction_requires_matching_source(
    sample_pdf: Path, tmp_path: Path
) -> None:
    ocr_path = write_ocr_json(
        tmp_path / "a.json", "other.pdf", [(1, PAGE_ONE_OCR_TEXT), (2, PAGE_TWO_OCR_TEXT)]
    )
    client = make_client(FakeTransport([]))
    with pytest.raises(ValueError, match="does not match PDF"):
        run_extraction(sample_pdf, ocr_path, client)


def test_run_extraction_requires_matching_page_count(
    sample_pdf: Path, tmp_path: Path
) -> None:
    ocr_path = write_ocr_json(tmp_path / "a.json", PDF_NAME, [(1, PAGE_ONE_OCR_TEXT)])
    client = make_client(FakeTransport([]))
    with pytest.raises(ValueError, match="Page count mismatch"):
        run_extraction(sample_pdf, ocr_path, client)


def test_run_extraction_end_to_end(sample_pdf: Path, tmp_path: Path) -> None:
    ocr_path = write_ocr_json(
        tmp_path / "a.json",
        PDF_NAME,
        [(1, PAGE_ONE_OCR_TEXT), (2, PAGE_TWO_OCR_TEXT)],
    )
    transport = FakeTransport(
        [
            chat_response(
                fields_payload(
                    borrower_name="Jane Doe",
                    property_address="1 Main St, Las Vegas, NV",
                    loan_number="12345",
                    page_number="1",
                )
            ),
            chat_response(fields_payload(borrower_name="John Smith")),
        ]
    )
    client = make_client(transport)
    report = run_extraction(sample_pdf, ocr_path, client, dpi=DEFAULT_DPI)
    json.dumps(report, ensure_ascii=False)
    assert report["schema_version"] == "1.0"
    assert report["input_files"]["pdf"] == str(sample_pdf)
    assert report["model"]["thinking"] is False
    assert report["parameters"]["reasoning_effort"] == "none"
    assert len(report["pages"]) == 2
    first, second = report["pages"]
    assert first["page_number"] == 1
    assert first["page_ref"] == f"{PDF_NAME}#page-1"
    assert first["mismatches"] == []
    assert first["extracted"]["page_number"] == "1"
    assert first["verification"]["property_address"]["status"] == "ok"
    assert first["ocr_reference"]["mean_score"] == 0.9
    assert second["mismatches"] == ["borrower_name"]
    assert second["verification"]["borrower_name"]["status"] == "mismatch"
    summary = report["summary"]
    assert summary["page_count"] == 2
    assert summary["field_mismatch_count"] == 1
    assert summary["field_missing_count"] == 3
    assert summary["pages_with_mismatch"] == [2]
    assert summary["needs_review_page_count"] == 1
    assert summary["field_status_counts"]["borrower_name"] == {
        "ok": 1,
        "mismatch": 1,
        "missing": 0,
    }
    assert report["timings"]["vlm_requests"] == 2
    assert report["timings"]["vlm_seconds"] >= 0
    assert report["timings"]["avg_vlm_seconds_per_page"] is not None
    for page in report["pages"]:
        assert page["timings_seconds"]["render_seconds"] > 0
        assert page["timings_seconds"]["total_seconds"] > 0
