from __future__ import annotations

import json
from pathlib import Path

import pytest

import pipeline
from field_normalizer import CANONICALIZATION_VERSION
from pipeline import (
    build_documents,
    build_loan_level_fields,
    build_page_labels,
    build_result,
    build_timings,
    run_pipeline,
)

CATEGORY_LABELS = {
    "mortgage_closing_disclosure_seller": "Mortgage - Closing Disclosure - Seller",
    "lender_rate_note": "Lender - Rate Note",
    "title_rider": "Title - Rider",
    "property_tax_record_information_sheet": "Property - Tax Record Information Sheet",
    "title_signature_name_affidavit_ack": "Title - Signature / Name Affidavit (Ack)",
    "other": "Other / Unclassified",
}
EXTRACTION_FIELDS = ("borrower_name", "property_address", "loan_number", "page_number")
PAGE_ONE_ADDRESS = "604N C restview Hill Dr, Unit 1144 Las Vegas, NV 89139"
SPLIT_ADDRESS = "604 N Crest View Hill Dr Unit 1144, Las Vegas, NV 89139"
CLEAN_ADDRESS = "604 N Crestview Hill Dr Unit 1144, Las Vegas, NV 89139"
ANNOTATED_ADDRESS = (
    "604 N Crestview Hill Dr Unit 1144, Las Vegas, NV 89139 [Property Address]"
)
MAILING_ADDRESS = "4124 Silvercrest Avenue, Las Vegas, NV 89129"


def make_ocr_payload(
    page_texts: dict[int, str], page_seconds: dict[int, float] | None = None
) -> dict:
    seconds = page_seconds or {}
    pages = [
        {
            "page_number": number,
            "text": text,
            "line_count": len(text.splitlines()),
            "mean_score": 0.95,
            "duration_seconds": seconds.get(number, 1.0),
        }
        for number, text in sorted(page_texts.items())
    ]
    return {
        "source": "doc.pdf",
        "pages": pages,
        "total_duration_seconds": round(
            sum(page["duration_seconds"] for page in pages), 3
        ),
    }


def make_category_page(
    number: int,
    category: str | None,
    confidence: float | None = 0.9,
    needs_review: bool = False,
    review_reason: str | None = None,
    total_seconds: float = 0.5,
) -> dict:
    return {
        "page_number": number,
        "page_ref": f"doc.pdf#page-{number}",
        "category": category,
        "category_label": CATEGORY_LABELS.get(category) if category else None,
        "confidence": confidence,
        "probabilities": {key: 0.0 for key in CATEGORY_LABELS},
        "needs_review": needs_review,
        "review_reason": review_reason,
        "ocr_reference": {"line_count": 1, "mean_score": 0.95, "duration_seconds": 1.0},
        "timings_seconds": {
            "classify_seconds": total_seconds,
            "total_seconds": total_seconds,
        },
        "usage": None,
    }


def make_categories_payload(pages: list[dict], total_seconds: float = 2.0) -> dict:
    return {
        "categories": dict(CATEGORY_LABELS),
        "pages": pages,
        "timings": {"total_seconds": total_seconds},
    }


def make_verifications_payload(pages: list[dict], total_seconds: float = 0.3) -> dict:
    return {
        "categories": dict(CATEGORY_LABELS),
        "pages": pages,
        "timings": {"total_seconds": total_seconds},
    }


def make_vlm_page(
    number: int,
    extracted: dict,
    statuses: dict[str, str] | None = None,
    mismatches: tuple[str, ...] = (),
    total_seconds: float = 0.2,
    review_flags: dict[str, bool] | None = None,
    similarities: dict[str, float | None] | None = None,
) -> dict:
    explicit = statuses or {}
    flags = review_flags or {}
    overrides = similarities or {}
    verification = {}
    for field in EXTRACTION_FIELDS:
        value = extracted.get(field)
        status = explicit.get(field, "ok" if value is not None else "missing")
        verification[field] = {
            "status": status,
            "found_in_ocr": status == "ok",
            "similarity": overrides.get(field, 1.0 if status == "ok" else None),
            "match_mode": "exact" if status == "ok" else None,
            "needs_review": flags.get(field, False),
        }
    return {
        "page_number": number,
        "page_ref": f"doc.pdf#page-{number}",
        "extracted": extracted,
        "normalized": {},
        "verification": verification,
        "mismatches": list(mismatches),
        "ocr_reference": {"line_count": 1, "mean_score": 0.95, "duration_seconds": 1.0},
        "timings_seconds": {
            "render_seconds": 0.01,
            "vlm_seconds": total_seconds,
            "compare_seconds": 0.001,
            "total_seconds": total_seconds,
        },
    }


def fields(
    borrower_name: str | None = None,
    property_address: str | None = None,
    loan_number: str | None = None,
    page_number: str | None = None,
) -> dict:
    return {
        "borrower_name": borrower_name,
        "property_address": property_address,
        "loan_number": loan_number,
        "page_number": page_number,
    }


def make_vlm_payload(pages: list[dict], total_seconds: float = 4.0) -> dict:
    return {"pages": pages, "timings": {"total_seconds": total_seconds}}


def make_matches_payload(
    group_page_lists: list[list[int]],
    unique_pages: tuple[int, ...] = (),
    skipped_pages: tuple[int, ...] = (),
    total_seconds: float = 1.0,
) -> dict:
    groups = [
        {"group_id": index, "members": [{"page_number": number} for number in pages]}
        for index, pages in enumerate(group_page_lists, start=1)
    ]
    return {
        "groups": groups,
        "unique_pages": [{"page_number": number} for number in unique_pages],
        "skipped_pages": [{"page_number": number} for number in skipped_pages],
        "timings": {"total_seconds": total_seconds},
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_page_labels_maps_category_confidence_and_notes() -> None:
    categories = {
        1: make_category_page(1, "lender_rate_note", confidence=0.9876),
        2: make_category_page(
            2, None, confidence=None, needs_review=True, review_reason="empty text"
        ),
        3: make_category_page(
            3, "title_rider", confidence=0.4, needs_review=True, review_reason="low confidence"
        ),
    }
    vlm = {
        1: make_vlm_page(1, fields()),
        2: make_vlm_page(2, fields()),
        3: make_vlm_page(3, fields(), mismatches=("loan_number", "borrower_name")),
    }
    labels = build_page_labels([1, 2, 3], categories, vlm, CATEGORY_LABELS)
    assert labels[0] == {
        "page_number": 1,
        "label": "lender_rate_note",
        "confidence": 0.9876,
        "notes": "",
    }
    assert labels[1] == {
        "page_number": 2,
        "label": "other",
        "confidence": None,
        "notes": "empty text",
    }
    assert labels[2]["label"] == "title_rider"
    assert labels[2]["notes"] == "low confidence; vlm mismatch: loan_number, borrower_name"


def test_build_page_labels_rejects_unknown_category() -> None:
    categories = {1: make_category_page(1, "not_a_category")}
    vlm = {1: make_vlm_page(1, fields())}
    with pytest.raises(ValueError, match="unknown category"):
        build_page_labels([1], categories, vlm, CATEGORY_LABELS)


def test_build_documents_majority_and_page_order() -> None:
    labels = [
        {"page_number": 1, "label": "title_rider", "confidence": 1.0, "notes": ""},
        {"page_number": 2, "label": "title_rider", "confidence": 1.0, "notes": ""},
        {"page_number": 3, "label": "lender_rate_note", "confidence": 1.0, "notes": ""},
        {"page_number": 4, "label": "other", "confidence": None, "notes": "empty text"},
    ]
    matches = make_matches_payload([[1, 2, 3]], unique_pages=(4,))
    documents = build_documents(matches, labels, [1, 2, 3, 4])
    assert documents == [
        {"label": "title_rider", "pages": [1, 2, 3]},
        {"label": "other", "pages": [4]},
    ]


def test_build_documents_tie_uses_first_page_label() -> None:
    labels = [
        {"page_number": 1, "label": "title_rider", "confidence": 1.0, "notes": ""},
        {"page_number": 2, "label": "lender_rate_note", "confidence": 1.0, "notes": ""},
    ]
    documents = build_documents(make_matches_payload([[1, 2]]), labels, [1, 2])
    assert documents == [{"label": "title_rider", "pages": [1, 2]}]


def test_build_documents_requires_full_coverage() -> None:
    labels = [
        {"page_number": 1, "label": "title_rider", "confidence": 1.0, "notes": ""},
        {"page_number": 2, "label": "title_rider", "confidence": 1.0, "notes": ""},
    ]
    with pytest.raises(ValueError, match="cover all OCR pages"):
        build_documents(make_matches_payload([[1]]), labels, [1, 2])


def test_build_loan_level_fields_majority_exact_match() -> None:
    pages = [
        make_vlm_page(1, fields(borrower_name="Alya X", loan_number="L1")),
        make_vlm_page(2, fields(loan_number="L1")),
        make_vlm_page(3, fields(borrower_name="Alya X", loan_number="L2")),
    ]
    result = build_loan_level_fields(pages)
    assert result["borrower_name"] == {
        "value": "Alya X",
        "canonical_value": "alya x",
        "source_pages": [1, 3],
        "variants": [{"value": "Alya X", "pages": [1, 3]}],
        "needs_review": False,
    }
    assert result["loan_number"] == {
        "value": "L1",
        "canonical_value": "l1",
        "source_pages": [1, 2],
        "variants": [{"value": "L1", "pages": [1, 2]}],
        "needs_review": True,
    }
    assert result["property_address"] == {
        "value": None,
        "canonical_value": None,
        "source_pages": [],
        "variants": [],
        "needs_review": False,
    }


def test_build_loan_level_fields_clusters_address_variants() -> None:
    pages = [
        make_vlm_page(1, fields(property_address=PAGE_ONE_ADDRESS)),
        make_vlm_page(3, fields(property_address=SPLIT_ADDRESS)),
        make_vlm_page(4, fields(property_address=CLEAN_ADDRESS)),
        make_vlm_page(7, fields(property_address=ANNOTATED_ADDRESS)),
        make_vlm_page(11, fields(property_address=MAILING_ADDRESS)),
    ]
    result = build_loan_level_fields(pages)["property_address"]
    assert result["value"] == CLEAN_ADDRESS
    assert result["canonical_value"] == (
        "604 n crestview hill dr unit 1144 las vegas nv 89139"
    )
    assert result["source_pages"] == [1, 3, 4, 7]
    assert result["variants"] == [
        {"value": PAGE_ONE_ADDRESS, "pages": [1]},
        {"value": SPLIT_ADDRESS, "pages": [3]},
        {"value": CLEAN_ADDRESS, "pages": [4]},
        {"value": ANNOTATED_ADDRESS, "pages": [7]},
    ]
    assert result["needs_review"] is False


def test_build_loan_level_fields_tie_uses_earliest_page() -> None:
    pages = [
        make_vlm_page(1, fields(loan_number="L2")),
        make_vlm_page(2, fields(loan_number="L1")),
    ]
    assert build_loan_level_fields(pages)["loan_number"] == {
        "value": "L2",
        "canonical_value": "l2",
        "source_pages": [1],
        "variants": [{"value": "L2", "pages": [1]}],
        "needs_review": True,
    }


def test_build_loan_level_fields_flags_close_runner_up() -> None:
    pages = [
        make_vlm_page(1, fields(loan_number="20414784")),
        make_vlm_page(2, fields(loan_number="20-414-784")),
        make_vlm_page(3, fields(loan_number="99999999")),
    ]
    result = build_loan_level_fields(pages)["loan_number"]
    assert result["value"] == "20414784"
    assert result["source_pages"] == [1, 2]
    assert result["variants"] == [
        {"value": "20414784", "pages": [1]},
        {"value": "20-414-784", "pages": [2]},
    ]
    assert result["needs_review"] is True


def test_build_loan_level_fields_propagates_page_review_flag() -> None:
    pages = [
        make_vlm_page(
            1,
            fields(property_address=CLEAN_ADDRESS),
            review_flags={"property_address": True},
        ),
        make_vlm_page(2, fields(property_address=CLEAN_ADDRESS)),
    ]
    result = build_loan_level_fields(pages)["property_address"]
    assert result["value"] == CLEAN_ADDRESS
    assert result["source_pages"] == [1, 2]
    assert result["needs_review"] is True


def test_build_loan_level_fields_excludes_mismatch() -> None:
    pages = [
        make_vlm_page(1, fields(borrower_name="Real"), statuses={"borrower_name": "ok"}),
        make_vlm_page(
            2, fields(borrower_name="Wrong"), statuses={"borrower_name": "mismatch"}
        ),
    ]
    assert build_loan_level_fields(pages)["borrower_name"] == {
        "value": "Real",
        "canonical_value": "real",
        "source_pages": [1],
        "variants": [{"value": "Real", "pages": [1]}],
        "needs_review": True,
    }


def test_build_loan_level_fields_returns_null_when_only_mismatch() -> None:
    pages = [
        make_vlm_page(
            1, fields(borrower_name="Wrong"), statuses={"borrower_name": "mismatch"}
        )
    ]
    assert build_loan_level_fields(pages)["borrower_name"] == {
        "value": None,
        "canonical_value": None,
        "source_pages": [],
        "variants": [],
        "needs_review": False,
    }


def test_build_loan_level_fields_requires_review_flag() -> None:
    page = make_vlm_page(1, fields(loan_number="1"))
    del page["verification"]["loan_number"]["needs_review"]
    with pytest.raises(ValueError, match="needs_review"):
        build_loan_level_fields([page])


def test_build_loan_level_fields_rejects_invalid_similarity() -> None:
    page = make_vlm_page(1, fields(loan_number="1"), similarities={"loan_number": 1.5})
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        build_loan_level_fields([page])


def test_build_timings_sums_pages_and_stages() -> None:
    ocr = make_ocr_payload({1: "text", 2: ""}, page_seconds={1: 1.5, 2: 0.5})
    categories = make_categories_payload(
        [
            make_category_page(1, "lender_rate_note", total_seconds=0.4),
            make_category_page(
                2,
                None,
                confidence=None,
                needs_review=True,
                review_reason="empty text",
                total_seconds=0.0,
            ),
        ],
        total_seconds=1.1,
    )
    vlm = make_vlm_payload(
        [
            make_vlm_page(1, fields(), total_seconds=2.5),
            make_vlm_page(2, fields(), total_seconds=0.1),
        ],
        total_seconds=3.0,
    )
    verifications = make_verifications_payload(
        [
            make_category_page(1, "lender_rate_note", total_seconds=0.4),
            make_category_page(
                2,
                None,
                confidence=None,
                needs_review=True,
                review_reason="empty text",
                total_seconds=0.0,
            ),
        ],
        total_seconds=0.6,
    )
    matches = make_matches_payload([[1]], unique_pages=(2,), total_seconds=0.9)
    category_index = {1: categories["pages"][0], 2: categories["pages"][1]}
    vlm_index = {1: vlm["pages"][0], 2: vlm["pages"][1]}
    timings = build_timings(
        ocr,
        categories,
        verifications,
        vlm,
        matches,
        ocr["pages"],
        category_index,
        vlm_index,
        7.5,
    )
    assert timings["stages_seconds"] == {
        "ocr": 2.0,
        "classification": 1.1,
        "verification": 0.6,
        "vlm": 3.0,
        "matching": 0.9,
    }
    assert timings["total_seconds"] == 7.5
    assert timings["pages"][0] == {
        "page_number": 1,
        "ocr_seconds": 1.5,
        "classification_seconds": 0.4,
        "vlm_seconds": 2.5,
        "total_seconds": 4.4,
    }
    assert timings["pages"][1]["total_seconds"] == 0.6


def test_build_result_rejects_missing_category_pages() -> None:
    ocr = make_ocr_payload({1: "text", 2: "text"})
    categories = make_categories_payload([make_category_page(1, "title_rider")])
    verifications = make_verifications_payload(
        [make_category_page(1, "title_rider"), make_category_page(2, "title_rider")]
    )
    vlm = make_vlm_payload(
        [make_vlm_page(1, fields()), make_vlm_page(2, fields())]
    )
    matches = make_matches_payload([[1, 2]])
    with pytest.raises(ValueError, match="do not match the OCR pages"):
        build_result(ocr, categories, verifications, vlm, matches, 1.0)


def test_build_result_rejects_missing_verification_pages() -> None:
    ocr = make_ocr_payload({1: "text", 2: "text"})
    categories = make_categories_payload(
        [make_category_page(1, "title_rider"), make_category_page(2, "title_rider")]
    )
    verifications = make_verifications_payload([make_category_page(1, "title_rider")])
    vlm = make_vlm_payload(
        [make_vlm_page(1, fields()), make_vlm_page(2, fields())]
    )
    matches = make_matches_payload([[1, 2]])
    with pytest.raises(ValueError, match="Verification JSON pages do not match"):
        build_result(ocr, categories, verifications, vlm, matches, 1.0)


def make_fake_runner(fixtures: dict[str, dict], calls: list[str]):
    def runner(command: list[str]) -> None:
        script = Path(command[1]).name
        calls.append(script)
        if script == "ocr_pdf.py":
            write_json(Path(command[2]).with_suffix(".json"), fixtures["ocr"])
        elif script == "page_classifier.py":
            write_json(
                Path(command[2]).with_suffix(".categories.json"), fixtures["categories"]
            )
        elif script == "category_verifier.py":
            categories_path = Path(command[2])
            review_path = categories_path.with_name(
                categories_path.name.replace(
                    pipeline.CATEGORIES_SUFFIX, pipeline.REVIEW_SUFFIX
                )
            )
            write_json(review_path, fixtures["verifications"])
        elif script == "vlm_extract.py":
            write_json(Path(command[2]).with_suffix(".vlm.json"), fixtures["vlm"])
        elif script == "page_matcher.py":
            write_json(Path(command[2]).with_suffix(".matches.json"), fixtures["matches"])
        else:
            raise AssertionError(f"unexpected stage script: {script}")

    return runner


def test_run_pipeline_end_to_end_with_fake_runner(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    fixtures = {
        "ocr": make_ocr_payload({1: "tax text", 2: "", 3: "rider text", 4: "rider text"}),
        "categories": make_categories_payload(
            [
                make_category_page(
                    1, "property_tax_record_information_sheet", confidence=1.0
                ),
                make_category_page(
                    2,
                    None,
                    confidence=None,
                    needs_review=True,
                    review_reason="empty text",
                    total_seconds=0.0,
                ),
                make_category_page(3, "title_rider", confidence=0.8),
                make_category_page(4, "title_rider", confidence=0.9),
            ]
        ),
        "verifications": make_verifications_payload(
            [
                make_category_page(
                    1,
                    "property_tax_record_information_sheet",
                    confidence=1.0,
                    needs_review=True,
                    review_reason="verifier disagreement",
                ),
                make_category_page(
                    2,
                    None,
                    confidence=None,
                    needs_review=True,
                    review_reason="empty text",
                    total_seconds=0.0,
                ),
                make_category_page(3, "title_rider", confidence=0.8),
                make_category_page(4, "title_rider", confidence=0.9),
            ],
            total_seconds=0.3,
        ),
        "vlm": make_vlm_payload(
            [
                make_vlm_page(1, fields(borrower_name="Jane Doe", loan_number="L1")),
                make_vlm_page(2, fields()),
                make_vlm_page(
                    3,
                    fields(
                        borrower_name="Jane Doe",
                        property_address="1 Main St",
                        loan_number="L1",
                    ),
                ),
                make_vlm_page(4, fields(property_address="1 Main St")),
            ]
        ),
        "matches": make_matches_payload([[3, 4]], unique_pages=(1,), skipped_pages=(2,)),
    }
    calls: list[str] = []
    commands: list[list[str]] = []
    fake_runner = make_fake_runner(fixtures, calls)

    def runner(command: list[str]) -> None:
        commands.append(list(command))
        fake_runner(command)

    report = run_pipeline(pdf_path, runner=runner)
    assert calls == [
        "ocr_pdf.py",
        "page_classifier.py",
        "category_verifier.py",
        "vlm_extract.py",
        "page_matcher.py",
    ]
    assert report["schema_version"] == "2.1"
    assert report["canonicalization_version"] == CANONICALIZATION_VERSION
    assert [entry["label"] for entry in report["page_labels"]] == [
        "property_tax_record_information_sheet",
        "other",
        "title_rider",
        "title_rider",
    ]
    assert report["page_labels"][0]["notes"] == "verifier disagreement"
    assert report["page_labels"][1]["confidence"] is None
    assert report["page_labels"][1]["notes"] == "empty text"
    assert report["documents"] == [
        {"label": "property_tax_record_information_sheet", "pages": [1]},
        {"label": "other", "pages": [2]},
        {"label": "title_rider", "pages": [3, 4]},
    ]
    assert report["loan_level_fields"]["borrower_name"] == {
        "value": "Jane Doe",
        "canonical_value": "doe jane",
        "source_pages": [1, 3],
        "variants": [{"value": "Jane Doe", "pages": [1, 3]}],
        "needs_review": False,
    }
    assert report["loan_level_fields"]["property_address"] == {
        "value": "1 Main St",
        "canonical_value": "1 main st",
        "source_pages": [3, 4],
        "variants": [{"value": "1 Main St", "pages": [3, 4]}],
        "needs_review": False,
    }
    assert report["loan_level_fields"]["loan_number"] == {
        "value": "L1",
        "canonical_value": "l1",
        "source_pages": [1, 3],
        "variants": [{"value": "L1", "pages": [1, 3]}],
        "needs_review": False,
    }
    assert report["timings"]["stages_seconds"] == {
        "ocr": 4.0,
        "classification": 2.0,
        "verification": 0.3,
        "vlm": 4.0,
        "matching": 1.0,
    }
    assert report["timings"]["total_seconds"] > 0
    assert len(report["timings"]["pages"]) == 4
    matcher_command = commands[-1]
    assert Path(matcher_command[1]).name == "page_matcher.py"
    assert "--categories-json" in matcher_command
    assert matcher_command[-1] == str(tmp_path / "doc.category_review.json")
    json.dumps(report)


def test_run_pipeline_missing_pdf(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="PDF file not found"):
        run_pipeline(tmp_path / "missing.pdf", runner=lambda command: None)


def test_run_pipeline_rejects_non_pdf(tmp_path: Path) -> None:
    path = tmp_path / "doc.txt"
    path.write_text("not a pdf", encoding="utf-8")
    with pytest.raises(ValueError, match=r"\.pdf suffix"):
        run_pipeline(path, runner=lambda command: None)


def test_run_pipeline_fails_when_stage_fails(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    def runner(command: list[str]) -> None:
        if Path(command[1]).name == "page_classifier.py":
            raise RuntimeError("stage exploded")

    with pytest.raises(RuntimeError, match="stage exploded"):
        run_pipeline(pdf_path, runner=runner)


def test_main_writes_result_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    report = {
        "page_labels": [],
        "documents": [],
        "loan_level_fields": {},
        "timings": {"total_seconds": 0.0, "stages_seconds": {}, "pages": []},
    }
    monkeypatch.setattr(pipeline, "run_pipeline", lambda pdf_path: report)
    out_path = tmp_path / "custom-result.json"
    assert pipeline.main([str(pdf_path), "--out", str(out_path)]) == 0
    assert json.loads(out_path.read_text(encoding="utf-8")) == report


def test_main_returns_failure_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    def failing(pdf_path: Path) -> dict:
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline, "run_pipeline", failing)
    assert pipeline.main([str(pdf_path)]) == 1
    assert "ERROR: RuntimeError: boom" in capsys.readouterr().err
