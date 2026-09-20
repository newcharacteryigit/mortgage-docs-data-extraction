from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest

from ocr_pdf import (
    DEFAULT_DPI,
    DETECTION_MODEL_NAME,
    ENGINE_NAME,
    RECOGNITION_MODEL_NAME,
    default_output_path,
    ocr_pdf,
)


@pytest.fixture(scope="module")
def sample_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("pdf") / "mortgage_note.pdf"
    document = pymupdf.open()
    first = document.new_page()
    first.insert_text((72, 100), "MORTGAGE NOTE", fontsize=18)
    first.insert_text((72, 140), "Loan Amount: 250,000.00", fontsize=14)
    first.insert_text((72, 170), "Interest Rate: 6.25", fontsize=14)
    second = document.new_page()
    second.insert_text((72, 100), "Borrower: Jane Doe", fontsize=14)
    document.save(path)
    document.close()
    return path


def test_ocr_pdf_extracts_text_and_timing(sample_pdf: Path) -> None:
    result = ocr_pdf(sample_pdf, dpi=DEFAULT_DPI)
    assert result["source"] == str(sample_pdf)
    assert result["engine"] == ENGINE_NAME
    assert result["det_model"] == DETECTION_MODEL_NAME
    assert result["rec_model"] == RECOGNITION_MODEL_NAME
    assert len(result["pages"]) == 2
    assert result["total_duration_seconds"] > 0
    first = result["pages"][0]
    assert first["page_number"] == 1
    assert first["duration_seconds"] > 0
    assert first["mean_score"] is not None
    assert first["mean_score"] > 0.5
    compact = first["text"].replace(" ", "")
    assert "MORTGAGE" in first["text"].upper()
    assert "250" in compact
    assert "000" in compact
    assert "6.25" in compact
    second_text = result["pages"][1]["text"].upper()
    assert "BORROWER" in second_text
    assert "JANE" in second_text
    json.dumps(result, ensure_ascii=False)


def test_default_output_path(tmp_path: Path) -> None:
    assert default_output_path(tmp_path / "document.pdf") == tmp_path / "document.json"


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        ocr_pdf(tmp_path / "missing.pdf")


def test_non_pdf_extension_raises_value_error(tmp_path: Path) -> None:
    path = tmp_path / "document.txt"
    path.write_text("not a pdf", encoding="utf-8")
    with pytest.raises(ValueError, match=r"\.pdf suffix"):
        ocr_pdf(path)


def test_corrupt_pdf_raises_value_error(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.pdf"
    path.write_bytes(b"this is not a pdf")
    with pytest.raises(ValueError, match="Failed to open PDF"):
        ocr_pdf(path)
