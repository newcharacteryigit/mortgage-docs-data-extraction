from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator, Sequence
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

import numpy as np
import pymupdf
from numpy.typing import NDArray
from paddleocr import PaddleOCR

import paddle

DEFAULT_DPI = 200
DETECTION_MODEL_NAME = "PP-OCRv6_small_det"
RECOGNITION_MODEL_NAME = "PP-OCRv6_small_rec"
ENGINE_NAME = "paddleocr"
POINTS_PER_INCH = 72.0
PDF_SUFFIX = ".pdf"
DEFAULT_OUTPUT_SUFFIX = ".json"
STDIO_ENCODING = "utf-8"
CHANNELS_RGB = 3
CHANNELS_RGBA = 4


class PageResult(TypedDict):
    page_number: int
    text: str
    line_count: int
    mean_score: float | None
    duration_seconds: float


class OcrDocument(TypedDict):
    source: str
    engine: str
    det_model: str
    rec_model: str
    dpi: int
    device: str
    total_duration_seconds: float
    pages: list[PageResult]


def log_message(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding=STDIO_ENCODING, errors="replace")


def device_name() -> str:
    return paddle.device.get_device()


def default_output_path(pdf_path: Path) -> Path:
    return pdf_path.parent / f"{pdf_path.stem}{DEFAULT_OUTPUT_SUFFIX}"


def open_pdf(pdf_path: Path) -> pymupdf.Document:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if pdf_path.suffix.lower() != PDF_SUFFIX:
        raise ValueError(f"File does not have a .pdf suffix: {pdf_path}")
    try:
        document = pymupdf.open(pdf_path)
    except Exception as exc:
        raise ValueError(f"Failed to open PDF ({pdf_path}): {exc}") from exc
    if document.page_count == 0:
        document.close()
        raise ValueError(f"PDF contains no pages: {pdf_path}")
    return document


def render_pdf_pages(
    document: pymupdf.Document, dpi: int
) -> Iterator[tuple[int, NDArray[np.uint8]]]:
    if dpi <= 0:
        raise ValueError(f"DPI must be positive, got: {dpi}")
    scale = dpi / POINTS_PER_INCH
    matrix = pymupdf.Matrix(scale, scale)
    for page_index in range(document.page_count):
        page = document.load_page(page_index)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, pixmap.n
        )
        if array.shape[2] == CHANNELS_RGBA:
            array = array[:, :, :CHANNELS_RGB]
        if array.ndim != 3 or array.shape[2] != CHANNELS_RGB:
            raise RuntimeError(
                f"Unexpected image shape on page {page_index + 1}: {array.shape}"
            )
        bgr = np.flip(array, axis=2)
        yield page_index + 1, np.ascontiguousarray(bgr)


@lru_cache(maxsize=1)
def _build_engine() -> PaddleOCR:
    return PaddleOCR(
        text_detection_model_name=DETECTION_MODEL_NAME,
        text_recognition_model_name=RECOGNITION_MODEL_NAME,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
    )


def ocr_image(engine: PaddleOCR, image: NDArray[np.uint8]) -> tuple[str, int, float | None]:
    results = engine.predict(image)
    lines: list[str] = []
    scores: list[float] = []
    for result in results:
        lines.extend(str(text) for text in result["rec_texts"])
        scores.extend(float(score) for score in result["rec_scores"])
    mean_score = float(np.mean(scores)) if scores else None
    return "\n".join(lines), len(lines), mean_score


def ocr_pdf(pdf_path: Path, dpi: int = DEFAULT_DPI) -> OcrDocument:
    document = open_pdf(pdf_path)
    device = device_name()
    log_message(
        f"[ocr] file={pdf_path} dpi={dpi} engine={ENGINE_NAME} "
        f"det={DETECTION_MODEL_NAME} rec={RECOGNITION_MODEL_NAME} device={device}"
    )
    started = time.perf_counter()
    engine = _build_engine()
    pages: list[PageResult] = []
    try:
        for page_number, image in render_pdf_pages(document, dpi):
            page_started = time.perf_counter()
            try:
                text, line_count, mean_score = ocr_image(engine, image)
            except Exception as exc:
                raise RuntimeError(f"OCR failed on page {page_number}: {exc}") from exc
            duration = time.perf_counter() - page_started
            pages.append(
                {
                    "page_number": page_number,
                    "text": text,
                    "line_count": line_count,
                    "mean_score": mean_score,
                    "duration_seconds": round(duration, 3),
                }
            )
            score_text = f"{mean_score:.4f}" if mean_score is not None else "n/a"
            log_message(
                f"[ocr] page {page_number} done duration={duration:.2f}s "
                f"lines={line_count} mean_score={score_text}"
            )
    finally:
        document.close()
    total_duration = time.perf_counter() - started
    if pages:
        log_message(
            f"[ocr] total duration={total_duration:.2f}s pages={len(pages)} "
            f"avg_per_page={total_duration / len(pages):.2f}s"
        )
    return {
        "source": str(pdf_path),
        "engine": ENGINE_NAME,
        "det_model": DETECTION_MODEL_NAME,
        "rec_model": RECOGNITION_MODEL_NAME,
        "dpi": dpi,
        "device": device,
        "total_duration_seconds": round(total_duration, 3),
        "pages": pages,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run PaddleOCR (CPU) on a PDF file and produce page-level JSON."
    )
    parser.add_argument("pdf", type=Path, help="PDF file to OCR")
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"Render resolution (default: {DEFAULT_DPI})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"JSON output file (default: PDF path with {DEFAULT_OUTPUT_SUFFIX} suffix)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print JSON to stdout instead of writing a file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_stdio()
    args = build_parser().parse_args(argv)
    try:
        result = ocr_pdf(args.pdf, dpi=args.dpi)
    except Exception as exc:
        log_message(f"ERROR: {type(exc).__name__}: {exc}")
        return 1
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.stdout:
        print(payload)
        return 0
    out_path = args.out if args.out is not None else default_output_path(args.pdf)
    try:
        out_path.write_text(payload, encoding="utf-8")
    except OSError as exc:
        log_message(f"ERROR: failed to write JSON file ({out_path}): {exc}")
        return 1
    log_message(f"[ocr] JSON written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
