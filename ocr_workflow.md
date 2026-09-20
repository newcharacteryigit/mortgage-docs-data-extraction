# OCR Workflow

`ocr_pdf.py` converts a PDF file into page-level OCR text and writes a JSON result file.
It runs fully offline on CPU with PaddleOCR (PP-OCRv6 small models) and PyMuPDF.

## Pipeline

1. **Validate input** — the path must exist, end with `.pdf`, open with PyMuPDF, and contain at least one page. Any violation fails fast with a clear error.
2. **Render pages** — each page is rasterized at a fixed DPI (`200` by default) into an RGB image and converted to BGR for PaddleOCR. Pages are streamed one at a time, so memory stays flat for long documents.
3. **Load OCR engine once** — a lazily created singleton runs `PP-OCRv6_small_det` (text detection) and `PP-OCRv6_small_rec` (text recognition). Document orientation classification and unwarping are disabled; text-line orientation is enabled.
4. **OCR each page** — detected text lines are joined with newlines; the pipeline also reports the line count and the mean recognition confidence per page.
5. **Measure time** — per-page OCR duration and total wall-clock duration are recorded.
6. **Write JSON** — the result goes to a file (default) or to stdout, and progress/errors are printed to stderr.

## Usage

```bash
python ocr_pdf.py document.pdf
python ocr_pdf.py document.pdf --out result.json
python ocr_pdf.py document.pdf --dpi 300
python ocr_pdf.py document.pdf --stdout
```

| Flag | Meaning |
| --- | --- |
| `pdf` | PDF file to OCR (required) |
| `--dpi` | Render resolution, default `200` |
| `--out` | JSON output path; default is the PDF path with a `.json` suffix |
| `--stdout` | Print JSON to stdout instead of writing a file (overrides `--out`) |

Offline note: the first run downloads the ~30 MB PP-OCRv6 small models into `~/.paddlex/official_models`. Later runs work without a network connection.

## Outputs

### Terminal (stderr)

Progress and errors never pollute the JSON stream:

```text
[ocr] file=document.pdf dpi=200 engine=paddleocr det=PP-OCRv6_small_det rec=PP-OCRv6_small_rec device=cpu
[ocr] page 1 done duration=1.60s lines=3 mean_score=0.9966
[ocr] page 2 done duration=0.98s lines=1 mean_score=0.9728
[ocr] total duration=4.14s pages=2 avg_per_page=2.07s
[ocr] JSON written to: document.json
```

On failure the last line is `ERROR: <ExceptionType>: <message>` (for example `ERROR: FileNotFoundError: PDF file not found: missing.pdf`) and the process exits with code `1`. Success exits with code `0`.

### JSON file

```json
{
  "source": "document.pdf",
  "engine": "paddleocr",
  "det_model": "PP-OCRv6_small_det",
  "rec_model": "PP-OCRv6_small_rec",
  "dpi": 200,
  "device": "cpu",
  "total_duration_seconds": 4.142,
  "pages": [
    {
      "page_number": 1,
      "text": "MORTGAGE NOTE\nLoan Amount: 250,000.00\nInterest Rate: 6.25",
      "line_count": 3,
      "mean_score": 0.9966,
      "duration_seconds": 1.601
    }
  ]
}
```

| Field | Description |
| --- | --- |
| `source` | Input PDF path as given |
| `engine` | OCR engine name |
| `det_model` / `rec_model` | Detection and recognition model names |
| `dpi` | Render resolution used |
| `device` | Paddle device (`cpu`) |
| `total_duration_seconds` | Full pipeline time, including one-time engine load |
| `pages[].page_number` | 1-based page index |
| `pages[].text` | OCR text, lines joined with `\n` |
| `pages[].line_count` | Number of detected text lines |
| `pages[].mean_score` | Mean recognition confidence (0-1), `null` when no text is found |
| `pages[].duration_seconds` | OCR time for this page |

## Behavior Guarantees

- **Deterministic**: fixed page order, fixed DPI, greedy CTC decoding without sampling.
- **Fail fast**: missing files, non-PDF inputs, unreadable or empty PDFs, and per-page OCR failures raise an error that reaches the terminal; nothing is swallowed.
- **Isolated streams**: JSON goes to stdout only with `--stdout`; all logs go to stderr.

## Verification

```bash
python -m pytest tests/ -q
```

The test suite creates a synthetic two-page mortgage PDF with PyMuPDF and checks text and numeric extraction, timing fields, JSON serializability, and the three error paths (missing file, non-PDF suffix, corrupt PDF).

## Current Limitations

- Every page is OCR'd; there is no digital text-layer fast path yet.
- Output is plain text per page: no word boxes, no tables, no key-value fields.
- `mean_score` is a page-level signal only; there is no `needs_review` flag yet.
