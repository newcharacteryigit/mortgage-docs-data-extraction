# Mortgage Document Analysis Pipeline

This project turns a mortgage PDF into structured, reviewable data. It combines offline OCR, page classification, optional second-opinion verification, local vision-language extraction, and adjacent-page matching into a deterministic processing pipeline.

The pipeline produces both page-level evidence and loan-level fields such as the borrower name, property address, and loan number. Every stage writes JSON with model details, thresholds, timings, and review flags so results can be inspected and reproduced.

## Pipeline

`pipeline.py` runs these stages in order:

1. **OCR**: renders each PDF page and extracts text with PaddleOCR on the CPU.
2. **Classification**: assigns each page to one of six mortgage document categories using the TypeSafe Jev API.
3. **Verification**: sends low-confidence or low-margin classifications to a local LM Studio model for a second opinion.
4. **Field extraction**: uses a local vision-language model to extract `borrower_name`, `property_address`, `loan_number`, and the printed `page_number`.
5. **Matching**: groups consecutive pages with local text embeddings, using category and printed-page-number signals when available.
6. **Aggregation**: combines verified page values into loan-level fields and flags uncertain results for review.

Each stage can also be run independently. The corresponding workflow files document their input and output schemas in more detail.

## Requirements

- Python 3.11 or newer
- The packages pinned in `requirements.txt`
- A TypeSafe API key for page classification
- LM Studio for verification, VLM extraction, and embeddings

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Configure the TypeSafe key:

```bash
copy .env.example .env
```

Then set `TYPESAFE_API_KEY` in `.env`. The key can also be provided as an environment variable.

Before running the complete pipeline, configure LM Studio with:

- `google/gemma-4-e2b` for category verification
- `qwen/qwen3.5-9b` for vision-language extraction
- `jinaai/jina-embeddings-v5-text-small-text-matching` as an embedding model

The default LM Studio endpoint is `http://127.0.0.1:1234/v1`.

## Usage

Run the complete workflow for a PDF:

```bash
python pipeline.py AREAL_LOAN.pdf
```

The command creates these reports next to the PDF:

```text
AREAL_LOAN.json
AREAL_LOAN.categories.json
AREAL_LOAN.category_review.json
AREAL_LOAN.vlm.json
AREAL_LOAN.matches.json
AREAL_LOAN.result.json
```

Run an individual stage when needed:

```bash
python ocr_pdf.py document.pdf
python page_classifier.py document.json
python category_verifier.py document.categories.json
python vlm_extract.py document.pdf --ocr document.json
python page_matcher.py document.json
```

Use `--stdout` on supported commands to emit JSON to standard output. Progress and errors are written to standard error, keeping the JSON stream clean.

## Output

The final `.result.json` report contains:

- page labels and classification notes
- grouped document sections
- aggregated loan-level fields with observed variants and source pages
- review flags for mismatches, weak evidence, or close competing values
- stage and per-page timings

The system fails fast on invalid inputs and malformed model responses. Missing or uncertain values are not guessed; they remain empty or are marked for review.

## Testing

Run the full test suite:

```bash
python -m pytest tests/ -q
```

The tests use synthetic inputs and fake HTTP transports where appropriate, so the test suite does not require a live TypeSafe or LM Studio service.

## Project Layout

| File | Purpose |
| --- | --- |
| `pipeline.py` | Runs all stages and builds the final report |
| `ocr_pdf.py` | PDF rendering and page-level OCR |
| `page_classifier.py` | TypeSafe-based page categorization |
| `category_verifier.py` | Local second-opinion category verification |
| `vlm_extract.py` | Image-based field extraction and OCR verification |
| `page_matcher.py` | Consecutive-page similarity and grouping |
| `field_normalizer.py` | Shared canonicalization and similarity rules |
| `tests/` | Unit and integration tests |
