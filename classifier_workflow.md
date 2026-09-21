# Page Classification Workflow

`page_classifier.py` reads the OCR JSON produced by `ocr_pdf.py` and assigns every page to
one of six fixed mortgage document categories using the TypeSafe Jev model (`choice`
primitive). It writes one JSON report with the category, confidence, full probability
distribution, per-page timings, and token usage.

It is independent from the OCR, matching, and extraction code: it never opens the PDF and
never imports `ocr_pdf`, `page_matcher`, or `vlm_extract`.

## What it does

1. Validates the OCR JSON (object with non-empty `source` and `pages`; positive, strictly
   increasing `page_number`; string `text`; optional numeric `line_count`, `mean_score`,
   and `duration_seconds`).
2. Fails fast if any non-empty page exceeds `--max-state-chars`, before sending requests.
3. Resolves `TYPESAFE_API_KEY` from the environment (wins) or from `.env`.
4. Sends one `POST /v1/systemone` request per non-empty page: `state` is the raw OCR page
   text, the question is a single `choice` named `page_category`.
5. Strictly validates the answer: known category, `type == "choice"`, numeric `confidence`
   in `[0, 1]`, `probabilities` with exactly the six category keys, each in `[0, 1]`,
   summing to 1, peaking on the chosen option. Invalid answers fail the run.
6. Flags pages for review (`needs_review` + `review_reason`):
   - `empty text`: page skipped, no request sent;
   - `low confidence`: `confidence < --confidence-threshold`;
   - `other category`: model chose `other`.
7. Records per-page classification seconds and token usage, plus document totals.
8. Writes JSON to a file or stdout; all progress and errors go to stderr.

## Categories

| Key | Label |
| --- | --- |
| `mortgage_closing_disclosure_seller` | Mortgage - Closing Disclosure - Seller |
| `lender_rate_note` | Lender - Rate Note |
| `title_rider` | Title - Rider |
| `property_tax_record_information_sheet` | Property - Tax Record Information Sheet |
| `title_signature_name_affidavit_ack` | Title - Signature / Name Affidavit (Ack) |
| `other` | Other / Unclassified |

The Choice instructions and criteria live in `page_classifier.py` and are versioned as
`PROMPT_VERSION` (`1.0`). A continuation page without its own title is classified as the
document type it continues.

## Model and API

| Item | Value |
| --- | --- |
| Endpoint | `POST https://api.typesafe.ai/v1/systemone` |
| Authentication | `Authorization: Bearer $TYPESAFE_API_KEY` |
| Model | `--model`, default `jev-latest` (`jev-1.13.0`); pin a version for reproducible runs |
| Input | Text only; the OCR page text is sent as `state` |
| Request shape | One `choice` question `page_category` with six criteria, no sampling parameters |
| Response read | `answers.page_category.choice/confidence/probabilities`, `model`, `usage` |
| Key file | `.env` (see `.env.example`); real key is gitignored |

The report records the requested model (`model.id`) and the versioned model that answered
(`model.response_model`).

## Usage

```bash
copy .env.example .env          # then put a real TYPESAFE_API_KEY in .env
python page_classifier.py AREAL_LOAN.json
python page_classifier.py AREAL_LOAN.json --out result.json
python page_classifier.py AREAL_LOAN.json --confidence-threshold 0.6 --stdout
python page_classifier.py AREAL_LOAN.json --model jev-1.13.0 --timeout 30
```

| Flag | Meaning |
| --- | --- |
| `ocr_json` | OCR JSON produced by `ocr_pdf.py` (required) |
| `--out` | Output path; default is the OCR path with a `.categories.json` suffix |
| `--stdout` | Print JSON to stdout instead of writing a file |
| `--base-url` | TypeSafe API base URL, default `https://api.typesafe.ai/v1` |
| `--model` | Model id or alias, default `jev-latest` |
| `--timeout` | HTTP timeout in seconds, default `60.0` |
| `--confidence-threshold` | Below this confidence a page is `needs_review`, default `0.5` |
| `--max-state-chars` | Fail fast when a page text is longer, default `30000` |
| `--env-file` | File providing `TYPESAFE_API_KEY`, default `.env` |

## Outputs

### Terminal (stderr)

```text
[classify] ocr=AREAL_LOAN.json pages=12 model=jev-latest base_url=https://api.typesafe.ai/v1 threshold=0.5 max_state_chars=30000
[classify] page 1 done category=property_tax_record_information_sheet confidence=0.9812 duration=1.42s review=-
[classify] page 7 skipped reason=empty text
[classify] total duration=14.20s pages=12 requests=12 avg_per_page=1.18s
[classify] needs_review pages=[] categories={...}
[classify] JSON written to: AREAL_LOAN.categories.json
```

Success exits `0`; on failure the last line is `ERROR: <ExceptionType>: <message>` and the
exit code is `1`.

### JSON report (schema `1.0`)

```json
{
  "schema_version": "1.0",
  "input_files": {"ocr_json": "AREAL_LOAN.json"},
  "model": {
    "id": "jev-latest",
    "base_url": "https://api.typesafe.ai/v1",
    "prompt_version": "1.0",
    "response_model": "jev-1.13.0"
  },
  "categories": {
    "mortgage_closing_disclosure_seller": "Mortgage - Closing Disclosure - Seller",
    "lender_rate_note": "Lender - Rate Note",
    "title_rider": "Title - Rider",
    "property_tax_record_information_sheet": "Property - Tax Record Information Sheet",
    "title_signature_name_affidavit_ack": "Title - Signature / Name Affidavit (Ack)",
    "other": "Other / Unclassified"
  },
  "parameters": {"confidence_threshold": 0.5, "max_state_chars": 30000, "timeout_seconds": 60.0},
  "summary": {
    "page_count": 12,
    "classified_page_count": 12,
    "skipped_page_count": 0,
    "needs_review_page_count": 0,
    "needs_review_pages": [],
    "category_counts": {
      "mortgage_closing_disclosure_seller": 2,
      "lender_rate_note": 4,
      "title_rider": 3,
      "property_tax_record_information_sheet": 2,
      "title_signature_name_affidavit_ack": 1,
      "other": 0
    }
  },
  "pages": [
    {
      "page_number": 1,
      "page_ref": "AREAL_LOAN.pdf#page-1",
      "category": "property_tax_record_information_sheet",
      "category_label": "Property - Tax Record Information Sheet",
      "confidence": 0.9812,
      "probabilities": {
        "mortgage_closing_disclosure_seller": 0.0,
        "lender_rate_note": 0.0,
        "title_rider": 0.0,
        "property_tax_record_information_sheet": 0.9812,
        "title_signature_name_affidavit_ack": 0.0188,
        "other": 0.0
      },
      "needs_review": false,
      "review_reason": null,
      "ocr_reference": {"line_count": 82, "mean_score": 0.9819, "duration_seconds": 6.096},
      "timings_seconds": {"classify_seconds": 1.42, "total_seconds": 1.42},
      "usage": {"input_tokens": 1420, "output_tokens": 38}
    }
  ],
  "skipped_pages": [],
  "timings": {"total_seconds": 14.2, "classify_seconds": 14.15, "requests": 12, "avg_classify_seconds_per_page": 1.179},
  "usage": {"input_tokens": 17040, "output_tokens": 456, "requests": 12}
}
```

| Field | Description |
| --- | --- |
| `model.id` / `base_url` | Requested model and endpoint |
| `model.prompt_version` / `response_model` | Instructions version and versioned answerer, `null` when nothing was classified |
| `categories` | Category key to label mapping used in the run |
| `summary.classified_page_count` | Pages sent to the API (excludes skipped empty pages) |
| `summary.category_counts` | Pages per category, including zero-count categories |
| `summary.needs_review_pages` | Page numbers flagged for review (includes skipped pages) |
| `pages[].category` / `category_label` / `confidence` / `probabilities` | Answer, `null` for skipped pages; confidence and probabilities rounded to 4 decimals |
| `pages[].needs_review` / `review_reason` | Review flag and one of `empty text`, `low confidence`, `other category` |
| `pages[].ocr_reference` | OCR line count, mean score, and duration from the input page |
| `pages[].timings_seconds` | Classification seconds and page total seconds |
| `pages[].usage` | Token usage for the page, `null` for skipped pages |
| `skipped_pages` | Empty pages that were not sent to the API |
| `timings` / `usage` | Document totals, request count, average classification seconds, token sums |

## Guarantees and Limits

- **Deterministic**: fixed page order, one request per page, no retries, no sampling
  parameters. Same OCR input and model version produce the same report.
- **Fail fast / fail closed**: invalid OCR JSON, oversized pages, a missing API key, HTTP
  errors, malformed responses, unknown categories, or non-normalized probabilities abort
  the run; no value is coerced.
- **Traceable**: model version, prompt version, parameters, confidence, and probabilities
  are all recorded.
- Sequential requests with no retry: a `429`/`529` fails the run; re-run after a short wait
  or pin `--model jev-1.13.0` for stable comparison.
- Quality depends on OCR text quality and the category descriptions; low-confidence or
  `other` pages are flagged, not corrected.
- Only the six fixed categories are supported; changing the taxonomy means editing the
  constants and bumping `PROMPT_VERSION`.

## Verification

```bash
python -m pytest tests/ -q
```

The 34 tests in `tests/test_page_classifier.py` cover OCR JSON validation, `.env` and key
resolution, request payload and bearer header, response validation failures, empty-page
skipping, confidence boundaries, `other` flagging, the state size guard, HTTP/JSON error
paths, and end-to-end `main()` runs with a fake transport, so no API key is needed.
