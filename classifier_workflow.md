# Page Classification Workflow

`page_classifier.py` assigns every page of an OCR JSON file produced by `ocr_pdf.py` to one
of six fixed mortgage document categories with the TypeSafe Jev model and the Choice
primitive. One JSON report holds the category, confidence, probability distribution,
per-page timings, and token usage for each page.

The script is independent from the OCR, matching, and extraction code: it only consumes the
OCR JSON output and never opens the original PDF or imports `ocr_pdf`, `page_matcher`, or
`vlm_extract`.

## Pipeline

1. **Validate inputs** — the OCR JSON must exist, end with `.json`, be a JSON object with a
   non-empty `source`, and contain a non-empty `pages` list. Each page needs a positive,
   strictly increasing integer `page_number` and a string `text`. Optional `line_count`,
   `mean_score`, and `duration_seconds` must be numeric when present. Page texts are checked
   against `--max-state-chars` before any request is sent, so an oversized page fails fast
   without spending API calls.
2. **Resolve the API key** — `.env` is parsed as simple `KEY=VALUE` lines (blank lines,
   `#` comments, `export`, and surrounding quotes are handled). An existing
   `TYPESAFE_API_KEY` environment variable wins over the file. A missing key fails with a
   clear error.
3. **Classify each page** — one `POST /v1/systemone` request per non-empty page. The state
   is the raw OCR text; the question is a single Choice with six options (five categories
   plus `other`). Requests are sequential and there are no retries.
4. **Validate the answer strictly** — the response must contain a `choice` answer named
   `page_category` whose option is one of the six categories, a numeric `confidence` within
   `[0, 1]`, and a `probabilities` map with exactly the six category keys, each value within
   `[0, 1]`, summing to 1, and peaked on the chosen option. The response `model` (for
   example `jev-1.13.0`) and token usage are captured. Any violation fails closed.
5. **Flag pages for review** — an empty page is skipped without a request, a confidence
   below `--confidence-threshold` is `low confidence`, and the `other` category is
   `other category`. Flagged pages get `needs_review: true` with the reason.
6. **Measure time** — per page: classification seconds and total seconds. Document level:
   total and classification sums, request count, and average classification seconds per
   page plus token totals.
7. **Write JSON** — the result goes to a file (default) or to stdout, and progress/errors
   are printed to stderr.

## Categories

| Category key | Label |
| --- | --- |
| `mortgage_closing_disclosure_seller` | Mortgage - Closing Disclosure - Seller |
| `lender_rate_note` | Lender - Rate Note |
| `title_rider` | Title - Rider |
| `property_tax_record_information_sheet` | Property - Tax Record Information Sheet |
| `title_signature_name_affidavit_ack` | Title - Signature / Name Affidavit (Ack) |
| `other` | Other / Unclassified |

The Choice instructions and criteria are part of the request and are versioned with the
code (`PROMPT_VERSION`). A continuation page without its own title is classified as the
document type it continues.

## Model

`jev-latest` (currently `jev-1.13.0`) served by TypeSafe:

| Feature | Value |
| --- | --- |
| Endpoint | `POST https://api.typesafe.ai/v1/systemone` |
| Authentication | `Authorization: Bearer $TYPESAFE_API_KEY` |
| Input | Text only; the OCR page text is sent as `state` |
| Question | One `choice` question `page_category` with six criteria |
| Context | 64k tokens per request, 32k for `state` plus the longest question; a page of this corpus is about 1k tokens |
| Sampling | No sampling parameters; answers are deterministic for a fixed state and model version |

The API key is read from `TYPESAFE_API_KEY` in the environment or from `.env` (see
`.env.example`). The alias `--model jev-latest` can move between releases, so pin
`--model jev-1.13.0` when comparing runs; the report records the versioned model id that
answered each request.

## Usage

```bash
python page_classifier.py AREAL_LOAN.json
python page_classifier.py AREAL_LOAN.json --out result.json
python page_classifier.py AREAL_LOAN.json --confidence-threshold 0.6
python page_classifier.py AREAL_LOAN.json --model jev-1.13.0 --stdout
```

| Flag | Meaning |
| --- | --- |
| `ocr_json` | OCR JSON produced by `ocr_pdf.py` (required) |
| `--out` | JSON output path; default is the OCR path with a `.categories.json` suffix |
| `--stdout` | Print JSON to stdout instead of writing a file (overrides `--out`) |
| `--base-url` | TypeSafe API base URL, default `https://api.typesafe.ai/v1` |
| `--model` | TypeSafe model id or alias, default `jev-latest` |
| `--timeout` | HTTP timeout in seconds, default `60.0` |
| `--confidence-threshold` | Confidence below this value is flagged as `needs_review`, default `0.5` |
| `--max-state-chars` | Fail fast when a page text exceeds this length, default `30000` |
| `--env-file` | File providing `TYPESAFE_API_KEY`, default `.env` |

## Outputs

### Terminal (stderr)

Progress and errors never pollute the JSON stream:

```text
[classify] ocr=AREAL_LOAN.json pages=12 model=jev-latest base_url=https://api.typesafe.ai/v1 threshold=0.5 max_state_chars=30000
[classify] page 1 done category=property_tax_record_information_sheet confidence=0.9812 duration=1.42s review=-
[classify] page 2 done category=property_tax_record_information_sheet confidence=0.9431 duration=1.18s review=-
[classify] total duration=14.20s pages=12 requests=12 avg_per_page=1.18s
[classify] needs_review pages=[] categories={...}
[classify] JSON written to: AREAL_LOAN.categories.json
```

On failure the last line is `ERROR: <ExceptionType>: <message>` and the process exits with
code `1`. Success exits with code `0`.

### JSON file

Schema version `1.0`:

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
  "parameters": {
    "confidence_threshold": 0.5,
    "max_state_chars": 30000,
    "timeout_seconds": 60.0
  },
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
  "timings": {
    "total_seconds": 14.2,
    "classify_seconds": 14.15,
    "requests": 12,
    "avg_classify_seconds_per_page": 1.179
  },
  "usage": {"input_tokens": 17040, "output_tokens": 456, "requests": 12}
}
```

| Field | Description |
| --- | --- |
| `schema_version` | Report schema version |
| `input_files.ocr_json` | OCR JSON path as given on the command line |
| `model.id` / `model.base_url` | Requested model alias and endpoint |
| `model.prompt_version` | Instructions and criteria version used, for traceability |
| `model.response_model` | Versioned model id that answered, `null` when nothing was classified |
| `categories` | Category key to label mapping used in the run |
| `parameters` | Confidence threshold, state size limit, and timeout used |
| `summary.category_counts` | Pages per category, including categories with zero pages |
| `summary.needs_review_pages` | Page numbers flagged for review |
| `pages[].page_ref` | `<source>#page-<n>` reference |
| `pages[].category` / `category_label` | Chosen category key and label, `null` for skipped pages |
| `pages[].confidence` | Choice confidence in `[0, 1]`, `null` for skipped pages |
| `pages[].probabilities` | Distribution across all six categories, `null` for skipped pages |
| `pages[].needs_review` / `review_reason` | Review flag and reason (`empty text`, `low confidence`, `other category`) |
| `pages[].ocr_reference` | OCR line count, mean score, and duration for the page |
| `pages[].timings_seconds` | Classification and total seconds for the page |
| `pages[].usage` | Input and output tokens for the page, `null` for skipped pages |
| `skipped_pages` | Empty pages that were not sent to the API |
| `timings` | Total, classification, request count, and average classification seconds |
| `usage` | Token totals over all classified pages |

## Behavior Guarantees

- **Independent**: the module never imports `ocr_pdf`, `page_matcher`, or `vlm_extract`; the
  OCR JSON is treated as read-only input.
- **Deterministic**: fixed page order, one request per page, full category list in every
  request, no retries, no sampling-dependent branching. TypeSafe answers depend only on the
  state and model version, and both are recorded in the report.
- **Fail fast**: missing or invalid OCR JSON, page texts over `--max-state-chars`, a missing
  API key, HTTP errors, malformed or schema-invalid answers, unknown categories,
  non-normalized probabilities, and an empty response model raise a clear error; nothing is
  swallowed and no request is sent for invalid input.
- **Fail closed**: the category is never coerced. Any answer that does not match the
  category set exactly is rejected.
- **Traceable**: requested model, versioned response model, prompt version, parameters, and
  per-page confidence and probabilities are recorded.
- **Isolated streams**: JSON goes to stdout only with `--stdout`; all logs go to stderr.

## Verification

```bash
python -m pytest tests/ -q
```

The 34 tests in `tests/test_page_classifier.py` cover OCR JSON validation, `.env` parsing
and key resolution, request payload and bearer header construction, response validation
(unknown category, missing answer, wrong keys, unnormalized probabilities, out-of-range
confidence, choice without the highest probability, bad usage), empty-page skipping,
confidence threshold boundaries, `other` flagging, the state size guard, HTTP and JSON error
paths, and end-to-end `main()` runs with a fake transport, so no TypeSafe API key is needed.

For a live end-to-end run, copy `.env.example` to `.env`, put a real key in it, and run:

```bash
python page_classifier.py AREAL_LOAN.json
```

## Current Limitations

- One request per page, sequential, with no retries. A `429` or `529` response fails the
  run; re-run after a short wait or pin a specific model version.
- `--max-state-chars` is a character guard, not a tokenizer; it is set far below the 32k
  token state budget so typical pages never come close.
- The confidence threshold is corpus and risk dependent; it only flags pages, it never
  changes the chosen category.
- Classification quality depends on OCR text quality and on the category descriptions in
  `CATEGORY_CRITERIA`; pages whose OCR title is wrong can be misclassified with high
  confidence, so the probability distribution should be inspected for critical decisions.
- Only the six fixed categories are supported; a different taxonomy requires editing the
  constants and bumping `PROMPT_VERSION`.
