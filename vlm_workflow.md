# VLM Extraction Workflow

`vlm_extract.py` extracts `borrower_name`, `property_address`, `loan_number`, and
`page_number` from every page of a mortgage PDF with a local vision-language model served
by LM Studio (`qwen/qwen3.5-9b`). Each page is rendered to an image and sent with a strict
JSON schema; the returned values are verified against the page text of the OCR JSON
produced by `ocr_pdf.py`. One JSON report holds every page with its extracted fields,
verification status, and per-page timings.

The script is independent from the OCR and matching code: it only consumes the OCR JSON
output and re-opens the original PDF itself.

## Pipeline

1. **Validate inputs** — the PDF must exist, end with `.pdf`, open with PyMuPDF, and contain
   at least one page. The OCR JSON must exist, end with `.json`, be an object with a
   non-empty `source` and a non-empty `pages` list; every page needs a positive, strictly
   increasing `page_number` and a string `text`. Optional `line_count`, `mean_score`, and
   `duration_seconds` must be numeric when present. Any violation fails fast.
2. **Check pairing** — the OCR `source` file name must equal the PDF file name, the PDF
   page count must equal the OCR page count, and OCR page numbers must be contiguous from 1.
3. **Render the page** — PyMuPDF rasterizes the page at `--dpi` (default `200`) into PNG
   bytes, which are base64-encoded into a `data:image/png;base64,...` URL.
4. **Extract with the VLM** — one `POST /v1/chat/completions` request per page. The system
   prompt lists the four fields and the rules (copy exactly as printed, bare printed page
   number, `null` when absent, never guess, JSON only). The response is constrained by
   `response_format: json_schema` (`strict`, nullable strings, `additionalProperties:
   false`).
5. **Parse and validate strictly** — the JSON object is read from `message.content` or,
   when content is empty, from `message.reasoning_content`; code fences and surrounding
   text are stripped. The payload must contain exactly the four keys with string or `null`
   values; empty strings, wrong keys, truncated responses (`finish_reason: length`), and
   malformed JSON are rejected.
6. **Verify against OCR** — the VLM value is canonicalized per field through the shared
   `field_normalizer.py` module: addresses get annotation and USPS designator
   normalization, person names are token-sorted and dash-insensitive, and loan numbers drop
   formatting so `20-414-784` equals `20414784`. Exact containment in the normalized page
   text scores `1.0` (`match_mode: exact`); otherwise the best similarity over token windows
   of size `n-1`, `n`, and `n+1` is used (`match_mode: fuzzy`). The effective threshold is
   the higher of `--mismatch-threshold` and the pinned per-field minimum
   (`property_address` 0.90, `borrower_name` 0.85, `loan_number` and `page_number` 1.0). A
   value at or above it is `ok`, below it is `mismatch`, and a `null` value is `missing`.
   Fuzzy `ok` values closer than `--review-margin` (default `0.10`) to the threshold are
   flagged with `needs_review: true` instead of being silently accepted.
7. **Measure time** — per page: render, VLM request, comparison, and total duration.
   Document level: total, render, VLM, and comparison sums, request count, and average VLM
   time per page.
8. **Write JSON** — the result goes to a file (default) or to stdout, and progress/errors
   are printed to stderr.

## Model

`qwen/qwen3.5-9b` served by LM Studio:

| Feature | Value |
| --- | --- |
| Endpoint | `POST /v1/chat/completions` |
| Image input | PNG data URL, rendered at `200` DPI |
| Structured output | `response_format: json_schema` (strict, four nullable string fields) |
| Thinking | Disabled by default via `reasoning_effort: "none"`; `--thinking` re-enables it |
| Sampling | `temperature: 0.0`, `top_p: 1.0`, `seed: 0`, `max_tokens: 512`, no streaming |
| Canonicalization | `field_normalizer.py` version `1.0`, recorded in the report |
| Prompt version | `1.1` (recorded in the report) |

LM Studio ignores `chat_template_kwargs.enable_thinking=false` on the REST path, so
thinking is switched off with `reasoning_effort: "none"`. When `--thinking` is used and the
JSON still arrives in `reasoning_content`, it is parsed from there.

## Usage

```bash
python vlm_extract.py AREAL_LOAN.pdf
python vlm_extract.py AREAL_LOAN.pdf --ocr other.json --out report.json
python vlm_extract.py AREAL_LOAN.pdf --dpi 300 --mismatch-threshold 0.8
python vlm_extract.py AREAL_LOAN.pdf --review-margin 0.05
python vlm_extract.py AREAL_LOAN.pdf --stdout
```

| Flag | Meaning |
| --- | --- |
| `pdf` | PDF file to process (required) |
| `--ocr` | OCR JSON from `ocr_pdf.py`; default is the PDF path with a `.json` suffix |
| `--out` | JSON output path; default is the PDF path with a `.vlm.json` suffix |
| `--stdout` | Print JSON to stdout instead of writing a file (overrides `--out`) |
| `--base-url` | LM Studio OpenAI-compatible base URL, default `http://127.0.0.1:1234/v1` |
| `--model` | LM Studio model id, default `qwen/qwen3.5-9b` |
| `--dpi` | Page render resolution, default `200` |
| `--timeout` | HTTP timeout in seconds, default `120.0` |
| `--max-tokens` | Maximum response tokens per page, default `512` |
| `--seed` | Sampling seed, default `0` |
| `--mismatch-threshold` | Global similarity floor; per-field minimums can be higher, default `0.75` |
| `--review-margin` | Fuzzy `ok` matches within this margin above the effective threshold are flagged for review, default `0.10` |
| `--thinking` | Enable model thinking mode (off by default) |

## Outputs

### Terminal (stderr)

Progress and errors never pollute the JSON stream:

```text
[vlm] pdf=AREAL_LOAN.pdf ocr=AREAL_LOAN.json pages=12 dpi=200 model=qwen/qwen3.5-9b base_url=http://127.0.0.1:1234/v1 threshold=0.75 thinking=False
[vlm] page 1 done render=0.12s vlm=13.03s compare=0.001s mismatches=- review=-
[vlm] page 2 done render=0.07s vlm=11.78s compare=0.000s mismatches=- review=-
[vlm] total duration=158.16s pages=12 vlm=156.80s avg_per_page=13.07s
[vlm] mismatched fields=1 missing fields=14 review flags=0 pages_with_mismatch=[11] pages_with_review_flag=[]
[vlm] JSON written to: AREAL_LOAN.vlm.json
```

On failure the last line is `ERROR: <ExceptionType>: <message>` and the process exits with
code `1`. Success exits with code `0`.

### JSON file

Schema version `1.1` (one page object per PDF page):

```json
{
  "schema_version": "1.1",
  "input_files": {"pdf": "AREAL_LOAN.pdf", "ocr_json": "AREAL_LOAN.json"},
  "model": {
    "id": "qwen/qwen3.5-9b",
    "base_url": "http://127.0.0.1:1234/v1",
    "thinking": false,
    "prompt_version": "1.1"
  },
  "parameters": {
    "dpi": 200,
    "temperature": 0.0,
    "top_p": 1.0,
    "seed": 0,
    "max_tokens": 512,
    "reasoning_effort": "none",
    "mismatch_similarity_threshold": 0.75,
    "review_margin": 0.1,
    "field_similarity_thresholds": {
      "borrower_name": 0.85,
      "property_address": 0.9,
      "loan_number": 1.0,
      "page_number": 1.0
    },
    "text_normalization": "nfkd_casefold_alnum",
    "canonicalization_version": "1.0"
  },
  "summary": {
    "page_count": 12,
    "field_status_counts": {
      "borrower_name": {"ok": 7, "mismatch": 0, "missing": 5},
      "property_address": {"ok": 4, "mismatch": 1, "missing": 7},
      "loan_number": {"ok": 10, "mismatch": 0, "missing": 2},
      "page_number": {"ok": 12, "mismatch": 0, "missing": 0}
    },
    "field_mismatch_count": 1,
    "field_missing_count": 14,
    "review_flag_count": 0,
    "pages_with_mismatch": [11],
    "pages_with_review_flag": [],
    "needs_review_page_count": 1
  },
  "pages": [
    {
      "page_number": 1,
      "page_ref": "AREAL_LOAN.pdf#page-1",
      "extracted": {
        "borrower_name": "Alya Renard—Van Mercer",
        "property_address": "604N C restview Hill Dr, Unit 1144",
        "loan_number": "20414784",
        "page_number": "1"
      },
      "normalized": {
        "borrower_name": "alya mercer renard van",
        "property_address": "604n c restview hill dr unit 1144",
        "loan_number": "20414784",
        "page_number": "1"
      },
      "verification": {
        "borrower_name": {"status": "ok", "found_in_ocr": true, "similarity": 1.0, "match_mode": "exact", "needs_review": false},
        "property_address": {"status": "ok", "found_in_ocr": true, "similarity": 1.0, "match_mode": "exact", "needs_review": false},
        "loan_number": {"status": "ok", "found_in_ocr": true, "similarity": 1.0, "match_mode": "exact", "needs_review": false},
        "page_number": {"status": "ok", "found_in_ocr": true, "similarity": 1.0, "match_mode": "exact", "needs_review": false}
      },
      "mismatches": [],
      "ocr_reference": {"line_count": 82, "mean_score": 0.9819, "duration_seconds": 6.096},
      "timings_seconds": {
        "render_seconds": 0.116,
        "vlm_seconds": 13.03,
        "compare_seconds": 0.001,
        "total_seconds": 13.146
      }
    }
  ],
  "timings": {
    "total_seconds": 158.155,
    "render_seconds": 1.248,
    "vlm_seconds": 156.804,
    "compare_seconds": 0.098,
    "vlm_requests": 12,
    "avg_vlm_seconds_per_page": 13.067
  }
}
```

| Field | Description |
| --- | --- |
| `schema_version` | Report schema version |
| `input_files` | PDF and OCR JSON paths as given on the command line |
| `model.id` / `model.base_url` | LM Studio model id and endpoint used |
| `model.thinking` | Whether thinking mode was enabled |
| `model.prompt_version` | Prompt and schema version used, for traceability |
| `parameters` | Render DPI, sampling settings, reasoning effort, global threshold, review margin, per-field thresholds, normalization rule, canonicalization version |
| `summary.field_status_counts` | `ok` / `mismatch` / `missing` counts per field across pages |
| `summary.field_mismatch_count` / `field_missing_count` | Totals over all fields and pages |
| `summary.review_flag_count` | Number of `ok` fields flagged `needs_review` |
| `summary.pages_with_mismatch` | PDF page numbers with at least one mismatch |
| `summary.pages_with_review_flag` | PDF page numbers with at least one review flag but no mismatch |
| `summary.needs_review_page_count` | Number of pages with a mismatch or a review flag |
| `pages[].page_number` | 1-based PDF page position, consistent with the OCR JSON |
| `pages[].page_ref` | `<source>#page-<n>` reference |
| `pages[].extracted` | VLM values, `null` when not found; `extracted.page_number` is the number printed on the page |
| `pages[].normalized` | Canonical form per field from `field_normalizer.py` (address designators, token-sorted names, formatting-free loan numbers) |
| `pages[].verification.<field>.status` | `ok` (found in OCR), `mismatch` (conflict with OCR), or `missing` (VLM returned `null`) |
| `pages[].verification.<field>.found_in_ocr` | Whether the value was located in the OCR text |
| `pages[].verification.<field>.similarity` | Best similarity score in `[0, 1]`, `null` for missing values |
| `pages[].verification.<field>.match_mode` | `exact` when the canonical value is contained in the normalized OCR text, `fuzzy` when window similarity was used, `null` for missing values |
| `pages[].verification.<field>.needs_review` | `true` for fuzzy `ok` values within the review margin of the effective threshold |
| `pages[].mismatches` | Names of the fields with status `mismatch` |
| `pages[].ocr_reference` | OCR line count, mean score, and duration for the page |
| `pages[].timings_seconds` | Render, VLM, comparison, and total seconds for the page |
| `timings` | Document-level totals plus VLM request count and average VLM seconds per page |

## Loan-Level Aggregation

`pipeline.py` consumes the `.vlm.json` report and reduces per-page values to one
loan-level value per field (`loan_level_fields` in the result JSON). Aggregation is
deterministic and uses the same `field_normalizer.py` module:

1. Only values with verification status `ok` are considered.
2. Values are clustered per field with canonical keys; address variants that differ only
   in punctuation, designators, bracket annotations, or word splitting (`Crest View`
   versus `Crestview`) fall into the same cluster.
3. The winning cluster has the most pages, then the highest mean OCR similarity, then the
   earliest page.
4. The reported `value` is the surface form with the most support inside the cluster, then
   the fewest non-ASCII characters, then no annotation artifacts, then the earliest page.
5. `variants` records every observed surface form with its pages, and `needs_review` is
   `true` when the winning cluster covers fewer than two pages, the runner-up is within one
   page, or any contributing page carried a VLM `needs_review` flag.

Result schema `2.0` adds top-level `schema_version` and `canonicalization_version`; each
loan field carries `value`, `canonical_value`, `source_pages`, `variants`, and
`needs_review`. Schema `2.1` additionally records the `verification` stage in
`timings.stages_seconds` and takes page labels from the `category_verifier.py` report.
For `AREAL_LOAN.pdf` the four property-address spellings cluster together,
the clean `604 N Crestview Hill Dr Unit 1144, Las Vegas, NV 89139` is reported, and the
single-page borrower mailing address on page 11 is outvoted.

## Behavior Guarantees

- **Deterministic**: fixed page order, one request per page, `temperature: 0`, fixed seed,
  no retries, and no sampling-dependent branching. The same PDF, OCR JSON, model, and
  threshold produce the same report.
- **Fail fast**: missing files, invalid OCR JSON, source/page-count mismatches, HTTP
  errors, malformed or schema-invalid model responses, and truncated responses raise a
  clear error; nothing is swallowed.
- **Traceable**: model id, prompt version, sampling parameters, normalization rule,
  canonicalization version, per-field thresholds, and per-field verification decisions are
  recorded in the report.
- **Independent**: the module never imports `ocr_pdf` or `page_matcher`; the OCR JSON is
  treated as read-only reference text.
- **Isolated streams**: JSON goes to stdout only with `--stdout`; all logs go to stderr.

## Verification

```bash
python -m pytest tests/ -q
```

The 30 tests in `tests/test_vlm_extract.py` cover input and pairing validation, request
payload construction (image data URL, JSON schema, deterministic parameters, thinking
switch), response parsing (plain JSON, code fences, `reasoning_content` fallback, truncated
and malformed responses, wrong keys, empty strings), text normalization, field-specific
thresholds, review flags, fuzzy matching boundaries, mismatch reporting, and an end-to-end
run with a fake HTTP transport, so no LM Studio instance is needed. The 16 tests in
`tests/test_field_normalizer.py` and the aggregation tests in `tests/test_pipeline.py`
cover canonicalization, clustering, representative selection, and review-flag propagation.
An end-to-end run against LM Studio with `qwen/qwen3.5-9b` processed `AREAL_LOAN.pdf`
(12 pages) in 158.2 s under the previous schema; with the field-specific thresholds the
page-11 borrower mailing address (`similarity 0.7674`, previously `ok`) is now reported as
a `property_address` mismatch, and the loan-level value is the clean property address.

## Current Limitations

- The OCR text is only a reference: when OCR itself misreads a value, a correct VLM value
  can still be flagged as `mismatch`. Raising `--mismatch-threshold` reduces this at the
  cost of missing real conflicts.
- Verification is string similarity, not semantics: a wrong value that literally appears
  on the page (for example a borrower mailing address) can still be `ok` on that page.
  Loan-level aggregation outvotes it across pages, but a document where the wrong value
  dominates remains a risk.
- Very short values (for example the printed page number `"1"`) can match any occurrence
  of that character in the OCR text.
- One model call per page with no retries or caching; the 9B model took about 13 s per page
  on the sample document.
- Only the four listed fields are extracted; there are no word boxes, tables, or additional
  key-value pairs.
- Output quality depends on the model loaded in LM Studio; a different model id must be
  passed with `--model`.
