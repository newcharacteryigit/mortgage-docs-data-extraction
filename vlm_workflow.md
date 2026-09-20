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
6. **Verify against OCR** — the VLM value is normalized (NFKD, casefold, dashes and
   punctuation to spaces) and searched in the normalized OCR page text. Exact containment
   scores `1.0`; otherwise the best similarity over token windows of size `n-1`, `n`, and
   `n+1` is used. A value at or above `--mismatch-threshold` is `ok`, below it is
   `mismatch`, and a `null` value is `missing`.
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
| Prompt version | `1.1` (recorded in the report) |

LM Studio ignores `chat_template_kwargs.enable_thinking=false` on the REST path, so
thinking is switched off with `reasoning_effort: "none"`. When `--thinking` is used and the
JSON still arrives in `reasoning_content`, it is parsed from there.

## Usage

```bash
python vlm_extract.py AREAL_LOAN.pdf
python vlm_extract.py AREAL_LOAN.pdf --ocr other.json --out report.json
python vlm_extract.py AREAL_LOAN.pdf --dpi 300 --mismatch-threshold 0.8
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
| `--mismatch-threshold` | Similarity below this value is a mismatch, default `0.75` |
| `--thinking` | Enable model thinking mode (off by default) |

## Outputs

### Terminal (stderr)

Progress and errors never pollute the JSON stream:

```text
[vlm] pdf=AREAL_LOAN.pdf ocr=AREAL_LOAN.json pages=12 dpi=200 model=qwen/qwen3.5-9b base_url=http://127.0.0.1:1234/v1 threshold=0.75 thinking=False
[vlm] page 1 done render=0.12s vlm=13.03s compare=0.001s mismatches=-
[vlm] page 2 done render=0.07s vlm=11.78s compare=0.000s mismatches=-
[vlm] total duration=158.16s pages=12 vlm=156.80s avg_per_page=13.07s
[vlm] mismatched fields=0 missing fields=17 pages_with_mismatch=[]
[vlm] JSON written to: AREAL_LOAN.vlm.json
```

On failure the last line is `ERROR: <ExceptionType>: <message>` and the process exits with
code `1`. Success exits with code `0`.

### JSON file

Schema version `1.0` (one page object per PDF page):

```json
{
  "schema_version": "1.0",
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
    "text_normalization": "nfkd_casefold_alnum"
  },
  "summary": {
    "page_count": 12,
    "field_status_counts": {
      "borrower_name": {"ok": 5, "mismatch": 0, "missing": 7},
      "property_address": {"ok": 5, "mismatch": 0, "missing": 7},
      "loan_number": {"ok": 10, "mismatch": 0, "missing": 2},
      "page_number": {"ok": 11, "mismatch": 0, "missing": 1}
    },
    "field_mismatch_count": 0,
    "field_missing_count": 17,
    "pages_with_mismatch": [],
    "needs_review_page_count": 0
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
      "verification": {
        "borrower_name": {"status": "ok", "found_in_ocr": true, "similarity": 1.0},
        "property_address": {"status": "ok", "found_in_ocr": true, "similarity": 1.0},
        "loan_number": {"status": "ok", "found_in_ocr": true, "similarity": 1.0},
        "page_number": {"status": "ok", "found_in_ocr": true, "similarity": 1.0}
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
| `parameters` | Render DPI, sampling settings, reasoning effort, mismatch threshold, normalization rule |
| `summary.field_status_counts` | `ok` / `mismatch` / `missing` counts per field across pages |
| `summary.field_mismatch_count` / `field_missing_count` | Totals over all fields and pages |
| `summary.pages_with_mismatch` | PDF page numbers with at least one mismatch |
| `summary.needs_review_page_count` | Number of pages with at least one mismatch |
| `pages[].page_number` | 1-based PDF page position, consistent with the OCR JSON |
| `pages[].page_ref` | `<source>#page-<n>` reference |
| `pages[].extracted` | VLM values, `null` when not found; `extracted.page_number` is the number printed on the page |
| `pages[].verification.<field>.status` | `ok` (found in OCR), `mismatch` (conflict with OCR), or `missing` (VLM returned `null`) |
| `pages[].verification.<field>.found_in_ocr` | Whether the value was located in the OCR text |
| `pages[].verification.<field>.similarity` | Best similarity score in `[0, 1]`, `null` for missing values |
| `pages[].mismatches` | Names of the fields with status `mismatch` |
| `pages[].ocr_reference` | OCR line count, mean score, and duration for the page |
| `pages[].timings_seconds` | Render, VLM, comparison, and total seconds for the page |
| `timings` | Document-level totals plus VLM request count and average VLM seconds per page |

## Behavior Guarantees

- **Deterministic**: fixed page order, one request per page, `temperature: 0`, fixed seed,
  no retries, and no sampling-dependent branching. The same PDF, OCR JSON, model, and
  threshold produce the same report.
- **Fail fast**: missing files, invalid OCR JSON, source/page-count mismatches, HTTP
  errors, malformed or schema-invalid model responses, and truncated responses raise a
  clear error; nothing is swallowed.
- **Traceable**: model id, prompt version, sampling parameters, normalization rule, and
  per-field verification decisions are recorded in the report.
- **Independent**: the module never imports `ocr_pdf` or `page_matcher`; the OCR JSON is
  treated as read-only reference text.
- **Isolated streams**: JSON goes to stdout only with `--stdout`; all logs go to stderr.

## Verification

```bash
python -m pytest tests/ -q
```

The 23 tests in `tests/test_vlm_extract.py` cover input and pairing validation, request
payload construction (image data URL, JSON schema, deterministic parameters, thinking
switch), response parsing (plain JSON, code fences, `reasoning_content` fallback, truncated
and malformed responses, wrong keys, empty strings), text normalization, fuzzy matching
boundaries, mismatch reporting, and an end-to-end run with a fake HTTP transport, so no LM
Studio instance is needed. An end-to-end run against LM Studio with `qwen/qwen3.5-9b`
processed `AREAL_LOAN.pdf` (12 pages) in 158.2 s with 0 mismatches and 17 missing fields.

## Current Limitations

- The OCR text is only a reference: when OCR itself misreads a value, a correct VLM value
  can still be flagged as `mismatch`. Raising `--mismatch-threshold` reduces this at the
  cost of missing real conflicts.
- Verification is string similarity, not semantics: reordered or reformatted values can be
  flagged, and very short values (for example the printed page number `"1"`) can match any
  occurrence of that character in the OCR text.
- One model call per page with no retries or caching; the 9B model took about 13 s per page
  on the sample document.
- Only the four listed fields are extracted; there are no word boxes, tables, or additional
  key-value pairs.
- Output quality depends on the model loaded in LM Studio; a different model id must be
  passed with `--model`.
