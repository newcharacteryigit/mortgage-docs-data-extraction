# Page Matching Workflow

`page_matcher.py` detects duplicate pages in the OCR JSON files produced by `ocr_pdf.py`.
It embeds each page with a local LM Studio embedding model and compares only consecutive
pages (`i` and `i+1`) of the same source document. Pages from different input files are
never compared.

The script is independent from the OCR code: it only consumes the OCR JSON output and
never opens the original PDF.

## Pipeline

1. **Validate inputs** — every input must exist, end with `.json`, be a JSON object with a
   non-empty `source`, and contain a non-empty `pages` list. Each page must have a positive
   integer `page_number` (strictly increasing) and a string `text`. Duplicate page
   references across inputs fail fast.
2. **Resolve the embedding model** — `--model` wins when given. Otherwise the script calls
   `GET /models` on LM Studio and accepts the exact match for
   `jina-embeddings-v5-text-small-text-matching` or a single model id containing it.
   Ambiguous or missing matches fail with the list of available model ids.
3. **Normalize page text** — OCR line breaks and repeated whitespace are collapsed into
   single spaces so layout noise does not affect the comparison.
4. **Embed pages in batches** — up to `--batch-size` pages per request are sent to
   `POST /v1/embeddings`. Every text is prefixed with `Document: ` because the
   text-matching adapter was trained with that prefix. Responses are validated strictly:
   count, index order, and numeric vectors must match the request exactly.
5. **Match adjacent pages** — only pairs with the same `source` and consecutive
   `page_number` values are compared. Page `i+1` joins the current group when the cosine
   similarity to page `i` is greater than or equal to `--threshold`; otherwise a new group
   starts. Groups are therefore contiguous runs of pages.
6. **Measure time** — embedding time, matching time, and total wall-clock time are recorded.
7. **Write JSON** — the result goes to a file (default) or to stdout, and progress/errors
   are printed to stderr.

## Model

`jinaai/jina-embeddings-v5-text-small-text-matching` served by LM Studio:

| Feature | Value |
| --- | --- |
| Parameters | 677M |
| Embedding dimension | 1024 |
| Max sequence length | 32768 tokens |
| Pooling | Last-token (server side) |
| Task | Symmetric text matching / duplicate detection |
| Input prefix | `Document: ` |
| Similarity | Cosine |

LM Studio must expose the model as an embedding model. If the GGUF is classified as an
LLM, the request fails and the error message explains how to fix it via
`Override Domain Type` → `Text Embedding` in LM Studio.

## Usage

```bash
python page_matcher.py document.json
python page_matcher.py document.json --threshold 0.85
python page_matcher.py a.json b.json --out matches.json
python page_matcher.py document.json --stdout
```

| Flag | Meaning |
| --- | --- |
| `ocr_json` | One or more OCR JSON files produced by `ocr_pdf.py` (required) |
| `--out` | JSON output path; default is the first input path with a `.matches.json` suffix |
| `--stdout` | Print JSON to stdout instead of writing a file (overrides `--out`) |
| `--base-url` | LM Studio OpenAI-compatible base URL, default `http://127.0.0.1:1234/v1` |
| `--model` | LM Studio model id; auto-resolved from `GET /models` when omitted |
| `--threshold` | Cosine similarity threshold, default `0.78` (inclusive) |
| `--batch-size` | Pages per embedding request, default `8` |
| `--timeout` | HTTP timeout in seconds, default `120.0` |
| `--prefix` | Prefix added to every page before embedding, default `Document: ` |

## Outputs

### Terminal (stderr)

Progress and errors never pollute the JSON stream:

```text
[match] files=1 pages=12 model=text-embedding-jina-embeddings-v5-text-small-text-matching base_url=http://127.0.0.1:1234/v1 threshold=0.8
[match] embedded 12/12 pages in 2 requests duration=1.33s skipped=0
[match] duplicate groups=4 duplicate pages=8 unique pages=4 skipped=0
[match] total duration=1.33s
[match] JSON written to: AREAL_LOAN.matches.json
```

On failure the last line is `ERROR: <ExceptionType>: <message>` and the process exits with
code `1`. Success exits with code `0`.

### JSON file

Schema version `1.1`:

```json
{
  "schema_version": "1.1",
  "input_files": ["AREAL_LOAN.json"],
  "model": {
    "id": "text-embedding-jina-embeddings-v5-text-small-text-matching",
    "base_url": "http://127.0.0.1:1234/v1",
    "document_prefix": "Document: ",
    "embedding_dimension": 1024
  },
  "parameters": {
    "similarity_threshold": 0.78,
    "batch_size": 8,
    "text_normalization": "whitespace_collapsed"
  },
  "summary": {
    "page_count": 12,
    "embedded_page_count": 11,
    "skipped_page_count": 1,
    "duplicate_group_count": 1,
    "duplicate_page_count": 2,
    "unique_page_count": 9
  },
  "groups": [
    {
      "group_id": 1,
      "size": 2,
      "page_refs": ["AREAL_LOAN.pdf#page-4", "AREAL_LOAN.pdf#page-5"],
      "representative": "AREAL_LOAN.pdf#page-4",
      "members": [
        {
          "page_ref": "AREAL_LOAN.pdf#page-4",
          "source": "AREAL_LOAN.pdf",
          "page_number": 4,
          "similarity_to_previous_page": null
        },
        {
          "page_ref": "AREAL_LOAN.pdf#page-5",
          "source": "AREAL_LOAN.pdf",
          "page_number": 5,
          "similarity_to_previous_page": 0.810434
        }
      ],
      "min_edge_similarity": 0.810434,
      "mean_edge_similarity": 0.810434
    }
  ],
  "unique_pages": [
    {"page_ref": "AREAL_LOAN.pdf#page-1", "source": "AREAL_LOAN.pdf", "page_number": 1}
  ],
  "skipped_pages": [
    {
      "page_ref": "AREAL_LOAN.pdf#page-7",
      "source": "AREAL_LOAN.pdf",
      "page_number": 7,
      "reason": "empty text"
    }
  ],
  "timings": {
    "total_seconds": 1.334,
    "embedding_seconds": 1.331,
    "matching_seconds": 0.001,
    "embedding_requests": 2
  }
}
```

| Field | Description |
| --- | --- |
| `schema_version` | Report schema version |
| `input_files` | OCR JSON paths as given on the command line |
| `model.id` / `model.base_url` | Resolved LM Studio model id and endpoint |
| `model.document_prefix` | Prefix applied to every page before embedding |
| `model.embedding_dimension` | Vector size reported by LM Studio, `null` when nothing was embedded |
| `parameters` | Threshold, batch size, and text normalization rule used |
| `summary.duplicate_group_count` | Number of groups with at least two pages |
| `summary.duplicate_page_count` | Total pages inside those groups |
| `groups[].group_id` | 1-based group index in document order |
| `groups[].page_refs` | Group members in page order, first page first |
| `groups[].representative` | First page of the group |
| `groups[].members[].similarity_to_previous_page` | Cosine score against the previous page, `null` for the first member |
| `groups[].min_edge_similarity` / `mean_edge_similarity` | Aggregates over the group's adjacent-pair scores |
| `unique_pages` | Pages without an adjacent match |
| `skipped_pages` | Empty pages that were not embedded, with the reason |
| `timings` | Total, embedding, and matching durations plus the number of embedding requests |

## Behavior Guarantees

- **Sequential and scoped**: only consecutive pages of one source document are compared.
  Non-adjacent duplicates (for example page 1 and page 3 with different page 2) are not
  grouped, and pages from different input files are never grouped.
- **Gaps break chains**: a skipped empty page interrupts the sequence because the
  `page_number + 1` condition no longer holds.
- **Deterministic**: fixed page order, fixed batching, no sampling, no unbounded retries.
  Same input, model, and threshold produce the same grouping.
- **Fail fast**: invalid OCR JSON, duplicate page references, ambiguous model ids, HTTP
  errors, and malformed embedding responses raise a clear error; nothing is swallowed.
- **Traceable**: model id, prefix, threshold, and normalization are recorded in the report.
- **Isolated streams**: JSON goes to stdout only with `--stdout`; all logs go to stderr.

## Verification

```bash
python -m pytest tests/ -q
```

The 45 tests cover input validation, model resolution, embedding response parsing,
adjacent grouping (threshold boundary, non-adjacent pages, source boundaries, skipped
page gaps), and two end-to-end runs with a fake HTTP transport, so no LM Studio instance
is needed. An end-to-end run against LM Studio with the jina v5 model matched
`AREAL_LOAN.json` (12 pages) in 2 requests in 1.33 s.

## Current Limitations

- Only adjacent pages are compared; there is no global all-pairs duplicate search and no
  cross-document matching yet.
- The threshold is corpus dependent: template-heavy documents (mortgage forms) can share
  high similarity without being duplicates, so tune `--threshold` on real samples.
- Matching uses page text only: no word boxes, tables, or key-value fields.
- Embedding quality and prefix behavior depend on the model loaded in LM Studio.
