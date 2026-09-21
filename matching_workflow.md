# Page Matching Workflow

`page_matcher.py` groups consecutive pages of OCR JSON documents produced by `ocr_pdf.py`.
It embeds each page with a local LM Studio embedding model and compares only adjacent pages
(`i` and `i+1`) of the same source document. The embedding cosine can be combined with two
optional auxiliary signals: page categories from `page_classifier.py` and the printed page
number extracted by `vlm_extract.py`. Pages from different input files are never compared.

The script is independent from the OCR, classifier, and VLM code: it only consumes their
JSON outputs and never opens the original PDF.

## Pipeline

1. **Validate inputs** — every OCR input must exist, end with `.json`, be a JSON object with
   a non-empty `source`, and contain a non-empty `pages` list. Each page must have a
   positive integer `page_number` (strictly increasing) and a string `text`. Duplicate page
   references across inputs fail fast.
2. **Load auxiliary signals** — categories JSON and VLM JSON are taken from `--categories-json`
   and `--vlm-json`, or discovered next to each OCR input as `<stem>.categories.json` and
   `<stem>.vlm.json` when the flags are omitted. Every non-empty OCR page must have an entry;
   unknown page references or missing pages fail fast. A category of `other`, a page flagged
   `needs_review`, or a missing VLM printed page number simply makes that signal unavailable
   for the affected edges (no penalty).
3. **Resolve the embedding model** — `--model` wins when given. Otherwise the script calls
   `GET /models` on LM Studio and accepts the exact match for
   `jina-embeddings-v5-text-small-text-matching` or a single model id containing it.
   Ambiguous or missing matches fail with the list of available model ids.
4. **Normalize page text** — OCR line breaks and repeated whitespace are collapsed into
   single spaces so layout noise does not affect the comparison.
5. **Embed pages in batches** — up to `--batch-size` pages per request are sent to
   `POST /v1/embeddings`. Every text is prefixed with `Document: ` because the
   text-matching adapter was trained with that prefix. Responses are validated strictly:
   count, index order, and numeric vectors must match the request exactly.
6. **Score adjacent pairs** — only pairs with the same `source` and consecutive
   `page_number` values are scored. The weighted score is compared with `--threshold`;
   page `i+1` joins the current group when the combined score passes, otherwise a new group
   starts. Groups are therefore contiguous runs of pages.
7. **Measure time** — embedding time, matching time, and total wall-clock time are recorded.
8. **Write JSON** — the result goes to a file (default) or to stdout, and progress/errors
   are printed to stderr.

## Scoring

For an adjacent pair `(i, i+1)` the combined score is the weighted average of the available
signals, renormalized by the available weight:

```
combined = Σ(w_k · s_k) / Σ(w_k)     for every signal k that is available for this pair
```

| Signal | Weight | Rule |
| --- | --- | --- |
| Embedding | `--weight-embedding`, default `0.60` | Cosine similarity of the two page embeddings, clamped to `[0, 1]` |
| Category | `--weight-category`, default `0.25` | Same category → `min(confidence_i, confidence_{i+1})`; different known categories → `0.0`; `other`, `needs_review`, or missing → unavailable |
| Page number | `--weight-page-number`, default `0.15` | Printed page numbers consecutive (`n` → `n+1`) → `1.0`; equal → `0.5`; other or missing → unavailable |

Two hard rules are applied before grouping:

- **Embedding gate** — an edge whose cosine is below `--min-embedding-similarity`
  (default `0.60`) can never match, regardless of category or page number bonuses.
- **Threshold** — the renormalized combined score must be greater than or equal to
  `--threshold` (default `0.78`).

Worked example from the sample document: edge `7 → 8` has embedding `0.827355`, category
`0.99`, and consecutive printed page numbers (`1 → 2`), giving
`0.6·0.827355 + 0.25·0.99 + 0.15·1.0 = 0.893913`. Edge `3 → 4` has embedding `0.810434`
but different categories and no printed page number on page 3, giving
`(0.6·0.810434 + 0.25·0.0) / 0.85 = 0.572071`, so it stays below the threshold.

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
python page_matcher.py document.json --categories-json cats.json --vlm-json vlm.json
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
| `--categories-json` | Categories JSON from `page_classifier.py`; default is the OCR path with `.categories.json` when present |
| `--vlm-json` | VLM JSON from `vlm_extract.py`; default is the OCR path with `.vlm.json` when present |
| `--weight-embedding` | Embedding weight, default `0.60` |
| `--weight-category` | Category weight, default `0.25` |
| `--weight-page-number` | Printed page number weight, default `0.15` |
| `--min-embedding-similarity` | Hard cosine gate, default `0.60` |
| `--threshold` | Combined score threshold, default `0.78` (inclusive) |
| `--batch-size` | Pages per embedding request, default `8` |
| `--timeout` | HTTP timeout in seconds, default `120.0` |
| `--prefix` | Prefix added to every page before embedding, default `Document: ` |

Explicit `--categories-json` and `--vlm-json` paths can only be used with a single OCR
input; with multiple inputs the sibling files are auto-discovered per input.

## Outputs

### Terminal (stderr)

Progress and errors never pollute the JSON stream:

```text
[match] files=1 pages=12 model=text-embedding-jina-embeddings-v5-text-small-text-matching base_url=http://127.0.0.1:1234/v1 threshold=0.78 weights=0.6/0.25/0.15 min_embedding=0.6 categories=AREAL_LOAN.categories.json vlm=AREAL_LOAN.vlm.json
[match] embedded 12/12 pages in 2 requests duration=1.38s skipped=0
[match] duplicate groups=4 duplicate pages=11 unique pages=1 skipped=0
[match] total duration=1.38s
[match] JSON written to: AREAL_LOAN.matches.json
```

On failure the last line is `ERROR: <ExceptionType>: <message>` and the process exits with
code `1`. Success exits with code `0`.

### JSON file

Schema version `1.2`, scoring version `1.0`:

```json
{
  "schema_version": "1.2",
  "scoring_version": "1.0",
  "input_files": ["AREAL_LOAN.json"],
  "auxiliary": {
    "categories_json": "AREAL_LOAN.categories.json",
    "vlm_json": "AREAL_LOAN.vlm.json"
  },
  "model": {
    "id": "text-embedding-jina-embeddings-v5-text-small-text-matching",
    "base_url": "http://127.0.0.1:1234/v1",
    "document_prefix": "Document: ",
    "embedding_dimension": 1024
  },
  "parameters": {
    "similarity_threshold": 0.78,
    "batch_size": 8,
    "text_normalization": "whitespace_collapsed",
    "min_embedding_similarity": 0.6,
    "weights": {"embedding": 0.6, "category": 0.25, "page_number": 0.15}
  },
  "summary": {
    "page_count": 12,
    "embedded_page_count": 12,
    "skipped_page_count": 0,
    "duplicate_group_count": 4,
    "duplicate_page_count": 11,
    "unique_page_count": 1
  },
  "groups": [
    {
      "group_id": 3,
      "size": 4,
      "page_refs": [
        "AREAL_LOAN.pdf#page-7",
        "AREAL_LOAN.pdf#page-8",
        "AREAL_LOAN.pdf#page-9",
        "AREAL_LOAN.pdf#page-10"
      ],
      "representative": "AREAL_LOAN.pdf#page-7",
      "members": [
        {
          "page_ref": "AREAL_LOAN.pdf#page-7",
          "source": "AREAL_LOAN.pdf",
          "page_number": 7,
          "edge": null
        },
        {
          "page_ref": "AREAL_LOAN.pdf#page-8",
          "source": "AREAL_LOAN.pdf",
          "page_number": 8,
          "edge": {
            "combined_similarity": 0.893913,
            "signals": {
              "embedding_similarity": 0.827355,
              "category_match": 0.99,
              "page_number_sequence": 1.0
            },
            "available_weight": 1.0
          }
        },
        {
          "page_ref": "AREAL_LOAN.pdf#page-9",
          "source": "AREAL_LOAN.pdf",
          "page_number": 9,
          "edge": {
            "combined_similarity": 0.835407,
            "signals": {
              "embedding_similarity": 0.775159,
              "category_match": 0.98,
              "page_number_sequence": null
            },
            "available_weight": 0.85
          }
        }
      ],
      "min_combined_similarity": 0.831813,
      "mean_combined_similarity": 0.853711
    }
  ],
  "unique_pages": [
    {"page_ref": "AREAL_LOAN.pdf#page-3", "source": "AREAL_LOAN.pdf", "page_number": 3}
  ],
  "skipped_pages": [],
  "timings": {
    "total_seconds": 1.38,
    "embedding_seconds": 1.378,
    "matching_seconds": 0.0,
    "embedding_requests": 2
  }
}
```

| Field | Description |
| --- | --- |
| `schema_version` / `scoring_version` | Output shape and scoring rule versions |
| `input_files` | OCR JSON paths as given on the command line |
| `auxiliary` | Auxiliary files used, `null` when a signal was not available |
| `model.id` / `model.base_url` | Resolved LM Studio model id and endpoint |
| `model.document_prefix` | Prefix applied to every page before embedding |
| `model.embedding_dimension` | Vector size reported by LM Studio, `null` when nothing was embedded |
| `parameters` | Threshold, batch size, normalization, embedding gate, and signal weights |
| `summary.duplicate_group_count` | Number of groups with at least two pages |
| `summary.duplicate_page_count` | Total pages inside those groups |
| `groups[].group_id` | 1-based group index in document order |
| `groups[].page_refs` | Group members in page order, first page first |
| `groups[].representative` | First page of the group |
| `groups[].members[].edge` | `null` for the first member, otherwise the scored edge from the previous page |
| `groups[].members[].edge.combined_similarity` | Renormalized weighted score compared with the threshold |
| `groups[].members[].edge.signals` | Raw signal values; `null` when a signal was unavailable for that edge |
| `groups[].members[].edge.available_weight` | Sum of weights that contributed to this edge |
| `groups[].min_combined_similarity` / `mean_combined_similarity` | Aggregates over the group's edges |
| `unique_pages` | Pages without an adjacent match |
| `skipped_pages` | Empty pages that were not embedded, with the reason |
| `timings` | Total, embedding, and matching durations plus the number of embedding requests |

## Behavior Guarantees

- **Sequential and scoped**: only consecutive pages of one source document are compared.
  Non-adjacent duplicates (for example page 1 and page 3 with different page 2) are not
  grouped, and pages from different input files are never grouped.
- **Gaps break chains**: a skipped empty page interrupts the sequence because the
  `page_number + 1` condition no longer holds.
- **Auxiliary signals are optional**: without the categories or VLM file the matching runs
  on embeddings alone (the combined score equals the cosine), using the same CLI.
- **Deterministic**: fixed page order, fixed batching, no sampling, no unbounded retries.
  Same input, files, model, weights, and threshold produce the same grouping.
- **Fail fast**: invalid OCR or auxiliary JSON, duplicate or mismatched page references,
  ambiguous model ids, HTTP errors, and malformed embedding responses raise a clear error;
  nothing is swallowed.
- **Traceable**: model id, prefix, weights, threshold, gate, and auxiliary files are
  recorded in the report.
- **Isolated streams**: JSON goes to stdout only with `--stdout`; all logs go to stderr.

## Verification

```bash
python -m pytest tests/ -q
```

The 66 tests in `tests/test_page_matcher.py` cover input validation, auxiliary loading and
page reference matching, model resolution, embedding response parsing, category and page
number rules, weighted combination with renormalization, the embedding gate, adjacent
grouping, and end-to-end runs with a fake HTTP transport, so no LM Studio instance is
needed. On the sample document with the jina v5 model and default settings, all 12 pages
were embedded in 2 requests in 1.38 s and grouped as
`[1, 2] [4, 5, 6] [7, 8, 9, 10] [11, 12]` with page 3 unique.

| Edge | Embedding | Category | Page number | Combined |
| --- | --- | --- | --- | --- |
| 1 → 2 | 0.881327 | 1.00 | 1.0 | 0.928796 |
| 2 → 3 | 0.763439 | 0.00 | unavailable | 0.538898 |
| 3 → 4 | 0.810434 | 0.00 | unavailable | 0.572071 |
| 4 → 5 | 0.785084 | 1.00 | 1.0 | 0.871050 |
| 5 → 6 | 0.886860 | 1.00 | 1.0 | 0.932116 |
| 6 → 7 | 0.777498 | 0.00 | unavailable | 0.548822 |
| 7 → 8 | 0.827355 | 0.99 | 1.0 | 0.893913 |
| 8 → 9 | 0.775159 | 0.98 | unavailable | 0.835407 |
| 9 → 10 | 0.770068 | 0.98 | unavailable | 0.831813 |
| 10 → 11 | 0.657748 | 0.00 | unavailable | 0.464293 |
| 11 → 12 | 0.761873 | 0.84 | 1.0 | 0.817124 |

## Current Limitations

- Weights and threshold are corpus dependent; recalibrate them on real documents. In the
  sample the matched edges score at least `0.817` and rejected edges at most `0.572`, so
  the default `0.78` threshold sits in a wide gap.
- Only adjacent pages are compared; there is no global all-pairs duplicate search and no
  cross-document matching yet.
- Printed page number anomalies are left uncorrected: the Note pages are numbered
  `1, 2, 4, 3` yet stay in one group because the category signal carries them.
- Empty pages are skipped and reported; they do not contribute signals and they break the
  adjacency chain.
- OCR `mean_score` reliability and `needs_review` flags are not part of the scoring yet;
  low-quality OCR pages are treated like any other page.
- Embedding quality and prefix behavior depend on the model loaded in LM Studio.
