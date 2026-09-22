# Category Verification Workflow

`category_verifier.py` reads the `.categories.json` report produced by `page_classifier.py`
and the OCR JSON produced by `ocr_pdf.py`, and asks a local LM Studio model for a second
opinion on every page whose classifier confidence or top-two probability margin is below a
configured threshold. The page image is never rendered: the verifier sends the same OCR page
text that the TypeSafe Jev classifier received. The verifier writes a decision report in the
shape of the Categories JSON, enriched with a per-page `verification` block, so downstream
consumers (`page_matcher.py`, `pipeline.py`) can use it without changes.

The script never opens the PDF and never imports `ocr_pdf`, `page_matcher`, or
`vlm_extract`. It reuses the OCR loader together with the category labels, criteria, and
question instructions from `page_classifier.py`, so the taxonomy and the OCR interpretation
cannot drift between the classifier and the verifier.

## What it does

1. Validates the Categories JSON (object with `input_files.ocr_json`, a `categories` mapping
   that matches the six category keys, and positive, strictly increasing `page_number`
   entries with `page_ref`, `category`, `confidence`, `probabilities`, `needs_review`, and
   `review_reason`; probabilities must have exactly the six category keys, lie in `[0, 1]`,
   sum to 1, and peak on the recorded category).
2. Loads the OCR JSON: from `--ocr` when given, otherwise from the recorded
   `input_files.ocr_json` (resolved next to the Categories JSON when the stored path is not
   reachable). OCR page numbers and `page_ref` values must match the Categories JSON exactly.
3. Computes the trigger per answered page: `margin = p(top1) − p(top2)` over the recorded
   probabilities. A page is verified when `confidence < --confidence-threshold` or
   `margin < --margin-threshold`. Pages without a category (empty text) are never verified.
4. Fails fast when a verified page text exceeds `--max-state-chars`, before any request.
5. Sends one `POST /v1/chat/completions` request per verified page to LM Studio with the OCR
   page text as the user message, the versioned system prompt, and a strict JSON schema
   answer (`category` enum over the six keys).
6. Strictly validates the answer: `choices[0].message` content (or `reasoning_content` as a
   fallback), stripped code fences, exactly one `category` key, a known category, and a
   `usage` object with integer token counts. Invalid answers mark the page, they do not
   abort the run.
7. Decides per verified page and rewrites `needs_review` / `review_reason`:
   - same category and not `other` → `accepted`, review flag cleared;
   - both sides chose `other` → `mutual_other`, review kept with `other category`;
   - different categories → `disagreement`, review with `verifier disagreement`;
   - request or answer failure → `verifier failed`, review with `verifier failed`.
   Pages that were not verified keep their original review state (`not_triggered`).
8. Records the trigger reasons, margin, verifier category, resolution, error, per-page
   seconds, and token usage, plus document totals.
9. Writes JSON to a file or stdout; all progress and errors go to stderr.

## Triggers and Decisions

| Trigger | Rule |
| --- | --- |
| `low confidence` | `confidence < --confidence-threshold` (default `0.5`) |
| `low margin` | `p(top1) − p(top2) < --margin-threshold` (default `0.5`) |

Threshold comparisons are strict, so a value exactly at the threshold does not trigger.
When both rules fire, both reasons are recorded and the page is verified once.

| Resolution | Condition | Effect |
| --- | --- | --- |
| `not_triggered` | No threshold breached | Original `needs_review` / `review_reason` preserved |
| `accepted` | Verifier category equals the classifier category and is not `other` | Review flag cleared |
| `mutual_other` | Both sides chose `other` | Review kept, reason `other category` |
| `disagreement` | Categories differ | Review set, reason `verifier disagreement` |
| `verifier_failed` | HTTP error, malformed or invalid answer | Review set, reason `verifier failed`; run continues |

## Model and API

| Item | Value |
| --- | --- |
| Endpoint | `POST http://127.0.0.1:1234/v1/chat/completions` (LM Studio) |
| Input | Text only: the OCR page text, no image or PDF rendering |
| Model | `--model`, default `google/gemma-4-e2b`; pin the model for reproducible runs |
| Structured output | `response_format: json_schema` (strict, single `category` enum field) |
| Thinking | Disabled with `reasoning_effort: "none"` and `chat_template_kwargs.enable_thinking: false` |
| Sampling | `temperature: 0.0`, `top_p: 1.0`, `seed: 0`, `max_tokens: 128`, no streaming |
| Taxonomy | Category labels, criteria, and instructions imported from `page_classifier.py` |
| Prompt version | `1.0` (recorded in the report) |

## Usage

```bash
python category_verifier.py AREAL_LOAN.categories.json
python category_verifier.py AREAL_LOAN.categories.json --ocr AREAL_LOAN.json
python category_verifier.py AREAL_LOAN.categories.json --confidence-threshold 0.9
python category_verifier.py AREAL_LOAN.categories.json --margin-threshold 0.7 --stdout
python category_verifier.py AREAL_LOAN.categories.json --model qwen/qwen3.5-9b
```

| Flag | Meaning |
| --- | --- |
| `categories_json` | Categories JSON produced by `page_classifier.py` (required) |
| `--ocr` | OCR JSON from `ocr_pdf.py`; default is the OCR path recorded in the Categories JSON |
| `--out` | JSON output path; default is the categories path with a `.category_review.json` suffix |
| `--stdout` | Print JSON to stdout instead of writing a file (overrides `--out`) |
| `--base-url` | LM Studio OpenAI-compatible base URL, default `http://127.0.0.1:1234/v1` |
| `--model` | LM Studio model id, default `google/gemma-4-e2b` |
| `--timeout` | HTTP timeout in seconds, default `120.0` |
| `--max-tokens` | Maximum response tokens per page, default `128` |
| `--seed` | Sampling seed, default `0` |
| `--confidence-threshold` | Pages below this confidence are verified, default `0.5` |
| `--margin-threshold` | Pages below this top-two probability margin are verified, default `0.5` |
| `--max-state-chars` | Fail fast when a verified page text exceeds this size, default `30000` |

## Outputs

### Terminal (stderr)

```text
[verify] categories=AREAL_LOAN.categories.json ocr=AREAL_LOAN.json pages=12 model=google/gemma-4-e2b base_url=http://127.0.0.1:1234/v1 confidence_threshold=0.5 margin_threshold=0.5 input=ocr_text
[verify] page 12 triggered reasons=low confidence margin=0.7000
[verify] page 12 done verifier_category=mortgage_closing_disclosure_seller resolution=accepted
[verify] total duration=1.31s pages=12 triggered=1 accepted=1 mutual_other=0 disagreement=0 failed=0
[verify] needs_review before=0 after=0
[verify] JSON written to: AREAL_LOAN.category_review.json
```

On failure the last line is `ERROR: <ExceptionType>: <message>` and the process exits with
code `1`. Success exits with code `0`.

### JSON file

Schema version `1.0`, prompt version `1.0`:

```json
{
  "schema_version": "1.0",
  "input_files": {
    "categories_json": "AREAL_LOAN.categories.json",
    "ocr_json": "AREAL_LOAN.json"
  },
  "model": {
    "id": "google/gemma-4-e2b",
    "base_url": "http://127.0.0.1:1234/v1",
    "prompt_version": "1.0",
    "response_model": "google/gemma-4-e2b"
  },
  "categories": {"lender_rate_note": "Lender - Rate Note", "...": "..."},
  "parameters": {
    "input_mode": "ocr_text",
    "confidence_threshold": 0.5,
    "margin_threshold": 0.5,
    "max_state_chars": 30000,
    "timeout_seconds": 120.0,
    "temperature": 0.0,
    "top_p": 1.0,
    "seed": 0,
    "max_tokens": 128,
    "reasoning_effort": "none"
  },
  "summary": {
    "page_count": 12,
    "triggered_page_count": 1,
    "accepted_page_count": 1,
    "mutual_other_page_count": 0,
    "disagreement_page_count": 0,
    "verifier_failed_page_count": 0,
    "needs_review_before": 0,
    "needs_review_after": 0,
    "triggered_pages": [12]
  },
  "pages": [
    {
      "page_number": 12,
      "page_ref": "AREAL_LOAN.pdf#page-12",
      "category": "mortgage_closing_disclosure_seller",
      "category_label": "Mortgage - Closing Disclosure - Seller",
      "confidence": 0.82,
      "probabilities": {"mortgage_closing_disclosure_seller": 0.85, "other": 0.15},
      "needs_review": false,
      "review_reason": null,
      "verification": {
        "triggered": true,
        "trigger_reasons": ["low confidence"],
        "margin": 0.7,
        "verifier_category": "mortgage_closing_disclosure_seller",
        "resolution": "accepted",
        "error": null,
        "verifier_seconds": 1.31,
        "usage": {"prompt_tokens": 1420, "completion_tokens": 12}
      }
    }
  ],
  "timings": {
    "total_seconds": 1.312,
    "verify_seconds": 1.31,
    "requests": 1,
    "avg_verify_seconds_per_request": 1.31
  },
  "usage": {"prompt_tokens": 1420, "completion_tokens": 12, "requests": 1}
}
```

| Field | Description |
| --- | --- |
| `input_files` | Categories JSON and resolved OCR JSON paths |
| `model.id` / `model.base_url` | Requested LM Studio model and endpoint |
| `model.prompt_version` | Verifier instructions and schema version |
| `model.response_model` | Model id reported by LM Studio, `null` when no request succeeded |
| `categories` | Category key to label mapping used in the run |
| `parameters.input_mode` | Always `ocr_text` (no page images are rendered) |
| `summary.triggered_pages` | PDF page numbers sent to the verifier |
| `summary.needs_review_before` / `needs_review_after` | Input and output review-page counts |
| `pages[]` | The original Categories JSON page fields plus `verification` |
| `pages[].verification.trigger_reasons` | `low confidence` and/or `low margin` |
| `pages[].verification.margin` | `p(top1) − p(top2)`, `null` for unverified pages |
| `pages[].verification.resolution` | `not_triggered`, `accepted`, `mutual_other`, `disagreement`, or `verifier_failed` |
| `pages[].verification.error` | Exception summary for a failed verification, otherwise `null` |
| `timings` / `usage` | Verifier duration, request count, and token totals |

## Pipeline Integration

`pipeline.py` runs five stages: OCR, classification, verification, VLM extraction, and
matching. The verification stage consumes the `.categories.json` report and writes
`<stem>.category_review.json` next to it. The matching stage receives the review report as
its `--categories-json` input, so a page marked `needs_review` by the verifier makes the
category signal unavailable and can change grouping. The final result report (schema `2.1`)
builds page labels from the review report and records the verification duration in
`timings.stages_seconds.verification`; per-page classification seconds still come from the
original Categories JSON.

## Behavior Guarantees

- **Deterministic**: fixed page order, one text request per verified page, `temperature: 0`,
  fixed seed, no retries, no sampling-dependent branching. Same inputs, model, and
  thresholds produce the same report.
- **Fail fast**: invalid Categories JSON, missing or mismatched OCR JSON, oversized verified
  pages, and invalid thresholds raise a clear error before or during the run.
- **Per-page resilience**: HTTP errors, malformed answers, and truncated responses mark only
  that page as `verifier failed` and the remaining pages are still verified; nothing is
  swallowed, the error text is recorded.
- **Traceable**: model id, response model, prompt version, trigger reasons, margin,
  resolution, and token usage are recorded per page.
- **Isolated streams**: JSON goes to stdout only with `--stdout`; all logs go to stderr.
- **Schema-compatible output**: the report keeps every Categories JSON field, so
  `page_matcher.py` and `pipeline.py` accept it unchanged; the extra `verification` block is
  ignored by both.

## Verification

```bash
python -m pytest tests/ -q
```

The 37 tests in `tests/test_category_verifier.py` cover Categories JSON validation, trigger
selection and boundaries, text-only request payload construction (image-free, deterministic
parameters, enum schema), response parsing (plain JSON, code fences, `reasoning_content`
fallback, wrong fields, unknown category, truncation, missing usage), every resolution
(`not_triggered`, `accepted`, `mutual_other`, `disagreement`, `verifier_failed`), per-page
failure recovery, OCR coverage and `page_ref` checks, the state size guard, matcher
compatibility of the output, and the CLI paths, all with a fake HTTP transport, so no LM
Studio instance is needed. With the default thresholds the sample document triggers no
requests (lowest confidence 0.82, lowest margin 0.70); `--confidence-threshold 0.9` verifies
page 12, whose runner-up mass is `other` (0.15), and a stopped LM Studio degrades that page
to `verifier failed` without failing the run.

## Current Limitations

- The verifier sees the same OCR text as the classifier, so OCR errors remain correlated;
  the second opinion only adds model-family and prompt diversity, not an independent view.
- Agreement is not proof of correctness: two models can share a blind spot, and the
  `accepted` decision is only as good as the model. A labeled golden set is needed to
  measure the agreement-error rate before trusting it blindly.
- Default thresholds are uncalibrated placeholders; on the sample document they trigger
  nothing, and a correctly classified low-confidence page (12) is only verified when the
  confidence threshold is raised above 0.82.
- Only confidence and margin trigger verification; a page whose runner-up mass is `other`
  but whose margin stays above the threshold is not re-checked.
- One request per verified page with no retries or caching; a failing LM Studio turns every
  triggered page into `verifier failed`.
- The verifier does not change the classifier category; a `disagreement` keeps the Jev label
  and only adds a review flag for human triage.
