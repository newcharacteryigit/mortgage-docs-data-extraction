# AGENTS.md

This file guides AI agents working in this repo. Follow it for every change.

## Scope

- You are a coding agent. Keep changes small, focused, and reversible.
- Do not document the product system here. This file is only operating instructions.

## How to Work

1. Read relevant code and configs before editing. Do not assume.
2. Plan the smallest step that solves the task. If ambiguous, ask.
3. Implement, then verify with the repo's configured tests, lint, and type checks.
4. Summarize what changed, how it was verified, and what is left uncertain.

## Clean Code

- Follow KISS and YAGNI. Prefer explicit, readable code over clever code.
- Small pure functions, explicit inputs/outputs, no hidden side effects.
- Full type annotations. No `Any` or untyped boundaries without justification.
- No magic values: name constants, make thresholds and timeouts configurable.
- Fail fast with clear errors. Never swallow exceptions silently.
- Write comments and error messages in English.
- Match existing style and structure. Do not introduce new patterns unilaterally.

## Determinism

- Same input must produce same output. Set temperatures to 0 and fix seeds where applicable.
- Pin behavior: no unbounded retries, no non-deterministic ordering, no unpinned dependencies affecting logic.
- Use structured outputs with strict schema validation. Reject invalid outputs, do not coerce silently.
- Version prompts, schemas, and rules alongside code. Log model, prompt version, and decisions for traceability.

## Extraction Quality

- Schema first: define expected fields and types before implementing extraction.
- Validate everything: required fields, formats, ranges. Fail closed on invalid data.
- Never hallucinate. If a value is missing or uncertain, leave it empty and flag `needs_review`.
- Prefer rules and exact matching over model inference where precision matters.
- Keep classification and extraction logic separate and independently testable.

## Testing and Verification

- Add or update tests for every logic change. Cover happy path, edge cases, and invalid inputs.
- Prefer golden-file tests for classification and extraction behavior.
- All configured checks must pass before finishing. Do not skip verification.

## Don'ts

- No over-engineering: no new frameworks, services, queues, or abstractions without explicit request.
- No prompt, schema, or taxonomy changes bundled with unrelated refactors.
- No broad refactors or renames. Touch only what the task requires.
- No secrets in code, logs, or tests. Use placeholders and env vars.
