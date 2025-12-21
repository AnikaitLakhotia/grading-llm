# Essay Dataset Schema (canonical)

This document specifies the canonical schema for the essay dataset used in the grading-llm project.

## Columns

- `essay_id` (string, required)
  - Unique identifier for each essay. Deterministic IDs are preferred (e.g., UUID or seeded integer tokens).

- `score` (integer, required)
  - Holistic score on a 1..6 scale (inclusive), aligned with the Holistic Rating Form in `docs/rubric.md`.

- `full_text` (string, required)
  - The sanitized essay body. Text must be plain UTF-8; no HTML. PII should be removed or replaced.

- `assignment` (string, optional)
  - Assignment short name / code.

- `prompt_name` (string, optional)
  - Short identifier of prompt used.

- `grade_level` (string or int, optional)
  - Student grade level (e.g., `10`, `11`, or `K-12`). Optional, but encouraged.

- `essay_word_count` (int, required)
  - Word count of `full_text`. Must approximate `len(full_text.split())` (±2 words tolerance allowed).

## Notes & constraints

- `score` must be integer values 1, 2, 3, 4, 5, or 6.
- `essay_id` must be unique across the dataset.
- `full_text` should not contain raw email addresses, phone numbers, or SSNs. If such values exist, the anonymizer will replace them with `[EMAIL]`, `[PHONE]`, `[SSN]`.
- If additional personal identifiers are encountered (e.g., `First Last`) the anonymizer attempts to replace them with deterministic hashed tokens like `[PERSON_ab12cd34]`. This heuristic is intentionally conservative: inspect automated replacements before trusting for real data.
- Any extra columns are allowed but must be documented elsewhere. The validation script enforces the columns above exist.

## Example row

| essay_id | score | full_text          | assignment | prompt_name | grade_level | essay_word_count |
|----------|-------|--------------------|------------|-------------|-------------|------------------|
| e00001   | 4     | "The author argues..." | asm-01    | prompt-a    | 10          | 243              |