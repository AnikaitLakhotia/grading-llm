# Grading-LLM

**TL;DR:** An end-to-end repository for an LLM-assisted grading system (data → model → API → UI → monitoring).  
Safe-by-default: this repo uses synthetic fixtures for tests and **does not** commit real student essays or PII.

---

## Quick start (developer flow)

> Assumes you are on macOS / Linux and have Python 3.9+ installed.

### 1. Create & activate a virtualenv
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
Runtime:
```bash
pip install -r requirements.txt
```
Developer (linters, tests, notebooks):
```bash
pip install -r requirements-dev.txt
# or (preferred) editable install with extras:
pip install -e .[dev]
```

### 3. Generate sample data (safe, deterministic)
```bash
python3 scripts/generate_synthetic_dataset.py --out data/sample_sanitized.csv --n 200 --seed 42
```

### 4. Run preprocessing (produces Parquet/JSONL artifacts)
```bash
bash scripts/run_preprocess.sh data/sample_sanitized.csv data/processed 50
# or
python3 -m src.preprocess.cli data/sample_sanitized.csv --out-prefix data/processed --sample 50
```

### 5. Run baseline evaluation (headless)
```bash
python3 scripts/run_baseline_eval.py
```
This prints Quadratic Weighted Kappa (QWK), ±1 accuracy, confusion matrices for TF-IDF baseline and a deterministic Mock LLM.

### 6. Run the API locally
```bash
# dev server
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
# then visit http://127.0.0.1:8000/docs for OpenAPI UI
```

### 7. Run lint & tests
```bash
# formatting & lint
black --check .
isort --check-only .

# unit tests
pytest -q
```

### 8. Run in Docker (local)
```bash
docker-compose -f infra/docker-compose.yml up --build -d
# smoke test
curl -sS -X POST http://localhost:8000/grade -H "Content-Type: application/json"   -d '{"essay_id":"e1","full_text":"This is a sample essay...", "assignment":"asm1", "prompt_name":"p1", "grade_level":10}'
```

---

## Useful scripts

- `scripts/generate_synthetic_dataset.py` - create a safe synthetic dataset (seeded).
- `scripts/run_preprocess.sh` / `src.preprocess.cli` - run the preprocessing pipeline (anonymize, normalize, feature extraction).
- `scripts/run_baseline_eval.py` - headless baseline evaluation (TF-IDF baseline + MockLLM).
- `scripts/run_prettify.sh` (optional) - run black/isort on the repo (create if desired).

---

## Project structure (high level)

```
grading-llm/
├─ README.md
├─ pyproject.toml
├─ requirements.txt
├─ requirements-dev.txt
├─ data/                 # sanitized samples & generated artifacts (ignored raw data)
├─ src/
│  ├─ data/              # schema + anonymizer
│  ├─ preprocess/        # pipeline + feature extractors
│  ├─ baseline/          # baselines + prompt runner
│  ├─ api/               # FastAPI app & schemas
│  └─ ...                # train, finetune, rag, etc (later commits)
├─ notebooks/            # evaluation notebooks (strip outputs before commit)
├─ infra/                # Dockerfile, docker-compose, monitoring infra
├─ docs/                 # rubric (kept out of tests), API docs, labeling guide
├─ scripts/              # helper scripts
└─ tests/                # unit & integration tests
```

---

## Security & privacy (read carefully)

- **Do not** commit raw student essays or PII. Use the anonymizer and synthetic fixtures for development and CI.
- Keep secrets out of the repo. Use `.env` for local development and CI secrets (see `.env.example`).
- MockLLMClient is used in CI to avoid external API calls and billing.

---

## Notes

- Key evaluation metrics: **Quadratic Weighted Kappa (QWK)** and **±1 accuracy** (appropriate for ordinal labels 1..6).
- The repo demonstrates:
  - Deterministic data tooling (schema, anonymizer, synthetic data)
  - Reproducible baselines (TF-IDF + logistic) and a prompt-runner for LLM workflows
  - A Dockerized FastAPI grading service with OpenAPI docs and structured logging
  - Human-in-the-loop components (teacher UI & label store planned in later commits)

---

## Contributing

- Follow Conventional Commits: `feat(...)`, `chore(...)`, `fix(...)`, `docs(...)`, `test(...)`.
- Run `make format` / `make lint` before committing.
- Add unit tests for new features; avoid committing sensitive data.

---

## License & Contact

This project is MIT licensed.

Contact: Anikait Lakhotia <anikhoti@uwaterloo.ca>
