# Grading-LLM

**TL;DR:** An end-to-end repository for an LLM based automated grading system (data → model → API → UI → monitoring). 

---

## Quick start (developer flow)

### 1. Create & activate a virtualenv
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install (dev)
pip install -r requirements-dev.txt
pip install -r requirements.txt

### 3. Run lint & tests
make lint
make test

### 4. Run CI locally
Use act or run the commands in .github/workflows/ci.yml

## Project structure (high level)
	data/ - datasets and derived artifacts.
	src/ - source code (preprocessing, baseline, api, store, etc).
	notebooks/ - reproducible notebooks for evaluation and analysis.
	ui/ — teacher review UI (Streamlit) and related assets.
	infra/ — Dockerfiles, docker-compose, monitoring infra.
	docs/ — rubric, labeling guide, model card, privacy notes.
	evaluation/ — evaluation artifacts, results tables.
	scripts/ — small utilities & developer scripts.
	tests/ — unit and integration tests.

