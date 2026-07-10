.PHONY: install lint format typecheck test test-backend test-frontend dev-backend dev-frontend pre-commit clean eval eval-answers

VENV_PYTHON ?= .venv/Scripts/python.exe

# ---------- Setup ----------
install:
	pip install -r requirements.txt -r requirements-dev.txt
	npm install
	pre-commit install

# ---------- Code Quality ----------
lint:
	ruff check api/ core/ config.py main.py
	npx eslint src/

lint-fix:
	ruff check --fix api/ core/ config.py main.py
	npx eslint src/ --fix

format:
	ruff format api/ core/ config.py main.py
	npx prettier --write "src/**/*.{js,jsx,css}"

format-check:
	ruff format --check api/ core/ config.py main.py
	npx prettier --check "src/**/*.{js,jsx,css}"

typecheck:
	mypy api/ core/ config.py

pre-commit:
	pre-commit run --all-files

# ---------- Testing ----------
test: test-backend test-frontend

test-backend:
	pytest tests/ -v --cov=api --cov=core --cov-report=term-missing

test-frontend:
	npx react-scripts test --watchAll=false --coverage

# ---------- Evaluation ----------
# Deterministic retrieval metrics (hit-rate@k / MRR per toggle combo) - fast, no judge.
# Pass EVAL_USER=<uuid8> for a specific user's collection (auto-detected when only one exists).
EVAL_USER ?= default
eval:
	$(VENV_PYTHON) scripts/eval_retrieval.py --golden tests/eval/golden.jsonl --user $(EVAL_USER)

# DeepEval answer quality (Faithfulness + ContextualRelevancy) - slow, judge-based, comparative only
eval-answers:
	$(VENV_PYTHON) scripts/eval_answers.py --golden tests/eval/golden.jsonl --user $(EVAL_USER)

# Tier 3.1: measured embedder/reranker upgrade program (per-profile eval collections; results in docs/EVAL_RESULTS.md)
eval-program:
	$(VENV_PYTHON) scripts/eval_pipeline.py

# Tier 3.2: LongMemEval memory regression harness (downloads dataset on first run; MEMEVAL_N caps questions)
MEMEVAL_N ?= 20
memory-eval:
	$(VENV_PYTHON) tests/memory_eval.py --questions $(MEMEVAL_N)

# ---------- Development ----------
dev-backend:
	python -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

dev-frontend:
	npm start

# ---------- Docker ----------
docker-build:
	docker build -t rag-assistant:latest .

docker-up:
	docker compose up --build

# ---------- Cleanup ----------
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml
