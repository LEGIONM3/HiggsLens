.PHONY: setup dev test lint download-data validate-data train-fast check

PYTHON := uv run --python 3.12 python
PYTEST := uv run --python 3.12 pytest
RUFF := uv run --python 3.12 ruff
MYPY := uv run --python 3.12 mypy

setup:
	uv venv --python 3.12 .venv
	uv pip install -e backend/
	cd frontend && npm install

dev:
	@echo "Starting FastAPI backend and Next.js frontend in development mode..."
	@bash scripts/start_dev.sh

test:
	$(PYTEST) -v backend/tests
	cd frontend && npm test

lint:
	$(RUFF) check backend/app backend/tests scripts
	$(MYPY) backend/app backend/tests scripts
	cd frontend && npm run typecheck

download-data:
	$(PYTHON) scripts/download_dataset.py

validate-data:
	$(PYTHON) scripts/validate_dataset.py

train-fast:
	$(PYTHON) scripts/train_models.py --mode fast

check: lint test
