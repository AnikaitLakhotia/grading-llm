.PHONY: help install-dev lint format test

help:
	@echo "Available targets:"
	@echo "  make install-dev  - Install dev requirements into current venv"
	@echo "  make lint         - Run formatting checks"
	@echo "  make format       - Autoformat with black/isort"
	@echo "  make test         - Run unit tests"

install-dev:
	python -m pip install --upgrade pip
	pip install -r requirements-dev.txt
	pip install -r requirements.txt

lint:
	black --check .
	isort --check-only .
	flake8 .

format:
	black .
	isort .

test:
	pytest -q