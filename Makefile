PYTHON ?= python

.PHONY: install prepare eda train evaluate ablation test lint format typecheck serve all docker-build

install:
	$(PYTHON) -m pip install -e ".[train,dev]"

prepare:
	$(PYTHON) -m house_price.cli prepare

eda:
	$(PYTHON) -m house_price.evaluation.eda

train:
	$(PYTHON) -m house_price.cli train

evaluate:
	$(PYTHON) -m house_price.cli evaluate

ablation:
	$(PYTHON) -m house_price.cli ablation

test:
	$(PYTHON) -m pytest --cov=house_price --cov-report=term-missing

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

format:
	$(PYTHON) -m ruff format src tests
	$(PYTHON) -m ruff check --fix src tests

typecheck:
	$(PYTHON) -m mypy src

serve:
	$(PYTHON) -m uvicorn house_price.serving.api:app --reload --port 8000

all:
	$(PYTHON) -m house_price.cli all

docker-build:
	docker build -t house-price-api:latest .
