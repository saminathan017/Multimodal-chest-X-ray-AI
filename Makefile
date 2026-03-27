# ─────────────────────────────────────────────────────────────────
# ClinicalAI — Makefile
# Usage: make <target>
# ─────────────────────────────────────────────────────────────────

.PHONY: install run test lint format clean docker-build docker-run help

PYTHON      := python3
PIP         := $(PYTHON) -m pip
APP         := app/streamlit_app.py
PORT        := 8501
IMAGE_NAME  := clinical-ai-demo

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev: ## Install dev dependencies
	$(PIP) install black ruff pytest pytest-asyncio

run: ## Run the Streamlit demo app
	streamlit run $(APP) --server.port $(PORT)

run-demo: ## Run in demo mode (no model download needed)
	DEMO_MODE=true streamlit run $(APP) --server.port $(PORT)

test: ## Run the test suite
	$(PYTHON) -m pytest tests/ -v --tb=short

lint: ## Lint with ruff
	$(PYTHON) -m ruff check src/ app/ tests/

format: ## Format with black
	$(PYTHON) -m black src/ app/ tests/

clean: ## Remove __pycache__, .pyc, build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".DS_Store" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache dist build *.egg-info

docker-build: ## Build Docker image
	docker build -t $(IMAGE_NAME) -f deploy/docker/Dockerfile .

docker-run: ## Run Docker container
	docker run -p $(PORT):$(PORT) --env-file .env $(IMAGE_NAME)

deploy-sagemaker: ## Deploy to AWS SageMaker (requires AWS credentials)
	$(PYTHON) deploy/sagemaker/deploy_endpoint.py --action deploy

teardown-sagemaker: ## Delete SageMaker endpoint (stop billing)
	$(PYTHON) deploy/sagemaker/deploy_endpoint.py --action delete
