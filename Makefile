.PHONY: help setup up down train stream consume ingest api dashboard test lint format clean

help: ## Show help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## First-time: install deps, download data, build Docker
	cp -n .env.example .env || true
	pip install -r requirements.txt
	python -c "from src.data.download import download_dataset; download_dataset()"
	docker compose build

up: ## Start all services
	docker compose up -d
	@echo "\n  MLflow:    http://localhost:5000"
	@echo "  Airflow:   http://localhost:8080  (admin/admin)"
	@echo "  API:       http://localhost:8001/docs"
	@echo "  Dashboard: http://localhost:8501\n"

down: ## Stop all services
	docker compose down

logs: ## Tail logs
	docker compose logs -f --tail=50

train: ## Train models + log to MLflow
	python -m src.ml.train

stream: ## Start Kafka producer
	python -m src.streaming.producer

consume: ## Start Kafka consumer
	python -m src.streaming.consumer

ingest: ## Ingest SOPs into ChromaDB
	python -c "from src.rag.engine import RAGEngine; e = RAGEngine(); e.ingest()"

api: ## Start FastAPI locally
	uvicorn src.api.main:app --host 0.0.0.0 --port 8001 --reload

dashboard: ## Start Streamlit locally
	streamlit run src/dashboard/app.py --server.port 8501

test: ## Run tests
	python -m pytest tests/ -v

lint: ## Lint with ruff
	ruff check src/ tests/

format: ## Format with ruff
	ruff format src/ tests/

clean: ## Remove generated files
	rm -rf data/raw/*.csv data/vectordb/* data/models/* __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
