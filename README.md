# ShopFloor AI

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-15%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-7%20services-%230db7ed.svg?logo=docker&logoColor=white)](docker-compose.yml)

An end-to-end manufacturing intelligence platform that predicts machine failures from sensor data, streams real-time production KPIs through Apache Kafka, and enables natural-language queries over maintenance documentation using Retrieval-Augmented Generation (RAG).

Built on the [AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020) (10,000 records, 14 features, 5 failure modes from a CNC milling process).

<!-- Screenshots: replace these with your actual screenshots -->
<!-- ![Dashboard](docs/dashboard.png) -->
<!-- ![Predictor](docs/predictor.png) -->
<!-- ![MLflow](docs/mlflow.png) -->

---

## Overview

ShopFloor AI addresses a core challenge in manufacturing: predicting equipment failures before they occur and giving operators immediate access to procedural knowledge. The platform combines classical ML for failure classification, real-time streaming for KPI monitoring, and LLM-powered document retrieval for on-demand Q&A.

**Key results:**
- Gradient Boosting classifier achieved **0.85 F1 score** and **0.96 AUC-ROC** across 5 failure modes (tool wear, heat dissipation, power, overstrain, random)
- 16 engineered features derived from raw sensor readings including power output, thermal differential, and overstrain product
- 3 model architectures (Logistic Regression, Random Forest, Gradient Boosting) trained, evaluated, and tracked in MLflow
- 6 manufacturing SOPs indexed and retrievable via natural-language queries

## Architecture

```
                          ┌─────────────────────────────────┐
                          │       Streamlit Dashboard       │
                          │ KPIs | Predictor | Chat | Drift │
                          └──────────────┬──────────────────┘
                                         │ HTTP
                          ┌──────────────▼──────────────────┐
                          │         FastAPI Backend         │
                          │   /predict | /ask | /health     │
                          └───┬──────────┬────────────┬─────┘
                              │          │            │
                    ┌─────────▼──┐ ┌─────▼──────┐ ┌──▼──────────┐
                    │   MLflow   │ │  ChromaDB  │ │    Kafka    │
                    │    Model   │ │  Vector DB │ │   Broker    │
                    │  Registry  │ │  (RAG)     │ │             │
                    └─────┬──────┘ └─────┬──────┘ └──┬───────┬──┘
                          │              │           │       │
                          │         ┌────▼─────┐ ┌──▼──┐ ┌──▼────────┐
                          │         │ LangChain│ │Prod-│ │ Consumer  │
                          │         │ + OpenAI │ │ucer │ │ (rolling  │
                          │         │ API      │ │     │ │  KPIs +   │
                          │         └──────────┘ └─────┘ │  alerts)  │
                          │                              └───────────┘
                    ┌─────▼──────────────────────────────────────────┐
                    │           Apache Airflow Orchestration         │
                    │ Data Quality > Train > Evaluate > Promote > RAG│
                    │                  (Daily @ 2 AM)                │
                    └────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Streaming | Apache Kafka | Real-time sensor data ingestion and anomaly alerting |
| ML Tracking | MLflow | Experiment logging, model comparison, and registry |
| Orchestration | Apache Airflow | Scheduled retraining with data quality gates |
| RAG | LangChain + ChromaDB + OpenAI API | Natural-language Q&A over manufacturing SOPs |
| API | FastAPI | REST endpoints for predictions and document queries |
| Dashboard | Streamlit + Plotly | Interactive KPI monitoring, failure predictor, chat |
| Infrastructure | Docker Compose | 7 containerized microservices |
| ML | scikit-learn | Failure classification (Gradient Boosting, Random Forest) |

## Dataset

**Source:** [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) (UCI ML Repository, CC BY 4.0)

| Feature | Description |
|---------|-------------|
| `Air temperature [K]` | Ambient temperature, mean ~300K |
| `Process temperature [K]` | Cutting zone temperature, mean ~310K |
| `Rotational speed [rpm]` | Spindle rotation speed |
| `Torque [Nm]` | Applied cutting force, mean ~40 Nm |
| `Tool wear [min]` | Cumulative tool usage in minutes |
| `Type` | Product quality variant: L (50%), M (30%), H (20%) |
| `Machine failure` | Binary target (3.39% positive rate) |
| `TWF, HDF, PWF, OSF, RNF` | Five independent failure mode indicators |

## Model Performance

| Model | F1 Score | AUC-ROC |
|-------|----------|---------|
| **Gradient Boosting** | **0.8548** | **0.9591** |
| Random Forest | 0.8387 | 0.9823 |
| Logistic Regression | 0.3054 | 0.9455 |

Gradient Boosting was selected and registered in MLflow as the production model. All experiments are fully reproducible via `make train`.

## Quick Start

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- OpenAI API key ([platform.openai.com](https://platform.openai.com))

### 1. Clone and configure

```bash
git clone https://github.com/TanayPratapSingh/shopfloor-ai.git
cd shopfloor-ai

cp .env.example .env
# Open .env and add your OPENAI_API_KEY
```

### 2. Install dependencies and download data

```bash
make setup
```

This installs Python packages, downloads the AI4I dataset from UCI, and builds Docker images.

### 3. Run tests

```bash
make test
```

Runs 15 unit tests covering data loading, feature engineering, streaming logic, and SOP document structure.

### 4. Train models

```bash
make train
```

Trains three classifiers, logs all parameters, metrics, and artifacts to MLflow, and registers the best model.

### 5. Start all services

```bash
make up
```

| Service | URL | Credentials |
|---------|-----|-------------|
| Dashboard | http://localhost:8501 | |
| API Docs | http://localhost:8001/docs | |
| MLflow | http://localhost:5001 | |
| Airflow | http://localhost:8080 | admin / admin |

### 6. Ingest documents for RAG

```bash
make ingest
```

Loads 6 manufacturing SOPs into ChromaDB. The "Ask ShopFloor AI" chat page is now functional.

### 7. Run the streaming demo

```bash
# Terminal 1: Start Kafka producer (replays sensor data)
make stream

# Terminal 2: Start Kafka consumer (rolling KPIs + alerts)
make consume
```

## Project Structure

```
shopfloor-ai/
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI
├── configs/
│   └── config.yaml                 # Kafka, MLflow, ChromaDB, LLM settings
├── data/
│   ├── raw/                        # AI4I dataset (auto-downloaded, gitignored)
│   ├── processed/                  # Feature-engineered outputs
│   ├── models/                     # MLflow artifacts
│   └── vectordb/                   # ChromaDB persistence
├── src/
│   ├── data/
│   │   ├── download.py             # Dataset retrieval from UCI repository
│   │   └── sop_documents.py        # 6 manufacturing SOPs for RAG ingestion
│   ├── streaming/
│   │   ├── producer.py             # Kafka producer (sensor data stream)
│   │   └── consumer.py             # Kafka consumer (KPI computation + alerts)
│   ├── ml/
│   │   ├── features.py             # 16 engineered features from raw sensors
│   │   └── train.py                # Model training pipeline with MLflow
│   ├── rag/
│   │   └── engine.py               # LangChain RAG chain over ChromaDB
│   ├── api/
│   │   └── main.py                 # FastAPI with /predict and /ask endpoints
│   ├── dashboard/
│   │   └── app.py                  # Streamlit (4 pages: KPIs, predictor, chat, drift)
│   └── airflow/
│       └── dags/
│           └── retrain_dag.py      # Daily: quality check > train > evaluate > promote
├── tests/
│   └── test_core.py                # 15 tests across all modules
├── docker-compose.yml              # 7 services
├── Dockerfile
├── Makefile                        # All project commands
├── requirements.txt
├── LICENSE
└── README.md
```

## Feature Engineering

16 features derived from 5 raw sensor readings:

| Feature | Derivation | Purpose |
|---------|-----------|---------|
| `temp_diff` | Process temp - Air temp | Heat dissipation failure indicator |
| `power_W` | Torque × (2π × RPM / 60) | Power failure detection |
| `wear_torque_product` | Tool wear × Torque | Overstrain failure indicator |
| `tool_wear_ratio` | Tool wear / 240 | Normalized wear proximity to failure |
| `torque_per_rpm` | Torque / RPM | Cutting efficiency metric |
| `power_low` | Power < 3500W flag | Underpowered condition |
| `power_high` | Power > 9000W flag | Overpowered condition |
| `hdf_risk` | Low temp diff + low RPM | Combined heat dissipation risk |
| `air_temp_norm` | (Air temp - 300) / 2 | Standardized ambient reading |
| `process_temp_norm` | (Process temp - 310) / 1 | Standardized process reading |
| `type_encoded` | Label encoding of L/M/H | Product variant numeric representation |

## RAG Document Coverage

Six manufacturing SOPs are indexed for natural-language retrieval:

1. **Predictive Maintenance Protocol**: Sensor thresholds, failure mode reference, escalation matrix
2. **Tool Wear Management**: Replacement schedules by product variant, wear pattern analysis
3. **Temperature Monitoring**: Heat dissipation failure criteria, coolant system checks
4. **Power and Torque Monitoring**: Safe operating envelope (3,500W to 9,000W), torque specs
5. **Product Quality Variants**: L/M/H specifications, tolerance bands, inspection protocols
6. **Shift Operations and KPIs**: OEE calculation, failure rate targets, reporting cadence

## Airflow DAG

The retraining pipeline runs daily at 2:00 AM:

```
check_data_quality --> train_model --> should_register --> [register | skip] --> refresh_rag
```

- **Data quality gate**: Validates row count, null targets, value ranges, and class distribution
- **Training**: Fits all three model architectures and logs to MLflow
- **Conditional promotion**: New model must exceed 0.80 F1 threshold to be registered
- **RAG refresh**: Re-ingests SOPs to update vector embeddings

## Development

```bash
make test         # Run 15 unit tests
make lint         # Lint with ruff
make format       # Format with ruff
make api          # Start FastAPI locally (no Docker)
make dashboard    # Start Streamlit locally (no Docker)
make clean        # Remove generated data and caches
```

## Limitations and Future Work

**Current limitations:**
- The AI4I dataset is synthetic and contains 10,000 records. Production deployments would require integration with real MES/SCADA systems and significantly larger data volumes.
- The Kafka pipeline replays historical records rather than consuming live sensor feeds. The architecture supports real connectors but the current demo uses simulated streaming.
- RAG responses depend on the 6 bundled SOPs. Adding organization-specific documents would improve retrieval relevance.

**Planned improvements:**
- Integrate SHAP explanations into the prediction API for failure mode attribution
- Add multi-model A/B testing in the Airflow DAG
- Implement WebSocket-based live updates on the Streamlit dashboard
- Deploy to AWS ECS with S3-backed MLflow artifact storage

## Citation

```bibtex
@inproceedings{matzka2020ai4i,
  title={Explainable Artificial Intelligence for Predictive Maintenance Applications},
  author={Matzka, Stephan},
  booktitle={2020 Third International Conference on Artificial Intelligence for Industries (AI4I)},
  pages={69--74},
  year={2020},
  doi={10.1109/AI4I49448.2020.00023}
}
```

## License

MIT. See [LICENSE](LICENSE) for details.
