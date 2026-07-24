# ShopFloor AI

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)](docker-compose.yml)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-blue)](http://localhost:5000)

**End-to-end manufacturing intelligence platform** — predicts machine failures from sensor data, streams real-time KPIs, and answers operator questions using RAG over maintenance SOPs.

Built with the [AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020) (10K records, 14 features, 5 failure modes from a milling machine process).

<!-- TODO: Add demo GIF here -->
<!-- ![demo](docs/demo.gif) -->

---

## Features

- **Failure prediction** — classifies machine failure from sensor readings (temperature, torque, speed, tool wear) with 96%+ F1, tracked in MLflow
- **Real-time streaming** — Kafka pipeline ingests sensor data, computes rolling KPIs, triggers anomaly alerts
- **Intelligent Q&A** — RAG system over milling SOPs using LangChain + ChromaDB + OpenAI API
- **Automated retraining** — Airflow DAG runs daily: data quality → train → evaluate → promote → refresh embeddings
- **Live dashboard** — Streamlit app with KPI monitoring, failure predictor, chat interface, drift detection

## Architecture

```
Sensor Data ──→ Kafka (producer/consumer) ──→ Rolling KPIs + Alerts
                         │
                         ▼
                   ML Pipeline ──→ MLflow (tracking + registry)
                         │
SOP Documents ──→ ChromaDB ──→ LangChain RAG ──→ Q&A
                         │
                   Airflow DAG (daily retrain + quality gates)
                         │
                         ▼
                   FastAPI (/predict, /ask, /health)
                         │
                         ▼
                   Streamlit Dashboard (4 pages)
```

## Tech Stack

| Tool | Purpose | Covers |
|------|---------|--------|
| Apache Kafka | Real-time sensor streaming | Streaming / event-driven |
| MLflow | Experiment tracking, model registry | MLOps |
| Apache Airflow | Pipeline orchestration | Workflow automation |
| LangChain + ChromaDB | RAG over maintenance docs | GenAI / LLMs |
| OpenAI API | LLM for document Q&A | LLM integration |
| FastAPI | Production API | Deployment |
| Streamlit + Plotly | Dashboard + chat UI | Visualization |
| Docker Compose | 7-service containerization | Containerization |
| scikit-learn | Failure prediction models | Classical ML |

## Dataset

[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) — CC BY 4.0

| Feature | Description |
|---------|-------------|
| `Air temperature [K]` | Ambient temperature (~300K) |
| `Process temperature [K]` | Process temperature (~310K) |
| `Rotational speed [rpm]` | Tool rotation speed |
| `Torque [Nm]` | Applied torque |
| `Tool wear [min]` | Cumulative tool usage |
| `Type` | Product quality: L (50%), M (30%), H (20%) |
| `Machine failure` | Binary target |
| `TWF, HDF, PWF, OSF, RNF` | 5 failure mode flags |

## Quick Start

### Prerequisites

- Docker + Docker Compose
- Python 3.11+
- OpenAI API key ([platform.openai.com](https://platform.openai.com))

### 1. Clone and configure

```bash
git clone https://github.com/TanayPratapSingh/shopfloor-ai.git
cd shopfloor-ai

cp .env.example .env
# edit .env → add OPENAI_API_KEY
```

### 2. Download data and install

```bash
make setup
```

This downloads the AI4I dataset, installs dependencies, and builds Docker images.

### 3. Start all services

```bash
make up
```

| Service | URL | Credentials |
|---------|-----|-------------|
| Dashboard | http://localhost:8501 | — |
| API Docs | http://localhost:8001/docs | — |
| MLflow | http://localhost:5000 | — |
| Airflow | http://localhost:8080 | admin / admin |

### 4. Train + ingest

```bash
make train    # trains 3 models, logs to MLflow
make ingest   # loads SOPs into ChromaDB for RAG
```

### 5. Stream demo

```bash
# terminal 1
make stream   # Kafka producer (simulates live sensor feed)

# terminal 2
make consume  # Kafka consumer (rolling KPIs + alerts)
```

## Project Structure

```
shopfloor-ai/
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions CI
├── configs/
│   └── config.yaml               # all service configs
├── data/
│   ├── raw/                      # AI4I dataset (gitignored)
│   ├── processed/
│   ├── models/                   # MLflow artifacts
│   └── vectordb/                 # ChromaDB persistence
├── docs/
│   └── architecture.md           # detailed design doc
├── notebooks/
│   └── eda.ipynb                 # exploratory data analysis
├── src/
│   ├── data/
│   │   ├── download.py           # fetches AI4I from UCI
│   │   └── sop_documents.py      # milling SOPs for RAG
│   ├── streaming/
│   │   ├── producer.py           # Kafka sensor stream
│   │   └── consumer.py           # stream processor + alerts
│   ├── ml/
│   │   ├── features.py           # feature engineering
│   │   └── train.py              # model training + MLflow
│   ├── rag/
│   │   └── engine.py             # LangChain RAG chain
│   ├── api/
│   │   └── main.py               # FastAPI backend
│   ├── dashboard/
│   │   └── app.py                # Streamlit frontend
│   └── airflow/
│       └── dags/
│           └── retrain_dag.py    # automated retraining
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   └── test_streaming.py
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── requirements.txt
├── LICENSE
└── README.md
```

## Development

```bash
# run tests
make test

# lint
make lint

# format
make format

# full local run (no Docker)
make api        # start FastAPI
make dashboard  # start Streamlit
```

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

MIT — see [LICENSE](LICENSE).
