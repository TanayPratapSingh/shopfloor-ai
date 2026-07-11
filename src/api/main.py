"""FastAPI Backend — ML predictions, RAG Q&A, health checks."""

import os, yaml
from datetime import datetime
from contextlib import asynccontextmanager
import pandas as pd
import mlflow, mlflow.sklearn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger
from src.rag.engine import RAGEngine
from src.ml.features import engineer_features


class SensorInput(BaseModel):
    air_temp_k: float = Field(300.0, alias="Air temperature [K]")
    process_temp_k: float = Field(310.0, alias="Process temperature [K]")
    rpm: float = Field(1500.0, alias="Rotational speed [rpm]")
    torque_nm: float = Field(40.0, alias="Torque [Nm]")
    tool_wear_min: float = Field(100.0, alias="Tool wear [min]")
    product_type: str = Field("M", alias="Type")

    class Config:
        populate_by_name = True


class QuestionInput(BaseModel):
    question: str = Field(..., min_length=3)


class AppState:
    model = None
    rag: RAGEngine = None
    config: dict = {}

state = AppState()


def load_config():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.config = load_config()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", state.config["mlflow"]["tracking_uri"]))
    try:
        state.model = mlflow.sklearn.load_model(f"models:/{state.config['mlflow']['model_name']}/latest")
        logger.info("Model loaded from MLflow")
    except Exception as e:
        logger.warning(f"No model loaded: {e}")
    try:
        state.rag = RAGEngine(state.config)
        state.rag.build_chain()
        logger.info("RAG engine ready")
    except Exception as e:
        logger.warning(f"RAG not ready: {e}")
    yield


app = FastAPI(title="ShopFloor AI", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health():
    return {"status": "healthy", "model": state.model is not None, "rag": state.rag is not None and state.rag.qa_chain is not None}


@app.post("/predict")
async def predict(sensor: SensorInput):
    if state.model is None:
        raise HTTPException(503, "No model loaded")
    row = {
        "Air temperature [K]": sensor.air_temp_k,
        "Process temperature [K]": sensor.process_temp_k,
        "Rotational speed [rpm]": sensor.rpm,
        "Torque [Nm]": sensor.torque_nm,
        "Tool wear [min]": sensor.tool_wear_min,
        "Type": sensor.product_type,
        "Machine failure": 0,
    }
    df = pd.DataFrame([row])
    X, _ = engineer_features(df)
    proba = state.model.predict_proba(X)[0][1]

    risks = []
    if sensor.tool_wear_min > 200:
        risks.append(f"Tool wear {sensor.tool_wear_min:.0f} min — near replacement threshold")
    power = sensor.torque_nm * (2 * 3.14159 * sensor.rpm / 60)
    if power < 3500 or power > 9000:
        risks.append(f"Power {power:.0f}W outside safe range (3500-9000W)")
    temp_diff = sensor.process_temp_k - sensor.air_temp_k
    if temp_diff < 8.6 and sensor.rpm < 1380:
        risks.append("HDF risk: low temp differential + low RPM")
    if sensor.torque_nm > 55:
        risks.append(f"Torque {sensor.torque_nm:.1f} Nm elevated (normal: ~40 Nm)")

    return {"failure_prediction": bool(proba >= 0.5), "failure_probability": round(float(proba), 4), "risk_factors": risks}


@app.post("/ask")
async def ask(q: QuestionInput):
    if state.rag is None or state.rag.qa_chain is None:
        raise HTTPException(503, "RAG not ready")
    return state.rag.ask(q.question)


@app.post("/rag/ingest")
async def ingest():
    if state.rag is None:
        state.rag = RAGEngine(state.config)
    n = state.rag.ingest()
    state.rag.build_chain()
    return {"chunks_ingested": n}
