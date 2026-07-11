"""ML Pipeline — Failure Prediction with MLflow Tracking."""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
)
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
from loguru import logger
import yaml

from src.data.download import load_dataset
from src.ml.features import engineer_features


def load_config() -> dict:
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


def train_and_evaluate(register_best: bool = True) -> dict:
    config = load_config()
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    df = load_dataset()
    X, y = engineer_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)} | Failure rate: {y.mean():.2%}")

    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "RandomForest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=12, min_samples_split=5,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )),
        ]),
        "GradientBoosting": Pipeline([
            ("clf", GradientBoostingClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, random_state=42,
            )),
        ]),
    }

    results = {}

    for name, pipeline in models.items():
        with mlflow.start_run(run_name=name):
            logger.info(f"Training {name}...")
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_proba),
            }
            cv = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1")
            metrics["cv_f1_mean"] = cv.mean()
            metrics["cv_f1_std"] = cv.std()

            mlflow.log_params({
                "model_type": name,
                "n_features": X_train.shape[1],
                "n_train": len(X_train),
                "dataset": "AI4I_2020",
                "features": json.dumps(list(X.columns)),
            })
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipeline, artifact_path="model", input_example=X_test.iloc[:3])

            clf = pipeline.named_steps.get("clf")
            if hasattr(clf, "feature_importances_"):
                imp = pd.DataFrame({"feature": X.columns, "importance": clf.feature_importances_})
                imp = imp.sort_values("importance", ascending=False)
                imp.to_csv("/tmp/feature_importance.csv", index=False)
                mlflow.log_artifact("/tmp/feature_importance.csv")

            mlflow.log_text(classification_report(y_test, y_pred), "classification_report.txt")
            results[name] = metrics
            logger.info(f"  {name}: F1={metrics['f1']:.4f} | AUC={metrics['roc_auc']:.4f}")

    best_name = max(results, key=lambda k: results[k]["f1"])
    logger.info(f"\nBest: {best_name} (F1={results[best_name]['f1']:.4f})")

    if register_best:
        runs = mlflow.search_runs(
            filter_string=f"params.model_type = '{best_name}'",
            order_by=["metrics.f1 DESC"], max_results=1,
        )
        if not runs.empty:
            run_id = runs.iloc[0]["run_id"]
            mlflow.register_model(f"runs:/{run_id}/model", config["mlflow"]["model_name"])
            logger.info(f"Registered '{best_name}' as '{config['mlflow']['model_name']}'")

    return results


if __name__ == "__main__":
    results = train_and_evaluate()
    print("\n=== Results ===")
    for name, m in results.items():
        print(f"  {name}: F1={m['f1']:.4f} | AUC={m['roc_auc']:.4f}")
