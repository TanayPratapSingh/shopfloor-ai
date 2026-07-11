"""Airflow DAG — Daily retraining pipeline with quality gates."""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

default_args = {"owner": "shopfloor-ai", "depends_on_past": False, "retries": 1, "retry_delay": timedelta(minutes=5)}

dag = DAG("shopfloor_retrain", default_args=default_args, description="Daily model retrain",
          schedule="0 2 * * *", start_date=datetime(2026, 7, 1), catchup=False, tags=["ml", "shopfloor"])


def _check_data(**ctx):
    import pandas as pd
    df = pd.read_csv("/opt/airflow/data/raw/ai4i2020.csv")
    assert len(df) >= 1000, "Not enough data"
    assert df["Machine failure"].isna().sum() == 0, "Null targets"
    assert df["Machine failure"].nunique() > 1, "Single class"
    ctx["ti"].xcom_push(key="n_rows", value=len(df))
    ctx["ti"].xcom_push(key="failure_rate", value=float(df["Machine failure"].mean()))


def _train(**ctx):
    import sys; sys.path.insert(0, "/opt/airflow")
    from src.ml.train import train_and_evaluate
    results = train_and_evaluate(register_best=True)
    best = max(results, key=lambda k: results[k]["f1"])
    ctx["ti"].xcom_push(key="best_model", value=best)
    ctx["ti"].xcom_push(key="best_f1", value=results[best]["f1"])


def _should_register(**ctx):
    f1 = ctx["ti"].xcom_pull(task_ids="train", key="best_f1")
    return "register" if f1 and f1 >= 0.80 else "skip"


def _register(**ctx):
    model = ctx["ti"].xcom_pull(task_ids="train", key="best_model")
    f1 = ctx["ti"].xcom_pull(task_ids="train", key="best_f1")
    print(f"Registered {model} (F1={f1:.4f})")


def _refresh_rag(**ctx):
    import sys; sys.path.insert(0, "/opt/airflow")
    from src.rag.engine import RAGEngine
    engine = RAGEngine()
    n = engine.ingest()
    print(f"Refreshed {n} RAG chunks")


check = PythonOperator(task_id="check_data", python_callable=_check_data, dag=dag)
train = PythonOperator(task_id="train", python_callable=_train, dag=dag)
branch = BranchPythonOperator(task_id="should_register", python_callable=_should_register, dag=dag)
register = PythonOperator(task_id="register", python_callable=_register, dag=dag)
skip = EmptyOperator(task_id="skip", dag=dag)
rag = PythonOperator(task_id="refresh_rag", python_callable=_refresh_rag, dag=dag, trigger_rule="none_failed_min_one_success")

check >> train >> branch >> [register, skip] >> rag
