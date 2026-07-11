"""Tests for ShopFloor AI core components."""

import pytest
import pandas as pd
import numpy as np


class TestDataDownload:
    def test_load_returns_dataframe(self):
        from src.data.download import load_dataset
        df = load_dataset()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10000

    def test_required_columns(self):
        from src.data.download import load_dataset
        df = load_dataset()
        required = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
                     "Torque [Nm]", "Tool wear [min]", "Type", "Machine failure"]
        for col in required:
            assert col in df.columns, f"Missing: {col}"

    def test_failure_modes_present(self):
        from src.data.download import load_dataset
        df = load_dataset()
        for mode in ["TWF", "HDF", "PWF", "OSF", "RNF"]:
            assert mode in df.columns
            assert df[mode].sum() > 0

    def test_product_types(self):
        from src.data.download import load_dataset
        df = load_dataset()
        assert set(df["Type"].unique()) == {"L", "M", "H"}

    def test_schema_info(self):
        from src.data.download import get_schema_info
        info = get_schema_info()
        assert info["rows"] == 10000
        assert info["target"] == "Machine failure"


class TestFeatures:
    def test_engineer_features_shape(self):
        from src.data.download import load_dataset
        from src.ml.features import engineer_features
        df = load_dataset()
        X, y = engineer_features(df)
        assert len(X) == len(y) == 10000
        assert X.shape[1] >= 10

    def test_no_nans(self):
        from src.data.download import load_dataset
        from src.ml.features import engineer_features
        df = load_dataset()
        X, y = engineer_features(df)
        assert X.isna().sum().sum() == 0

    def test_engineered_columns_exist(self):
        from src.data.download import load_dataset
        from src.ml.features import engineer_features
        df = load_dataset()
        X, _ = engineer_features(df)
        for col in ["temp_diff", "power_W", "wear_torque_product", "tool_wear_ratio"]:
            assert col in X.columns, f"Missing engineered feature: {col}"

    def test_power_calculation(self):
        from src.data.download import load_dataset
        from src.ml.features import engineer_features
        df = load_dataset()
        X, _ = engineer_features(df)
        assert X["power_W"].min() > 0
        assert X["power_W"].max() < 20000


class TestStreaming:
    def test_stream_processor(self):
        from src.streaming.consumer import StreamProcessor
        proc = StreamProcessor(window=10)
        record = {"Machine failure": 0, "Torque [Nm]": 40.0,
                  "Process temperature [K]": 310, "Air temperature [K]": 300,
                  "Tool wear [min]": 100}
        enriched = proc.process(record)
        assert "rolling_failure_rate" in enriched
        assert "rolling_torque_avg" in enriched
        assert enriched["window_size"] == 1

    def test_alert_on_high_failure(self):
        from src.streaming.consumer import StreamProcessor
        proc = StreamProcessor(window=5)
        for _ in range(5):
            r = {"Machine failure": 1, "Torque [Nm]": 40, "Tool wear [min]": 100,
                 "Process temperature [K]": 310, "Air temperature [K]": 300}
            enriched = proc.process(r)
        alerts = proc.check_alerts(enriched)
        types = [a["type"] for a in alerts]
        assert "FAILURE_RATE_HIGH" in types

    def test_alert_on_high_torque(self):
        from src.streaming.consumer import StreamProcessor
        proc = StreamProcessor(window=5)
        r = {"Machine failure": 0, "Torque [Nm]": 70, "Tool wear [min]": 50,
             "Process temperature [K]": 310, "Air temperature [K]": 300}
        enriched = proc.process(r)
        alerts = proc.check_alerts(enriched)
        types = [a["type"] for a in alerts]
        assert "TORQUE_SPIKE" in types

    def test_alert_on_tool_wear(self):
        from src.streaming.consumer import StreamProcessor
        proc = StreamProcessor(window=5)
        r = {"Machine failure": 0, "Torque [Nm]": 40, "Tool wear [min]": 220,
             "Process temperature [K]": 310, "Air temperature [K]": 300}
        enriched = proc.process(r)
        alerts = proc.check_alerts(enriched)
        types = [a["type"] for a in alerts]
        assert "TOOL_WEAR_HIGH" in types


class TestSOPs:
    def test_sop_count(self):
        from src.data.sop_documents import get_sop_documents
        docs = get_sop_documents()
        assert len(docs) >= 5

    def test_sop_structure(self):
        from src.data.sop_documents import get_sop_documents
        for doc in get_sop_documents():
            assert "title" in doc
            assert "category" in doc
            assert "content" in doc
            assert len(doc["content"]) > 200
