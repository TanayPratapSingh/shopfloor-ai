"""Feature engineering for AI4I 2020 Predictive Maintenance Dataset."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


FEATURE_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build ML features from raw AI4I sensor data."""
    df = df.copy()

    # ── Raw sensor features ──
    features = list(FEATURE_COLUMNS)

    # ── Encoded product type ──
    le = LabelEncoder()
    df["type_encoded"] = le.fit_transform(df["Type"])
    features.append("type_encoded")

    # ── Engineered features ──

    # temp differential (key indicator for HDF)
    df["temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    features.append("temp_diff")

    # power (key indicator for PWF)
    df["power_W"] = df["Torque [Nm]"] * (2 * np.pi * df["Rotational speed [rpm]"] / 60)
    features.append("power_W")

    # overstrain product (key indicator for OSF)
    df["wear_torque_product"] = df["Tool wear [min]"] * df["Torque [Nm]"]
    features.append("wear_torque_product")

    # tool wear ratio to max life
    df["tool_wear_ratio"] = df["Tool wear [min]"] / 240.0
    features.append("tool_wear_ratio")

    # torque per rpm (cutting efficiency)
    df["torque_per_rpm"] = df["Torque [Nm]"] / (df["Rotational speed [rpm]"] + 1)
    features.append("torque_per_rpm")

    # power zone flags
    df["power_low"] = (df["power_W"] < 3500).astype(int)
    df["power_high"] = (df["power_W"] > 9000).astype(int)
    features.extend(["power_low", "power_high"])

    # HDF risk flag (both conditions approaching)
    df["hdf_risk"] = ((df["temp_diff"] < 10) & (df["Rotational speed [rpm]"] < 1500)).astype(int)
    features.append("hdf_risk")

    # temp normalized
    df["air_temp_norm"] = (df["Air temperature [K]"] - 300) / 2
    df["process_temp_norm"] = (df["Process temperature [K]"] - 310) / 1
    features.extend(["air_temp_norm", "process_temp_norm"])

    X = df[features].fillna(0)
    y = df["Machine failure"].astype(int)

    return X, y


def get_feature_names() -> list[str]:
    """Return the full list of engineered feature names."""
    return [
        *FEATURE_COLUMNS,
        "type_encoded", "temp_diff", "power_W", "wear_torque_product",
        "tool_wear_ratio", "torque_per_rpm", "power_low", "power_high",
        "hdf_risk", "air_temp_norm", "process_temp_norm",
    ]
