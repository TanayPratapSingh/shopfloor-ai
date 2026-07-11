"""
Downloads AI4I 2020 Predictive Maintenance Dataset from UCI ML Repository.
Source: https://archive.ics.uci.edu/dataset/601
License: CC BY 4.0
"""

import os
import pandas as pd
from pathlib import Path
from loguru import logger

DATA_DIR = Path("data/raw")
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
FILENAME = "ai4i2020.csv"


def download_dataset(force: bool = False) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filepath = DATA_DIR / FILENAME

    if filepath.exists() and not force:
        logger.info(f"Dataset already exists at {filepath}")
        return pd.read_csv(filepath)

    logger.info(f"Downloading AI4I 2020 dataset from UCI...")
    try:
        df = pd.read_csv(DATASET_URL)
    except Exception:
        logger.warning("UCI direct download failed, trying alternate URL...")
        alt_url = "https://raw.githubusercontent.com/stephanmatzka/ai4i-2020-predictive-maintenance-dataset/main/ai4i2020.csv"
        df = pd.read_csv(alt_url)

    df.to_csv(filepath, index=False)
    logger.info(f"Saved {len(df)} rows to {filepath}")
    return df


def load_dataset() -> pd.DataFrame:
    filepath = DATA_DIR / FILENAME
    if not filepath.exists():
        return download_dataset()
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns from {filepath}")
    return df


def get_schema_info() -> dict:
    return {
        "name": "AI4I 2020 Predictive Maintenance Dataset",
        "source": "UCI ML Repository (doi:10.24432/C5HS5C)",
        "license": "CC BY 4.0",
        "rows": 10000,
        "columns": 14,
        "target": "Machine failure",
        "features": {
            "UDI": "Unique identifier (1-10000)",
            "Product ID": "Product variant + serial (e.g. L50042)",
            "Type": "Quality variant: L (50%), M (30%), H (20%)",
            "Air temperature [K]": "Ambient temp, ~300K, std 2K",
            "Process temperature [K]": "Process temp, air+10K, std 1K",
            "Rotational speed [rpm]": "Tool rotation, ~2860W power base",
            "Torque [Nm]": "Applied torque, ~40Nm mean, std 10",
            "Tool wear [min]": "Cumulative wear, H/M/L variants differ",
            "Machine failure": "Binary target (0/1)",
            "TWF": "Tool Wear Failure",
            "HDF": "Heat Dissipation Failure",
            "PWF": "Power Failure",
            "OSF": "Overstrain Failure",
            "RNF": "Random Failure",
        },
        "failure_rates": {
            "TWF": 46, "HDF": 115, "PWF": 95, "OSF": 98, "RNF": 5,
            "total_failures": 339,
        },
    }


if __name__ == "__main__":
    df = download_dataset()
    info = get_schema_info()
    print(f"\n{info['name']}")
    print(f"Shape: {df.shape}")
    print(f"Failure rate: {df['Machine failure'].mean():.2%}")
    print(f"\nFailure mode counts:")
    for mode in ["TWF", "HDF", "PWF", "OSF", "RNF"]:
        print(f"  {mode}: {df[mode].sum()}")
    print(f"\nProduct types: {df['Type'].value_counts().to_dict()}")
