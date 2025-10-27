import os
import json
import joblib
from datetime import datetime
import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    # Expect columns: text, label
    expected = {"text", "label"}
    missing = expected - set(df.columns.str.lower())
    if missing:
        raise ValueError(
            f"CSV must contain columns: text,label (missing: {', '.join(missing)})"
        )
    # Normalize column names just in case
    df = df.rename(columns={"Text": "text", "Label": "label"})
    # Drop obvious bad rows
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    return df


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def save_model(model, path: str, meta: dict | None = None) -> None:
    ensure_dir(os.path.dirname(path))
    joblib.dump(model, path)
    if meta is not None:
        meta_path = os.path.splitext(path)[0] + "_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)
