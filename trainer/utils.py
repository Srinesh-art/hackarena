from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("trainer")
NUMERIC_FEATURES = [
    "avg_watch_time_minutes", "watch_sessions_per_week", "binge_watch_sessions",
    "completion_rate", "rating_given", "content_interactions",
    "recommendation_click_rate", "days_since_last_login",
]
GENRE_COLUMN = "favorite_genre"
REQUIRED_COLUMNS = ["user_id", *NUMERIC_FEATURES, GENRE_COLUMN]

def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def find_dataset(data_dir: Path) -> Path:
    candidates = sorted(p for p in data_dir.glob("*.csv") if p.name.lower() != "preprocessed_users.csv")
    if not candidates:
        raise FileNotFoundError(f"No source CSV dataset found in {data_dir}")
    if len(candidates) > 1:
        LOGGER.warning("Multiple CSV files found; using %s", candidates[0].name)
    return candidates[0]

def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list): return [json_safe(v) for v in value]
    return value

def load_and_clean(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(path)
    original_shape = df.shape
    missing_before = int(df.isna().sum().sum())
    duplicate_count = int(df.duplicated().sum())
    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise ValueError(f"Dataset missing required behavioral columns: {missing_required}")
    df = df.drop_duplicates().copy()
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    imputation_values = {}
    for col in NUMERIC_FEATURES:
        median = float(df[col].median())
        imputation_values[col] = median
        df[col] = df[col].fillna(median)
    df[GENRE_COLUMN] = df[GENRE_COLUMN].fillna("Unknown").astype(str).str.strip().replace({"": "Unknown", "nan": "Unknown"})
    negative_counts = {c: int((df[c] < 0).sum()) for c in NUMERIC_FEATURES}
    for col in NUMERIC_FEATURES:
        df[col] = df[col].clip(lower=0)
    clip_bounds = {}
    for col in NUMERIC_FEATURES:
        low, high = float(df[col].quantile(0.01)), float(df[col].quantile(0.99))
        clip_bounds[col] = {"lower": low, "upper": high}
        df[col] = df[col].clip(lower=low, upper=high)
    stats = {
        "source_file": path.name, "rows_input": int(original_shape[0]), "columns_input": int(original_shape[1]),
        "rows_after_cleaning": int(len(df)), "columns_after_cleaning": int(df.shape[1]),
        "missing_cells_before_cleaning": missing_before, "duplicate_rows_removed": duplicate_count,
        "negative_values_found": negative_counts, "imputation_values": imputation_values,
        "clip_bounds": clip_bounds, "genre_values": sorted(df[GENRE_COLUMN].unique().tolist()),
    }
    return df, json_safe(stats)

def save_preprocessed(df: pd.DataFrame, data_dir: Path) -> Path:
    output = data_dir / "preprocessed_users.csv"
    df.to_csv(output, index=False)
    LOGGER.info("Preprocessed dataset saved to %s", output)
    return output

def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
