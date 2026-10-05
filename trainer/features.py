from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils import NUMERIC_FEATURES, GENRE_COLUMN

def build_feature_matrix(df: pd.DataFrame, genre_categories: list[str] | None = None):
    genres = genre_categories or sorted(df[GENRE_COLUMN].astype(str).unique().tolist())
    numeric = df[NUMERIC_FEATURES].to_numpy(dtype=float)
    genre_matrix = np.zeros((len(df), len(genres)), dtype=float)
    genre_to_idx = {g: i for i, g in enumerate(genres)}
    for row, genre in enumerate(df[GENRE_COLUMN].astype(str)):
        if genre in genre_to_idx:
            genre_matrix[row, genre_to_idx[genre]] = 1.0
    X = np.hstack([numeric, genre_matrix])
    feature_names = NUMERIC_FEATURES + [f"genre__{g}" for g in genres]
    return X, feature_names, genres

def fit_scaler(X: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler

def profile_features(df: pd.DataFrame, labels: np.ndarray, genre_categories: list[str]):
    profiles = {}
    work = df.copy()
    work["_cluster"] = labels
    for cluster_id, group in work.groupby("_cluster", sort=True):
        genre_counts = group[GENRE_COLUMN].value_counts(normalize=True)
        profiles[str(int(cluster_id))] = {
            "count": int(len(group)),
            "percentage": round(float(len(group) / len(work) * 100), 3),
            "numeric_means": {col: round(float(group[col].mean()), 3) for col in NUMERIC_FEATURES},
            "dominant_genres": [
                {"genre": str(g), "share": round(float(s), 4)}
                for g, s in genre_counts.head(3).items()
            ],
        }
    return profiles
