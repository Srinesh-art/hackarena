from __future__ import annotations
import logging
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from features import build_feature_matrix, fit_scaler, profile_features
from utils import configure_logging, find_dataset, load_and_clean, save_json, save_preprocessed, NUMERIC_FEATURES

RANDOM_STATE = 42
K_RANGE = range(2, 9)
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))

def recommendation_map(profile: dict) -> list[str]:
    n = profile["numeric_means"]
    genres = [x["genre"] for x in profile["dominant_genres"]]
    engagement = n["avg_watch_time_minutes"] + n["watch_sessions_per_week"]*10 + n["binge_watch_sessions"]*8 + n["completion_rate"]*0.8 + n["content_interactions"]*2 + n["recommendation_click_rate"]*0.5 - n["days_since_last_login"]*1.5
    genre_text = " ".join(genres).lower()
    if engagement >= 320 and any(g in genre_text for g in ["action", "thriller", "horror"]):
        return ["Action & Thriller Deep-Dive", "High-Intensity Movie Picks", "Popular Action Releases"]
    if engagement < 180:
        return ["Trending Popular Movies", "Easy-Completion Picks", "Audience Favorites"]
    if len(genres) >= 3:
        return ["Genre Explorer Mix", "Trending Across Genres", "Curated Discovery Picks"]
    if n["watch_sessions_per_week"] >= 12 and n["avg_watch_time_minutes"] < 130:
        return ["Quick Watch Picks", "Popular Short Sessions", "Trending Movies"]
    primary = genres[0] if genres else "Popular"
    return [f"{primary} Recommendations", f"Popular {primary} Picks", "Personalized Trending Mix"]

def make_segment_name(profile: dict) -> str:
    n = profile["numeric_means"]
    genres = [x["genre"] for x in profile["dominant_genres"]]
    engagement = n["avg_watch_time_minutes"] + n["watch_sessions_per_week"]*10 + n["binge_watch_sessions"]*8 + n["completion_rate"]*0.8 + n["content_interactions"]*2 - n["days_since_last_login"]*1.5
    genre = genres[0] if genres else "General"
    if engagement >= 320 and genre in {"Action", "Thriller", "Horror"}:
        return f"High-Engagement {genre} Viewers"
    if engagement < 180:
        return "Low-Activity Viewers"
    if n["watch_sessions_per_week"] >= 12 and n["avg_watch_time_minutes"] < 130:
        return "Casual Short-Session Viewers"
    if len(genres) >= 3:
        return "Genre Explorers"
    return f"Engaged {genre} Viewers"

def choose_k(X_scaled: np.ndarray):
    evaluations = []
    rng = np.random.default_rng(RANDOM_STATE)
    sample_size = min(15000, len(X_scaled))
    sample_idx = rng.choice(len(X_scaled), size=sample_size, replace=False)
    X_eval = X_scaled[sample_idx]
    for k in K_RANGE:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        sample_labels = model.fit_predict(X_eval)
        score = float(silhouette_score(X_eval, sample_labels, random_state=RANDOM_STATE))
        full_labels = model.predict(X_scaled)
        counts = np.bincount(full_labels, minlength=k)
        percentages = counts / len(full_labels) * 100
        evaluations.append({
            "k": k, "silhouette_score": round(score, 6), "inertia_on_selection_sample": round(float(model.inertia_), 6),
            "cluster_counts_full_data": counts.astype(int).tolist(), "cluster_percentages_full_data": [round(float(x), 4) for x in percentages],
            "min_cluster_percentage": round(float(percentages.min()), 4), "max_cluster_percentage": round(float(percentages.max()), 4),
            "selection_sample_size": sample_size,
        })
    def rank(item: dict) -> float:
        penalty = max(0.0, 5.0 - item["min_cluster_percentage"]) * 0.01
        return item["silhouette_score"] - penalty
    selected = max(evaluations, key=rank)["k"]
    return selected, evaluations

def main() -> None:
    configure_logging()
    log = logging.getLogger("trainer")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    source = find_dataset(DATA_DIR)
    log.info("Loading dataset: %s", source.name)
    df, cleaning_stats = load_and_clean(source)
    preprocessed = save_preprocessed(df, DATA_DIR)
    df = pd.read_csv(preprocessed)
    X, feature_names, genres = build_feature_matrix(df)
    scaler = fit_scaler(X)
    X_scaled = scaler.transform(X)
    selected_k, k_evaluations = choose_k(X_scaled)
    log.info("Selected K=%d", selected_k)
    kmeans = KMeans(n_clusters=selected_k, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    silhouette = float(silhouette_score(X_scaled, labels, sample_size=min(10000, len(X_scaled)), random_state=RANDOM_STATE))
    profiles = profile_features(df, labels, genres)
    for profile in profiles.values():
        profile["segment_name"] = make_segment_name(profile)
        profile["recommendations"] = recommendation_map(profile)
    inference_defaults = {feature: float(df[feature].median()) for feature in NUMERIC_FEATURES}
    artifact = {
        "version": 1, "scaler": scaler, "kmeans": kmeans, "feature_names": feature_names,
        "numeric_features": NUMERIC_FEATURES, "genre_categories": genres,
        "clip_bounds": cleaning_stats["clip_bounds"], "inference_defaults": inference_defaults,
    }
    import joblib
    joblib.dump(artifact, MODEL_DIR / "pipeline.joblib")
    metadata = {
        "version": 1, "algorithm": "KMeans", "random_state": RANDOM_STATE, "selected_k": selected_k,
        "feature_names": feature_names, "numeric_features": NUMERIC_FEATURES, "genre_categories": genres,
        "k_selection": k_evaluations, "final_silhouette_score": silhouette, "final_inertia": float(kmeans.inertia_),
        "cluster_profiles": profiles,
        "recommendation_strategy": "Transparent rule-based mapping from cluster behavioral profile and dominant genre.",
        "preprocessing": cleaning_stats, "preprocessed_dataset": str(preprocessed),
    }
    save_json(metadata, MODEL_DIR / "metadata.json")
    log.info("Model artifacts written to %s", MODEL_DIR)
    log.info("Training complete: rows=%d, features=%d, silhouette=%.4f", len(df), len(feature_names), silhouette)

if __name__ == "__main__":
    main()
