from __future__ import annotations
from typing import Any
import numpy as np

def build_vector(request: Any, artifact: dict) -> np.ndarray:
    total_minutes = float(request.watch_time_hours) * 60.0
    estimated_sessions = total_minutes / max(float(request.avg_session_mins), 1.0) if request.avg_session_mins > 0 else 0.0
    values = {
        "avg_watch_time_minutes": total_minutes,
        "watch_sessions_per_week": min(19.0, estimated_sessions),
        "binge_watch_sessions": min(14.0, estimated_sessions * 0.7),
        "completion_rate": 0.0,
        "rating_given": 0.0,
        "content_interactions": 0.0,
        "recommendation_click_rate": 0.0,
        "days_since_last_login": 0.0,
    }
    defaults = artifact.get("inference_defaults", {})
    for key, value in defaults.items():
        values[key] = float(value)
    values["avg_watch_time_minutes"] = total_minutes
    values["watch_sessions_per_week"] = min(19.0, estimated_sessions)
    values["binge_watch_sessions"] = min(14.0, estimated_sessions * 0.7)
    numeric = []
    for feature in artifact["numeric_features"]:
        bound = artifact["clip_bounds"][feature]
        numeric.append(max(bound["lower"], min(bound["upper"], values[feature])))
    genres = artifact["genre_categories"]
    weights = np.zeros(len(genres), dtype=float)
    recognized = []
    lookup = {g.lower(): g for g in genres}
    for genre in request.top_genres:
        canonical = lookup.get(genre.lower())
        if canonical and canonical not in recognized:
            recognized.append(canonical)
    if recognized:
        for genre in recognized:
            weights[genres.index(genre)] = 1.0 / len(recognized)
    return np.asarray(numeric + weights.tolist(), dtype=float).reshape(1, -1)

def predict(request: Any, artifact: dict, metadata: dict) -> dict:
    raw = build_vector(request, artifact)
    scaled = artifact["scaler"].transform(raw)
    model = artifact["kmeans"]
    segment = int(model.predict(scaled)[0])
    centroid = model.cluster_centers_[segment]
    distance = float(np.linalg.norm(scaled[0] - centroid))
    profile = metadata["cluster_profiles"][str(segment)]
    return {
        "user_id": request.user_id,
        "segment_id": segment,
        "segment_name": profile["segment_name"],
        "recommendations": profile["recommendations"],
        "distance_to_centroid": round(distance, 6),
    }
