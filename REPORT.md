# HACK ARENA Technical Report

## 1. Executive Summary
This implementation provides a lightweight, reproducible OTT audience segmentation and personalization system with three independent services: trainer, FastAPI API, and evaluator. It uses unsupervised KMeans clustering, persists the complete inference artifact, serves segment-aware recommendations, and automatically generates evaluation evidence.

## 2. Problem Understanding
The objective is to transform raw OTT viewer behavior into useful audience segments without manually assigned labels, then expose those segments through a REST API. The dataset's `churned` field is explicitly excluded from model training so the core task remains unsupervised.

## 3. Dataset Description
Source file: `data/netflix_user_behavior_dataset.csv`.

Initial inspection:
- 50,000 rows
- 20 columns
- 0 missing cells
- 0 duplicate rows
- 8 categorical fields plus behavioral/numeric fields and identifiers
- 8 unique favorite genres: Action, Comedy, Documentary, Drama, Horror, Romance, Sci-Fi, Thriller
- Behavioral numeric ranges include watch time 10–299 minutes, sessions/week 1–19, binge sessions 0–14, completion 30–99, ratings 1–5, interactions 0–49, recommendation click rate 0–99, and days since login 0–59.

## 4. Preprocessing
The trainer performs schema validation, duplicate removal, numeric coercion, infinity handling, median imputation for missing behavioral values, non-negative clipping, and deterministic 1st/99th percentile clipping. The cleaned dataset is persisted as `data/preprocessed_users.csv` and is the dataset used by the clustering stage.

The current supplied dataset had no missing values or duplicates, so the corresponding cleaning operations were verified but did not remove records.

## 5. Feature Selection
The clustering representation contains:
- `avg_watch_time_minutes`
- `watch_sessions_per_week`
- `binge_watch_sessions`
- `completion_rate`
- `rating_given`
- `content_interactions`
- `recommendation_click_rate`
- `days_since_last_login`
- one-hot `favorite_genre` features for the eight observed genres

This produces 16 model features. `user_id`, demographics, country, payment method, subscription, monthly fee, primary device, account age, and `churned` are excluded to keep the representation behavior-focused and prevent label leakage.

## 6. Model Architecture
The inference pipeline is:

`behavioral features -> genre encoding -> StandardScaler -> KMeans`

KMeans uses `random_state=42` and `n_init=10` for deterministic, CPU-friendly training.

## 7. K Selection
Candidate K values 2 through 8 were evaluated. To keep the 50,000-row dataset CPU-friendly during the timed hackathon, K selection uses a deterministic 15,000-row sample. Each candidate is fitted on that sample, silhouette is measured on the same sample, and cluster balance is projected across the full dataset.

| K | Silhouette | Full-data min cluster % | Full-data max cluster % |
|---:|---:|---:|---:|
| 2 | 0.082700 | 37.438 | 62.562 |
| 3 | 0.123777 | 12.470 | 75.012 |
| 4 | 0.165355 | 12.406 | 62.606 |
| 5 | 0.206296 | 12.406 | 37.646 |
| 6 | 0.246779 | 12.470 | 25.010 |
| 7 | 0.287920 | 12.406 | 25.082 |
| 8 | 0.328758 | 12.378 | 12.704 |

K=8 was selected because it produced the strongest silhouette score while also producing exceptionally balanced clusters. The final full-data model achieved a silhouette score of **0.328585** and inertia of **399937.3245**.

## 8. Cluster Profiles
The selected eight clusters are highly balanced:

| Cluster | Size | Share | Segment |
|---:|---:|---:|---|
| 0 | 6,189 | 12.378% | Engaged Sci-Fi Viewers |
| 1 | 6,257 | 12.514% | High-Engagement Thriller Viewers |
| 2 | 6,223 | 12.446% | High-Engagement Horror Viewers |
| 3 | 6,259 | 12.518% | Engaged Romance Viewers |
| 4 | 6,352 | 12.704% | Engaged Comedy Viewers |
| 5 | 6,235 | 12.470% | High-Engagement Action Viewers |
| 6 | 6,203 | 12.406% | Engaged Documentary Viewers |
| 7 | 6,282 | 12.564% | Engaged Drama Viewers |

The cluster imbalance ratio is only **1.0263**, indicating no dominant or pathological tiny cluster.

## 9. Segment Naming
Names are generated from observed cluster-level engagement statistics and dominant genres. They are not assigned arbitrarily from cluster IDs.

## 10. Personalization Logic
A transparent rule-based recommendation layer maps the predicted segment to a small content catalog. High-engagement Action/Thriller/Horror clusters receive deeper high-intensity recommendations; lower-engagement segments receive popular and easy-completion content; other segments receive genre-specific or discovery recommendations.

The API accepts `watch_time_hours`, `avg_session_mins`, and `top_genres`. When richer behavioral fields are unavailable, training medians are used as neutral defaults. `watch_sessions_per_week` and `binge_watch_sessions` are additionally estimated from total watch minutes and average session duration, making both supplied API behavior fields meaningful at inference time.

## 11. API Design
Endpoints:

- `GET /health`
- `POST /recommend`

The API loads `models/pipeline.joblib` and `models/metadata.json` exactly once at startup. It never retrains during inference.

Example valid response:
```json
{
  "user_id": "USR-8192",
  "segment_id": 1,
  "segment_name": "High-Engagement Thriller Viewers",
  "recommendations": [
    "Action & Thriller Deep-Dive",
    "High-Intensity Movie Picks",
    "Popular Action Releases"
  ],
  "distance_to_centroid": 2.73105
}
```

## 12. Validation and Error Handling
Pydantic rejects malformed JSON types, missing required fields, negative values, excessive values, unknown extra fields, and invalid genre item types. Unknown genres are ignored and empty genre lists are accepted. Model-unavailable inference returns HTTP 503. Unexpected inference failures return a safe HTTP 500 response without exposing stack traces.

## 13. Docker Architecture
The Compose stack contains:
- `trainer`: mounts `data` and `models`, creates the preprocessing artifact and persisted model, then exits.
- `api`: mounts the model directory read-only, exposes port 8000, and uses a healthcheck that requires `model_loaded=true`.
- `evaluator`: waits on API health and writes `results/metrics.json`.

Each container uses a Python 3.12 slim image, pinned dependencies, and a non-root runtime user. Per-service `.dockerignore` files keep build contexts lean.

## 14. Evaluation Results
Local end-to-end application verification completed successfully.

- Health check: passed
- Valid recommendation: passed
- Unknown genre: passed
- Empty genre list: passed
- Zero watch time: passed
- Extreme values: passed
- Missing required field: passed
- Invalid numeric type: passed
- Negative watch time: passed
- Negative session duration: passed
- Repeated request determinism: passed
- **11/11 tests passed (100% pass rate)**
- Measured API latency sample: approximately 246.9 ms for the first valid request in the final local evaluator run; subsequent individual requests were substantially lower after warm-up.

The authoritative machine-readable results are in `results/metrics.json`.

## 15. Reproducibility
The deterministic seed is 42. Model selection uses a deterministic 15,000-row sample, and final KMeans uses deterministic initialization. Repeated identical API requests returned identical responses.

## 16. Docker Verification Status
The source, trainer, persisted model, API, evaluator, and local application tests have been verified. A Docker Compose build could not be executed on the current machine because the Docker CLI and Docker Desktop executable are not installed/available in PATH. Therefore no Docker build/run result is claimed here. Once Docker Desktop is installed, the final validation command is:

```powershell
cd D:\Movie
docker compose down -v
docker compose build --no-cache
docker compose up
```

## 17. Limitations
The challenge API contract exposes fewer behavioral dimensions than the training dataset. Neutral training medians and deterministic derived proxies are therefore used for unavailable dimensions. The recommendation catalog is intentionally rule-based rather than a full collaborative/content-based recommender.

## 18. Future Improvements
Potential production extensions include richer request features, online feature aggregation, drift detection, scheduled retraining, model versioning, authenticated access, a larger content catalog, and A/B testing of recommendation policies.

## 19. Development Notes
The implementation deliberately avoids LLMs, paid APIs, and unnecessary infrastructure. The system remains CPU-friendly and focused on proving the complete segmentation-to-personalization workflow.


## 25. GitHub Repository & Version Control

The completed project was committed to Git version control and pushed to the official GitHub repository:

https://github.com/Srinesh-art/hackarena.git

The repository contains the Docker Compose configuration, Trainer, API, Evaluator, preprocessing outputs, model artifacts, evaluation results, README, REPORT.md, and the presentation-ready Word report. It provides a reproducible source-control snapshot of the implemented hackathon system.

Initial project commit: b3a29fa — feat: containerized OTT audience segmentation system.

Repository: Srinesh-art/hackarena | Default branch: main.
