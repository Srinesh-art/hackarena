# HACK ARENA — OTT Audience Segmentation & Personalization

## Overview
A lightweight, CPU-friendly, containerized audience segmentation service using behavioral KMeans clustering and a FastAPI personalization API.

## Architecture
`dataset -> trainer -> persisted model -> API -> evaluator -> metrics.json`

Services: `trainer`, `api`, `evaluator`.

## Dataset
Source dataset: `data/netflix_user_behavior_dataset.csv`. The trainer validates the behavioral schema, removes duplicates, handles missing/invalid numeric values, clips pathological values deterministically, writes `data/preprocessed_users.csv`, and trains only on behavioral features. The `churned` field is never used as a training target.

## Run
```powershell
cd D:\Movie
docker compose up --build
```

Clean rebuild:
```powershell
docker compose down -v
docker compose build --no-cache
docker compose up
```

## API
Health:
```powershell
Invoke-RestMethod http://localhost:8000/health
```

Recommendation:
```powershell
$body = @{ user_id="USR-8192"; watch_time_hours=32.5; top_genres=@("Action","Thriller"); avg_session_mins=85.0 } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/recommend -Method Post -ContentType "application/json" -Body $body
```

## Outputs
- `models/pipeline.joblib` — persisted scaler + KMeans + inference metadata
- `models/metadata.json` — training, cluster, and recommendation metadata
- `data/preprocessed_users.csv` — cleaned dataset used by training
- `results/metrics.json` — evaluator evidence
- `REPORT.md` — technical report

## Notes
The API exposes the challenge's stable profile fields. Behavioral dimensions not supplied by the request use actual training medians stored in the model artifact. Unknown genres are ignored; empty genre lists are valid. The API never retrains.
