from __future__ import annotations
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from recommender import predict
from schemas import HealthResponse, RecommendRequest, RecommendResponse

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
MODEL_PATH = MODEL_DIR / "pipeline.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"
LOGGER = logging.getLogger("api")
state = {"artifact": None, "metadata": None}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        if MODEL_PATH.exists() and METADATA_PATH.exists():
            state["artifact"] = joblib.load(MODEL_PATH)
            state["metadata"] = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            LOGGER.info("Model artifacts loaded successfully")
        else:
            LOGGER.error("Model artifacts are missing")
    except Exception:
        LOGGER.exception("Failed to load model artifacts")
        state["artifact"] = None
        state["metadata"] = None
    yield

app = FastAPI(title="HACK ARENA OTT Personalization API", version="1.0.0", lifespan=lifespan)

@app.get("/health", response_model=HealthResponse)
def health():
    loaded = state["artifact"] is not None and state["metadata"] is not None
    return {"status": "ok" if loaded else "degraded", "model_loaded": loaded}

@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    if state["artifact"] is None or state["metadata"] is None:
        raise HTTPException(status_code=503, detail="Model is not ready")
    try:
        return predict(request, state["artifact"], state["metadata"])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        LOGGER.exception("Inference failure")
        raise HTTPException(status_code=500, detail="Unable to generate recommendation")

@app.exception_handler(Exception)
async def unexpected_exception_handler(request, exc):
    LOGGER.exception("Unhandled API error")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
