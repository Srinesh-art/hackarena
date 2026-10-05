from __future__ import annotations
from pydantic import BaseModel, Field, ConfigDict, field_validator

class RecommendRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(min_length=1, max_length=64)
    watch_time_hours: float = Field(ge=0, le=1_000_000)
    top_genres: list[str] = Field(min_length=0, max_length=12)
    avg_session_mins: float = Field(ge=0, le=1_000_000)

    @field_validator("user_id")
    @classmethod
    def clean_user_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("user_id must not be empty")
        return value

    @field_validator("top_genres")
    @classmethod
    def clean_genres(cls, value: list[str]) -> list[str]:
        cleaned, seen = [], set()
        for item in value:
            if not isinstance(item, str):
                raise ValueError("top_genres must contain strings")
            item = item.strip()
            if item and item.lower() not in seen:
                cleaned.append(item)
                seen.add(item.lower())
        return cleaned

class RecommendResponse(BaseModel):
    user_id: str
    segment_id: int
    segment_name: str
    recommendations: list[str]
    distance_to_centroid: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
