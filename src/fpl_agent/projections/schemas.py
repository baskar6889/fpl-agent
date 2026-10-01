"""Validated projection output records."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PlayerProjection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    player_id: int = Field(gt=0)
    projected_points: float
    projection_version: str = Field(min_length=1)
    projection_timestamp: datetime
    projection_source: str = Field(min_length=1)