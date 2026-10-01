"""Validated request and response models for player ranking."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Position = Literal["GK", "DEF", "MID", "FWD"]


class RankingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    position: Position
    max_price: float | None = Field(default=None, ge=0)
    top_n: int = Field(default=5, gt=0)


class RankedPlayer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int = Field(gt=0)
    player_id: int = Field(gt=0)
    player_name: str = Field(min_length=1)
    team_id: int = Field(gt=0)
    team_name: str = Field(min_length=1)
    position: Position
    price: float = Field(ge=0)
    projected_points: float
    projection_version: str = Field(min_length=1)
    projection_timestamp: datetime
    projection_source: str = Field(min_length=1)


class RankingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    position: Position
    max_price: float | None
    top_n: int
    players: list[RankedPlayer]