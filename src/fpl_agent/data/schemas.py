"""Pydantic schema for canonical player records."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CanonicalPlayer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    player_id: int = Field(gt=0)
    player_name: str = Field(min_length=1)
    team_id: int = Field(gt=0)
    team_name: str = Field(min_length=1)
    position: str
    price: float = Field(ge=0)
    total_points: int
    minutes: int = Field(ge=0)
    status: str
    chance_of_playing_next_round: int | None = Field(default=None, ge=0, le=100)


class CanonicalDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    players: list[CanonicalPlayer]
