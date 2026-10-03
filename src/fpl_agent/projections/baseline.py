"""Simple season scoring-rate baseline, version 1."""

from __future__ import annotations

from datetime import datetime, timezone

from fpl_agent.data.schemas import CanonicalDataset

from .schemas import PlayerProjection


class BaselineV1:
    """Scale each player's season points-per-minute rate to expected minutes."""

    version = "baseline_v1"
    source = "season_scoring_rate"

    def __init__(self, expected_minutes: float = 60.0) -> None:
        if expected_minutes < 0:
            raise ValueError("expected_minutes must be non-negative")
        self.expected_minutes = expected_minutes

    def project(self, dataset: CanonicalDataset) -> list[PlayerProjection]:
        timestamp = datetime.now(timezone.utc)
        projections = []
        for player in dataset.players:
            if player.minutes == 0:
                points = 0.0
            else:
                base_scoring_rate = (player.total_points / player.minutes) * self.expected_minutes
                if dataset.gameweeks_elapsed > 0:
                    historical_availability = min(
                        1.0, player.minutes / (dataset.gameweeks_elapsed * 60)
                    )
                else:
                    historical_availability = 1.0
                chance = player.chance_of_playing_next_round
                next_round_probability = chance / 100 if chance is not None else 1.0
                points = (
                    base_scoring_rate
                    * historical_availability
                    * next_round_probability
                )
            projections.append(
                PlayerProjection(
                    player_id=player.player_id,
                    projected_points=round(points, 2),
                    projection_version=self.version,
                    projection_timestamp=timestamp,
                    projection_source=self.source,
                )
            )
        return projections