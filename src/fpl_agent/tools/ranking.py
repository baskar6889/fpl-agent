"""Deterministic ranking over canonical players and supplied projections."""

from __future__ import annotations

from fpl_agent.data.schemas import CanonicalDataset
from fpl_agent.projections.interface import ProjectionProvider

from .schemas import RankedPlayer, RankingRequest, RankingResult


def rank_players(
    dataset: CanonicalDataset,
    projection_provider: ProjectionProvider,
    position: str,
    max_price: float | None = None,
    top_n: int = 5,
) -> RankingResult:
    """Rank eligible players using projections supplied by a provider.

    This function consumes already available canonical data and performs no
    file access or API requests.
    """
    request = RankingRequest(position=position, max_price=max_price, top_n=top_n)
    projections = projection_provider.project(dataset)

    projection_by_id = {}
    for projection in projections:
        if projection.player_id in projection_by_id:
            raise ValueError(f"Duplicate projection for player_id {projection.player_id}")
        projection_by_id[projection.player_id] = projection

    player_ids = {player.player_id for player in dataset.players}
    projection_ids = set(projection_by_id)
    missing = player_ids - projection_ids
    unexpected = projection_ids - player_ids
    if missing or unexpected:
        raise ValueError(
            f"Projection player IDs do not match dataset; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )

    eligible = []
    for player in dataset.players:
        if player.position != request.position:
            continue
        if player.status != "a":
            continue
        chance = player.chance_of_playing_next_round
        if chance is not None and chance <= 0:
            continue
        if request.max_price is not None and player.price > request.max_price:
            continue
        eligible.append((player, projection_by_id[player.player_id]))

    eligible.sort(key=lambda pair: (-pair[1].projected_points, pair[0].player_id))
    ranked_players = [
        RankedPlayer(
            rank=index,
            player_id=player.player_id,
            player_name=player.player_name,
            team_id=player.team_id,
            team_name=player.team_name,
            position=player.position,
            price=player.price,
            projected_points=projection.projected_points,
            projection_version=projection.projection_version,
            projection_timestamp=projection.projection_timestamp,
            projection_source=projection.projection_source,
        )
        for index, (player, projection) in enumerate(eligible[: request.top_n], start=1)
    ]
    return RankingResult(
        position=request.position,
        max_price=request.max_price,
        top_n=request.top_n,
        players=ranked_players,
    )