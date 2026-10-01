"""Deterministic guardrails around the player ranking tool."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from fpl_agent.data.schemas import CanonicalDataset
from fpl_agent.projections.interface import ProjectionProvider
from fpl_agent.tools.ranking import rank_players
from fpl_agent.tools.schemas import RankingResult

from .schemas import GuardrailError, GuardrailErrorCode, GuardrailResult

SUPPORTED_ACTION = "rank_players"
SUPPORTED_POSITIONS = frozenset({"DEF", "MID", "FWD"})
MAX_TOP_N = 20


def _failure(
    code: GuardrailErrorCode, message: str, details: dict[str, Any] | None = None
) -> GuardrailResult:
    return GuardrailResult(error=GuardrailError(code=code, message=message, details=details))


def _validate_output(
    result: Any,
    dataset: CanonicalDataset,
    position: str,
    max_price: float | None,
    top_n: int,
) -> RankingResult:
    if isinstance(result, RankingResult):
        validated = RankingResult.model_validate(result.model_dump())
    else:
        validated = RankingResult.model_validate(result)

    if (
        validated.position != position
        or validated.max_price != max_price
        or validated.top_n != top_n
        or len(validated.players) > top_n
    ):
        raise ValueError("Ranking response does not match the validated request")

    players_by_id = {player.player_id: player for player in dataset.players}
    returned_ids = [item.player_id for item in validated.players]
    if len(returned_ids) != len(set(returned_ids)):
        raise ValueError("Ranking response contains duplicate players")

    previous_key = None
    for expected_rank, item in enumerate(validated.players, start=1):
        source = players_by_id.get(item.player_id)
        if source is None:
            raise ValueError(f"Ranking response contains unknown player_id {item.player_id}")
        if (
            item.rank != expected_rank
            or source.position != position
            or source.status != "a"
            or (source.chance_of_playing_next_round is not None
                and source.chance_of_playing_next_round <= 0)
            or (max_price is not None and source.price > max_price)
            or item.position != source.position
            or item.price != source.price
            or not math.isfinite(item.price)
            or not math.isfinite(item.projected_points)
        ):
            raise ValueError("Ranking response violates rank, eligibility, or filter rules")

        key = (-item.projected_points, item.player_id)
        if previous_key is not None and key < previous_key:
            raise ValueError("Ranking response is not in deterministic ranking order")
        previous_key = key
    return validated


def guarded_rank_players(
    dataset: CanonicalDataset,
    projection_provider: ProjectionProvider,
    *,
    action: str,
    position: str,
    max_price: float | None = None,
    top_n: int = 5,
    ranker: Callable[..., RankingResult] = rank_players,
) -> GuardrailResult:
    """Validate a ranking request, call the ranking tool, and validate its output."""
    if action != SUPPORTED_ACTION:
        return _failure(
            GuardrailErrorCode.OUT_OF_SCOPE,
            "Only the rank_players action is supported.",
            {"action": action},
        )

    if not isinstance(position, str) or position not in SUPPORTED_POSITIONS:
        return _failure(
            GuardrailErrorCode.INVALID_POSITION,
            "Position must be DEF, MID, or FWD.",
            {"position": position},
        )

    if max_price is not None:
        if (
            isinstance(max_price, bool)
            or not isinstance(max_price, (int, float))
            or not math.isfinite(max_price)
            or max_price < 0
        ):
            return _failure(
                GuardrailErrorCode.INVALID_PRICE,
                "max_price must be a finite, non-negative number or null.",
                {"max_price": repr(max_price)},
            )
        max_price = float(max_price)

    if type(top_n) is not int or not 1 <= top_n <= MAX_TOP_N:
        return _failure(
            GuardrailErrorCode.INVALID_TOP_N,
            f"top_n must be an integer from 1 to {MAX_TOP_N}.",
            {"top_n": repr(top_n), "maximum": MAX_TOP_N},
        )

    player_ids = [player.player_id for player in dataset.players]
    if len(player_ids) != len(set(player_ids)):
        return _failure(
            GuardrailErrorCode.INVALID_DATA,
            "Canonical dataset contains duplicate player IDs.",
        )

    try:
        raw_result = ranker(
            dataset,
            projection_provider,
            position=position,
            max_price=max_price,
            top_n=top_n,
        )
    except ValueError as exc:
        return _failure(
            GuardrailErrorCode.INVALID_DATA,
            "Canonical data or projection records are inconsistent.",
            {"reason": str(exc)},
        )
    except ValidationError as exc:
        return _failure(
            GuardrailErrorCode.INVALID_RANKING_OUTPUT,
            "Ranking output failed schema validation.",
            {"reason": str(exc)},
        )

    try:
        validated = _validate_output(raw_result, dataset, position, max_price, top_n)
    except (ValidationError, ValueError, TypeError) as exc:
        return _failure(
            GuardrailErrorCode.INVALID_RANKING_OUTPUT,
            "Ranking output failed validation.",
            {"reason": str(exc)},
        )

    if not validated.players:
        return _failure(
            GuardrailErrorCode.NO_RESULTS,
            "No eligible players match the requested constraints.",
        )
    return GuardrailResult(result=validated)