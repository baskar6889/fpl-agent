from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from fpl_agent.data.schemas import CanonicalDataset
from fpl_agent.guardrails.ranking import MAX_TOP_N, guarded_rank_players
from fpl_agent.guardrails.schemas import (
    GuardrailErrorCode,
    GuardrailResult,
)
from fpl_agent.projections.schemas import PlayerProjection
from fpl_agent.tools.ranking import rank_players

FIXTURE = Path(__file__).parents[1] / "fixtures" / "canonical_players_sample.json"


class StubProjectionProvider:
    def project(self, dataset):
        return [
            PlayerProjection(
                player_id=player.player_id,
                projected_points={101: 3.0, 202: 5.0, 303: 7.0}[player.player_id],
                projection_version="stub_v1",
                projection_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                projection_source="guardrail_test",
            )
            for player in dataset.players
        ]


@pytest.fixture
def dataset():
    return CanonicalDataset.model_validate_json(FIXTURE.read_text(encoding="utf-8"))


def call(dataset, **kwargs):
    defaults = {"action": "rank_players", "position": "MID"}
    defaults.update(kwargs)
    return guarded_rank_players(dataset, StubProjectionProvider(), **defaults)


def test_supported_request_returns_valid_ranking(dataset):
    result = call(dataset, position="MID", max_price=7.5, top_n=1)

    assert result.error is None
    assert [player.player_id for player in result.result.players] == [202]
    assert result.result.max_price == 7.5
    assert result.result.top_n == 1


def test_unsupported_action_is_out_of_scope_before_other_validation(dataset):
    result = call(dataset, action="best_xi", position="ALL", max_price=-1, top_n=0)

    assert result.error.code == GuardrailErrorCode.OUT_OF_SCOPE
    assert result.result is None


@pytest.mark.parametrize("position", ["GK", "ALL", "mid", ""])
def test_only_supported_positions_are_accepted(dataset, position):
    result = call(dataset, position=position)

    assert result.error.code == GuardrailErrorCode.INVALID_POSITION


@pytest.mark.parametrize("price", [-0.01, float("nan"), float("inf"), -float("inf"), True])
def test_invalid_prices_are_rejected(dataset, price):
    result = call(dataset, max_price=price)

    assert result.error.code == GuardrailErrorCode.INVALID_PRICE


def test_none_and_finite_nonnegative_price_are_allowed(dataset):
    no_cap = call(dataset, max_price=None)
    with_cap = call(dataset, max_price=7.5)

    assert no_cap.error is None
    assert with_cap.error is None
    assert all(player.price <= 7.5 for player in with_cap.result.players)


@pytest.mark.parametrize("top_n", [0, -1, 21, 1.5, True, "2"])
def test_invalid_or_excessive_top_n_is_rejected(dataset, top_n):
    result = call(dataset, top_n=top_n)

    assert result.error.code == GuardrailErrorCode.INVALID_TOP_N


def test_default_and_maximum_top_n_are_accepted(dataset):
    defaulted = call(dataset)
    maximum = call(dataset, top_n=MAX_TOP_N)

    assert defaulted.error is None
    assert defaulted.result.top_n == 5
    assert maximum.error is None
    assert maximum.result.top_n == MAX_TOP_N


def test_no_eligible_players_returns_no_results(dataset):
    result = call(dataset, position="FWD")

    assert result.error.code == GuardrailErrorCode.NO_RESULTS
    assert result.result is None


def test_non_available_players_are_not_returned(dataset):
    players = [player.model_copy() for player in dataset.players]
    players[1] = players[1].model_copy(update={"status": "d"})
    modified = CanonicalDataset(players=players)

    result = guarded_rank_players(
        modified, StubProjectionProvider(), action="rank_players", position="MID"
    )

    assert result.error.code == GuardrailErrorCode.NO_RESULTS


def test_projection_mismatch_is_invalid_data(dataset):
    class MissingProjectionProvider:
        def project(self, dataset):
            return []

    result = guarded_rank_players(
        dataset, MissingProjectionProvider(), action="rank_players", position="MID"
    )

    assert result.error.code == GuardrailErrorCode.INVALID_DATA


def test_duplicate_canonical_player_ids_are_invalid_data(dataset):
    duplicated = CanonicalDataset(players=[*dataset.players, dataset.players[0]])
    result = call(duplicated)

    assert result.error.code == GuardrailErrorCode.INVALID_DATA


def test_malformed_ranking_output_is_rejected(dataset):
    players = [player.model_copy() for player in dataset.players]
    players[2] = players[2].model_copy(
        update={"position": "MID", "chance_of_playing_next_round": None}
    )
    expanded = CanonicalDataset(players=players)

    def bad_ranker(*args, **kwargs):
        valid = rank_players(*args, **kwargs)
        return valid.model_copy(update={"players": list(reversed(valid.players))})

    result = guarded_rank_players(
        expanded,
        StubProjectionProvider(),
        action="rank_players",
        position="MID",
        ranker=bad_ranker,
    )

    assert result.error.code == GuardrailErrorCode.INVALID_RANKING_OUTPUT


def test_guardrail_result_requires_exactly_one_outcome():
    with pytest.raises(ValidationError):
        GuardrailResult()
    with pytest.raises(ValidationError):
        GuardrailResult(result=None, error=None)