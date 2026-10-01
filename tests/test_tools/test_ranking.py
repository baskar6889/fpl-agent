import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from fpl_agent.data.schemas import CanonicalDataset
from fpl_agent.projections.schemas import PlayerProjection
from fpl_agent.tools.ranking import rank_players
from fpl_agent.tools.schemas import RankingRequest, RankingResult

FIXTURE = Path(__file__).parents[1] / "fixtures" / "canonical_players_sample.json"


class StubProjectionProvider:
    def __init__(self, projections=None):
        if projections is None:
            projections = [
                self.projection(101, 4.0),
                self.projection(202, 4.0),
                self.projection(303, 10.0),
            ]
        self.projections = projections

    @staticmethod
    def projection(player_id, points):
        return PlayerProjection(
            player_id=player_id,
            projected_points=points,
            projection_version="test_v1",
            projection_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            projection_source="frozen_test_fixture",
        )

    def project(self, dataset):
        return self.projections


@pytest.fixture
def dataset():
    return CanonicalDataset.model_validate_json(FIXTURE.read_text(encoding="utf-8"))


def test_rank_players_filters_position_price_and_ineligible_chance(dataset):
    result = rank_players(
        dataset,
        StubProjectionProvider(),
        position="MID",
        max_price=7.5,
        top_n=5,
    )

    assert [player.player_id for player in result.players] == [202]
    assert result.position == "MID"
    assert result.max_price == 7.5
    assert result.top_n == 5


def test_sort_is_descending_and_ties_break_by_player_id(dataset):
    players = [player.model_copy() for player in dataset.players]
    players[1] = players[1].model_copy(update={"position": "GK"})
    players[2] = players[2].model_copy(
        update={"position": "GK", "chance_of_playing_next_round": None}
    )
    tied_dataset = CanonicalDataset(players=players)
    provider = StubProjectionProvider(
        [
            StubProjectionProvider.projection(202, 4.0),
            StubProjectionProvider.projection(101, 4.0),
            StubProjectionProvider.projection(303, 9.0),
        ]
    )
    result = rank_players(tied_dataset, provider, position="GK", top_n=5)

    assert [player.player_id for player in result.players] == [303, 101, 202]
    assert [player.rank for player in result.players] == [1, 2, 3]


def test_zero_chance_player_is_ineligible(dataset):
    result = rank_players(dataset, StubProjectionProvider(), position="FWD")

    assert result.players == []


def test_top_n_truncates_sorted_results(dataset):
    # Promote player 303 to MID and eligible status for this focused ordering case.
    players = [player.model_copy() for player in dataset.players]
    players[2] = players[2].model_copy(
        update={"position": "MID", "chance_of_playing_next_round": None}
    )
    expanded = CanonicalDataset(players=players)
    result = rank_players(expanded, StubProjectionProvider(), position="MID", top_n=1)

    assert [player.player_id for player in result.players] == [303]
    assert result.players[0].projected_points == 10.0


def test_non_available_status_is_excluded(dataset):
    players = [player.model_copy() for player in dataset.players]
    players[1] = players[1].model_copy(update={"status": "d"})
    modified = CanonicalDataset(players=players)

    result = rank_players(modified, StubProjectionProvider(), position="MID")

    assert result.players == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"position": "ALL"},
        {"position": "MID", "max_price": -1},
        {"position": "MID", "top_n": 0},
    ],
)
def test_invalid_requests_are_rejected(kwargs):
    with pytest.raises(ValidationError):
        RankingRequest(**kwargs)


def test_missing_duplicate_and_unexpected_projection_ids_are_rejected(dataset):
    missing = StubProjectionProvider([StubProjectionProvider.projection(101, 2.0)])
    with pytest.raises(ValueError, match="missing"):
        rank_players(dataset, missing, position="GK")

    duplicate = StubProjectionProvider(
        [
            StubProjectionProvider.projection(101, 2.0),
            StubProjectionProvider.projection(101, 3.0),
            StubProjectionProvider.projection(202, 1.0),
            StubProjectionProvider.projection(303, 0.0),
        ]
    )
    with pytest.raises(ValueError, match="Duplicate projection"):
        rank_players(dataset, duplicate, position="GK")

    unexpected = StubProjectionProvider(
        [
            StubProjectionProvider.projection(101, 2.0),
            StubProjectionProvider.projection(202, 1.0),
            StubProjectionProvider.projection(999, 0.0),
        ]
    )
    with pytest.raises(ValueError, match="unexpected"):
        rank_players(dataset, unexpected, position="GK")


def test_structured_response_validates_and_contains_projection_metadata(dataset):
    result = rank_players(dataset, StubProjectionProvider(), position="MID")

    validated = RankingResult.model_validate_json(result.model_dump_json())
    item = validated.players[0]
    assert item.projection_version == "test_v1"
    assert item.projection_source == "frozen_test_fixture"
    assert item.projection_timestamp == datetime(2026, 1, 1, tzinfo=timezone.utc)

def test_negative_projections_are_valid_and_sorted_descending(dataset):
    players = [player.model_copy() for player in dataset.players]
    players[1] = players[1].model_copy(update={"position": "MID"})
    players[2] = players[2].model_copy(
        update={"position": "MID", "chance_of_playing_next_round": None}
    )
    dataset_with_two_mids = CanonicalDataset(players=players)

    class NegativeProjectionProvider:
        def project(self, data):
            return [
                StubProjectionProvider.projection(101, 1.0),
                StubProjectionProvider.projection(202, -0.67),
                StubProjectionProvider.projection(303, -0.4),
            ]

    result = rank_players(
        dataset_with_two_mids, NegativeProjectionProvider(), position="MID"
    )

    assert [(player.player_id, player.projected_points) for player in result.players] == [
        (303, -0.4),
        (202, -0.67),
    ]
