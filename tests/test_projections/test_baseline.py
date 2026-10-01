import json
from datetime import timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from fpl_agent.data.schemas import CanonicalDataset
from fpl_agent.projections.baseline import BaselineV1
from fpl_agent.projections.schemas import PlayerProjection

FIXTURE = Path(__file__).parents[1] / "fixtures" / "canonical_players_sample.json"


@pytest.fixture
def dataset():
    return CanonicalDataset.model_validate_json(FIXTURE.read_text(encoding="utf-8"))


def test_baseline_projects_one_valid_record_per_player(dataset):
    projections = BaselineV1().project(dataset)

    assert [projection.player_id for projection in projections] == [101, 202, 303]
    assert [projection.projected_points for projection in projections] == [1.5, 4.0, 0.0]
    assert all(projection.projection_version == "baseline_v1" for projection in projections)
    assert all(projection.projection_source == "season_scoring_rate" for projection in projections)
    assert len({projection.projection_timestamp for projection in projections}) == 1
    assert projections[0].projection_timestamp.tzinfo == timezone.utc
    assert all(isinstance(item, PlayerProjection) for item in projections)


def test_projection_schema_rejects_missing_or_extra_fields(dataset):
    output = BaselineV1().project(dataset)[0].model_dump()
    del output["projection_source"]
    with pytest.raises(ValidationError):
        PlayerProjection.model_validate(output)

    output = BaselineV1().project(dataset)[0].model_dump()
    output["unrequested_field"] = True
    with pytest.raises(ValidationError):
        PlayerProjection.model_validate(output)


def test_baseline_expected_minutes_is_configurable_and_nonnegative(dataset):
    projection = BaselineV1(expected_minutes=90).project(dataset)[0]
    assert projection.projected_points == 2.25

    with pytest.raises(ValueError, match="expected_minutes"):
        BaselineV1(expected_minutes=-1)

def test_negative_historical_rate_produces_legitimate_negative_projection(dataset):
    player = dataset.players[0].model_copy(
        update={
            "total_points": -1,
            "minutes": 90,
            "chance_of_playing_next_round": 100,
        }
    )
    one_player_dataset = CanonicalDataset(players=[player])

    projection = BaselineV1().project(one_player_dataset)[0]

    assert projection.player_id == player.player_id
    assert projection.projected_points == -0.67
