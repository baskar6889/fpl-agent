import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from fpl_agent.data.schemas import CanonicalDataset
from fpl_agent.data.transform import transform_bootstrap
from fpl_agent.data.pipeline import run_pipeline

FIXTURE = Path(__file__).parents[1] / "fixtures" / "bootstrap_sample.json"


@pytest.fixture
def bootstrap_payload():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_transform_joins_team_maps_position_and_converts_price(bootstrap_payload):
    result = transform_bootstrap(bootstrap_payload)

    assert result["players"][0] == {
        "player_id": 101,
        "player_name": "Keeper",
        "team_id": 1,
        "team_name": "Team One",
        "position": "GK",
        "price": 4.5,
        "total_points": 88,
        "minutes": 1800,
        "status": "a",
        "chance_of_playing_next_round": 100,
    }
    assert result["players"][1]["position"] == "MID"
    assert result["players"][1]["price"] == 7.3
    assert result["players"][1]["chance_of_playing_next_round"] is None
    assert result["gameweeks_elapsed"] == 3


def test_schema_rejects_unknown_fields(bootstrap_payload):
    dataset = transform_bootstrap(bootstrap_payload)
    dataset["players"][0]["unexpected"] = "value"

    with pytest.raises(ValidationError):
        CanonicalDataset.model_validate(dataset)


def test_pipeline_writes_raw_and_validated_curated_json(tmp_path, bootstrap_payload):
    raw_dir = tmp_path / "raw"
    curated_dir = tmp_path / "curated"
    output = run_pipeline(
        raw_dir=raw_dir,
        curated_dir=curated_dir,
        fetcher=lambda: bootstrap_payload,
    )

    raw_files = list(raw_dir.glob("bootstrap-static-*.json"))
    assert len(raw_files) == 1
    assert json.loads(raw_files[0].read_text(encoding="utf-8")) == bootstrap_payload
    assert output.parent == curated_dir
    result = CanonicalDataset.model_validate_json(output.read_text(encoding="utf-8"))
    assert len(result.players) == 2


def test_finished_and_current_event_is_counted_once(bootstrap_payload):
    bootstrap_payload["events"] = [
        {"id": 1, "finished": True, "is_current": True, "is_next": False},
        {"id": 2, "finished": False, "is_current": True, "is_next": False},
        {"id": 3, "finished": False, "is_current": False, "is_next": True},
    ]

    result = transform_bootstrap(bootstrap_payload)

    assert result["gameweeks_elapsed"] == 2
