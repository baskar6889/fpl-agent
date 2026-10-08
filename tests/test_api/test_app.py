from pathlib import Path

from fastapi.testclient import TestClient

from fpl_agent.agents import RankingAgent
from fpl_agent.agents.schemas import AgentRankingRequest
from fpl_agent.api import app as api_module
from fpl_agent.api.app import create_app
from fpl_agent.application import build_backend
from fpl_agent.data.schemas import CanonicalDataset
from fpl_agent.projections.baseline import BaselineV1

FIXTURE = Path(__file__).parents[1] / "fixtures" / "canonical_players_sample.json"


def _write_dataset(directory: Path, dataset: CanonicalDataset, name="players-test.json"):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(dataset.model_dump_json(), encoding="utf-8")
    return path


def test_health_and_status_with_dataset(tmp_path, monkeypatch):
    curated = tmp_path / "curated"
    path = _write_dataset(curated, CanonicalDataset.model_validate_json(FIXTURE.read_text(encoding="utf-8")))
    monkeypatch.setattr(api_module, "CURATED_DIR", curated)
    app = create_app()
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        response = client.get("/data/status")
    assert response.json() == {"available": True, "curated_filename": path.name, "player_count": 3, "gameweeks_elapsed": 30}


def test_status_without_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(api_module, "CURATED_DIR", tmp_path / "empty")
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/data/status")
    assert response.status_code == 200
    assert response.json() == {"available": False, "curated_filename": None, "player_count": None, "gameweeks_elapsed": None}


def test_rank_valid_and_invalid_request(tmp_path, monkeypatch):
    curated = tmp_path / "curated"
    dataset = CanonicalDataset.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    _write_dataset(curated, dataset)
    monkeypatch.setattr(api_module, "CURATED_DIR", curated)
    app = create_app()
    with TestClient(app) as client:
        valid = client.post("/rank", json={"position": "MID", "max_price": 7.5, "top_n": 5})
        invalid = client.post("/rank", json={"position": "ALL", "max_price": 7.5, "top_n": 5})
    assert valid.status_code == 200
    assert valid.json()["result"]["players"][0]["player_id"] == 202
    assert invalid.status_code == 422


class FakeLLM:
    def interpret_request(self, utterance, prompt):
        return AgentRankingRequest(action="rank_players", position="MID", max_price=7.5, top_n=5)

    def generate_response(self, outcome):
        return "Mocked ranking response."


def test_chat_with_fake_llm(tmp_path, monkeypatch):
    curated = tmp_path / "curated"
    dataset = CanonicalDataset.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    path = _write_dataset(curated, dataset)
    monkeypatch.setattr(api_module, "CURATED_DIR", curated)
    app = create_app()
    with TestClient(app) as client:
        backend = build_backend(path)
        backend.agent = RankingAgent(dataset, BaselineV1(), FakeLLM())
        app.state.backend = backend
        response = client.post("/chat", json={"message": "Top midfielders under £7.5m"})
    assert response.status_code == 200
    assert response.json()["response"] == "Mocked ranking response."
    assert response.json()["guardrail_result"]["result"]["players"][0]["player_id"] == 202


def test_refresh_replaces_backend_and_subsequent_rank_uses_new_dataset(tmp_path, monkeypatch):
    curated, raw = tmp_path / "curated", tmp_path / "raw"
    initial = CanonicalDataset.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    _write_dataset(curated, initial)
    monkeypatch.setattr(api_module, "CURATED_DIR", curated)
    monkeypatch.setattr(api_module, "RAW_DIR", raw)
    refreshed = initial.model_copy(deep=True)
    refreshed.players[1] = refreshed.players[1].model_copy(update={"position": "FWD"})
    refreshed_path = curated / "players-refreshed.json"

    def fake_pipeline(*, raw_dir, curated_dir):
        assert Path(raw_dir) == raw
        assert Path(curated_dir) == curated
        return _write_dataset(curated, refreshed, refreshed_path.name)

    monkeypatch.setattr("fpl_agent.application.run_pipeline", fake_pipeline)
    app = create_app()
    with TestClient(app) as client:
        refresh_response = client.post("/data/refresh")
        rank_response = client.post("/rank", json={"position": "MID", "max_price": 7.5, "top_n": 5})
        status_response = client.get("/data/status")
    assert refresh_response.status_code == 200
    assert refresh_response.json()["curated_filename"] == refreshed_path.name
    assert status_response.json()["player_count"] == len(refreshed.players)
    assert rank_response.status_code == 200
    assert rank_response.json()["error"]["code"] == "NO_RESULTS"


def test_refresh_failure_keeps_working_backend(tmp_path, monkeypatch):
    curated = tmp_path / "curated"
    _write_dataset(curated, CanonicalDataset.model_validate_json(FIXTURE.read_text(encoding="utf-8")))
    monkeypatch.setattr(api_module, "CURATED_DIR", curated)
    monkeypatch.setattr(api_module, "RAW_DIR", tmp_path / "raw")

    def failed_pipeline(**kwargs):
        raise RuntimeError("upstream unavailable")

    monkeypatch.setattr("fpl_agent.application.run_pipeline", failed_pipeline)
    app = create_app()
    with TestClient(app) as client:
        original_backend = app.state.backend
        response = client.post("/data/refresh")
    assert response.status_code == 502
    assert app.state.backend is original_backend
