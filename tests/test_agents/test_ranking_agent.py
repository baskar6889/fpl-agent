from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from fpl_agent.data.schemas import CanonicalDataset
from fpl_agent.guardrails.schemas import GuardrailError, GuardrailErrorCode
from fpl_agent.projections.schemas import PlayerProjection
from fpl_agent.agents.prompts import REQUEST_INTERPRETATION_PROMPT
from fpl_agent.agents.ranking_agent import RankingAgent
from fpl_agent.agents.schemas import AgentRankingRequest

FIXTURE = Path(__file__).parents[1] / "fixtures" / "canonical_players_sample.json"


class FakeLLM:
    def __init__(self, request):
        self.request = request
        self.interpret_calls = []
        self.response_calls = []

    def interpret_request(self, utterance, prompt):
        self.interpret_calls.append((utterance, prompt))
        return self.request

    def generate_response(self, outcome):
        self.response_calls.append(outcome)
        if isinstance(outcome, GuardrailError):
            return f"Request rejected: {outcome.code.value}. {outcome.message}"
        return "Here are the validated ranked players."


class DeterministicProjectionProvider:
    def project(self, dataset):
        return [
            PlayerProjection(
                player_id=player.player_id,
                projected_points={101: 3.0, 202: 5.0, 303: 7.0}[player.player_id],
                projection_version="test_v1",
                projection_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                projection_source="agent_test_fixture",
            )
            for player in dataset.players
        ]


@pytest.fixture
def dataset():
    return CanonicalDataset.model_validate_json(FIXTURE.read_text(encoding="utf-8"))


def make_agent(dataset, request):
    fake_llm = FakeLLM(request)
    agent = RankingAgent(dataset, DeterministicProjectionProvider(), fake_llm)
    return agent, fake_llm


def test_agent_interprets_and_routes_supported_request(dataset):
    utterance = "Top midfielders under 7.5m"
    request = AgentRankingRequest(
        action="rank_players", position="MID", max_price=7.5, top_n=5
    )
    agent, fake_llm = make_agent(dataset, request)

    turn = agent.respond(utterance)

    assert turn.original_utterance == utterance
    assert turn.request == request
    assert turn.guardrail_result.result.players[0].player_id == 202
    assert turn.guardrail_result.result.max_price == 7.5
    assert turn.response == "Here are the validated ranked players."
    assert fake_llm.interpret_calls == [(utterance, REQUEST_INTERPRETATION_PROMPT)]
    assert len(fake_llm.response_calls) == 1
    assert fake_llm.response_calls[0] is turn.guardrail_result.result


def test_default_request_count_is_five(dataset):
    request = AgentRankingRequest(action="rank_players", position="DEF")
    agent, _ = make_agent(dataset, request)

    turn = agent.respond("Top defenders")

    assert turn.request.top_n == 5
    assert turn.guardrail_result.error.code == GuardrailErrorCode.NO_RESULTS


def test_out_of_scope_request_is_guarded_and_error_is_rendered(dataset):
    utterance = "Pick my best XI"
    request = AgentRankingRequest(action="best_xi", position="MID")
    agent, fake_llm = make_agent(dataset, request)

    turn = agent.respond(utterance)

    assert turn.original_utterance == utterance
    assert turn.guardrail_result.error.code == GuardrailErrorCode.OUT_OF_SCOPE
    assert fake_llm.response_calls == [turn.guardrail_result.error]
    assert "OUT_OF_SCOPE" in turn.response


def test_invalid_position_is_rejected_by_guardrails(dataset):
    request = AgentRankingRequest(action="rank_players", position="GK")
    agent, fake_llm = make_agent(dataset, request)

    turn = agent.respond("Top goalkeepers")

    assert turn.guardrail_result.error.code == GuardrailErrorCode.INVALID_POSITION
    assert fake_llm.response_calls == [turn.guardrail_result.error]


def test_response_generator_receives_only_validated_outcome(dataset):
    request = AgentRankingRequest(action="rank_players", position="MID")
    agent, fake_llm = make_agent(dataset, request)

    turn = agent.respond("Top midfielders")

    assert len(fake_llm.response_calls) == 1
    assert fake_llm.response_calls[0] is turn.guardrail_result.result
    assert isinstance(fake_llm.response_calls[0], list | tuple) is False


def test_llm_output_shape_rejects_unexpected_fields():
    with pytest.raises(ValidationError):
        AgentRankingRequest(
            action="rank_players", position="MID", projected_points=99
        )


def test_interpreter_receives_only_utterance_and_prompt(dataset):
    request = AgentRankingRequest(action="rank_players", position="MID")
    agent, fake_llm = make_agent(dataset, request)

    agent.respond("Top midfielders")

    # The LLM interface receives no dataset, provider, or FPL client.
    assert len(fake_llm.interpret_calls[0]) == 2
    assert fake_llm.interpret_calls[0][0] == "Top midfielders"

def test_forward_price_and_count_constraints_are_preserved_when_no_results(dataset):
    utterance = "Top 5 forwards under 8m"
    request = AgentRankingRequest(
        action="rank_players", position="FWD", max_price=8.0, top_n=5
    )
    agent, fake_llm = make_agent(dataset, request)

    turn = agent.respond(utterance)

    assert turn.original_utterance == utterance
    assert turn.request.position == "FWD"
    assert turn.request.max_price == 8.0
    assert turn.request.top_n == 5
    assert turn.guardrail_result.error.code == GuardrailErrorCode.NO_RESULTS
    assert fake_llm.response_calls == [turn.guardrail_result.error]