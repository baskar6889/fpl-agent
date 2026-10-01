import json
import sys
from types import SimpleNamespace

import pytest

from fpl_agent.agents.openai_provider import OpenAIResponsesProvider
from fpl_agent.agents.prompts import RESPONSE_GENERATION_PROMPT
from fpl_agent.agents.schemas import AgentRankingRequest
from fpl_agent.guardrails.schemas import GuardrailError, GuardrailErrorCode


class MockResponses:
    def __init__(self, parsed=None, output_text="Generated response"):
        self.parsed = parsed
        self.output_text = output_text
        self.parse_calls = []
        self.create_calls = []

    def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        return SimpleNamespace(output_parsed=self.parsed)

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return SimpleNamespace(output_text=self.output_text)


class MockClient:
    def __init__(self, responses):
        self.responses = responses


def test_interpret_uses_responses_structured_output_and_returns_request():
    request = AgentRankingRequest(
        action="rank_players", position="MID", max_price=7.5, top_n=5
    )
    responses = MockResponses(parsed=request)
    provider = OpenAIResponsesProvider("test-model", client=MockClient(responses))

    result = provider.interpret_request("Top midfielders under 7.5m", "Interpret prompt")

    assert result == request
    call = responses.parse_calls[0]
    assert call["model"] == "test-model"
    assert call["text_format"] is AgentRankingRequest
    assert call["input"][1]["content"] == "Top midfielders under 7.5m"
    assert "tools" not in call


def test_interpret_uses_default_prompt_and_rejects_missing_parsed_output():
    responses = MockResponses(parsed=None)
    provider = OpenAIResponsesProvider("test-model", client=MockClient(responses))

    with pytest.raises(ValueError, match="structured ranking request"):
        provider.interpret_request("Top defenders", "")
    assert "structured FPL player-ranking request" in responses.parse_calls[0]["input"][0]["content"]


def test_generate_response_sends_only_validated_outcome():
    error = GuardrailError(
        code=GuardrailErrorCode.NO_RESULTS,
        message="No eligible players match the requested constraints.",
    )
    responses = MockResponses(output_text="No players matched your constraints.")
    provider = OpenAIResponsesProvider("test-model", client=MockClient(responses))

    text = provider.generate_response(error)

    assert text == "No players matched your constraints."
    call = responses.create_calls[0]
    assert call["model"] == "test-model"
    assert call["instructions"] == RESPONSE_GENERATION_PROMPT
    assert json.loads(call["input"]) == error.model_dump(mode="json")
    assert "tools" not in call


def test_generate_response_accepts_only_ranking_result_or_guardrail_error():
    provider = OpenAIResponsesProvider("test-model", client=MockClient(MockResponses()))

    with pytest.raises(TypeError, match="validated RankingResult or GuardrailError"):
        provider.generate_response({"some": "unvalidated data"})


def test_api_key_is_read_from_environment(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, *, api_key):
            captured["api_key"] = api_key

    monkeypatch.setenv("OPENAI_API_KEY", "test-key-from-env")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    provider = OpenAIResponsesProvider("test-model")

    assert captured["api_key"] == "test-key-from-env"
    assert provider._client.__class__ is FakeOpenAI


def test_missing_api_key_fails_before_client_creation(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIResponsesProvider("test-model")
