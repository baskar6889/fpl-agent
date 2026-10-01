"""Provider-neutral orchestration for LLM-assisted FPL player ranking."""

from __future__ import annotations

from typing import Protocol

from pydantic import ValidationError

from fpl_agent.data.schemas import CanonicalDataset
from fpl_agent.guardrails.ranking import guarded_rank_players
from fpl_agent.guardrails.schemas import GuardrailError, GuardrailResult
from fpl_agent.projections.interface import ProjectionProvider
from fpl_agent.tools.schemas import RankingResult

from .prompts import REQUEST_INTERPRETATION_PROMPT
from .schemas import AgentRankingRequest, AgentTurn


class RankingLLM(Protocol):
    """Minimal LLM boundary; it receives no dataset or data access objects."""

    def interpret_request(
        self, utterance: str, prompt: str
    ) -> AgentRankingRequest | dict:
        """Convert an utterance to the structured ranking request shape."""
        ...

    def generate_response(self, outcome: RankingResult | GuardrailError) -> str:
        """Render only a validated ranking result or guardrail error."""
        ...


class AgentRequestError(ValueError):
    """Raised when the LLM does not return a valid structured request."""


class RankingAgent:
    def __init__(
        self,
        dataset: CanonicalDataset,
        projection_provider: ProjectionProvider,
        llm: RankingLLM,
    ) -> None:
        self._dataset = dataset
        self._projection_provider = projection_provider
        self._llm = llm

    def respond(self, utterance: str) -> AgentTurn:
        """Interpret, guard, rank, and render one user utterance."""
        if not isinstance(utterance, str) or not utterance.strip():
            raise ValueError("utterance must be a non-empty string")

        try:
            request = AgentRankingRequest.model_validate(
                self._llm.interpret_request(utterance, REQUEST_INTERPRETATION_PROMPT)
            )
        except ValidationError as exc:
            raise AgentRequestError("LLM returned an invalid ranking request") from exc

        guardrail_result: GuardrailResult = guarded_rank_players(
            self._dataset,
            self._projection_provider,
            action=request.action,
            position=request.position,
            max_price=request.max_price,
            top_n=request.top_n,
        )
        outcome: RankingResult | GuardrailError
        if guardrail_result.result is not None:
            outcome = guardrail_result.result
        else:
            # GuardrailResult validates exactly one outcome is present.
            assert guardrail_result.error is not None
            outcome = guardrail_result.error

        response = self._llm.generate_response(outcome)
        if not isinstance(response, str):
            raise TypeError("LLM response generation must return a string")

        return AgentTurn(
            original_utterance=utterance,
            request=request,
            guardrail_result=guardrail_result,
            response=response,
        )