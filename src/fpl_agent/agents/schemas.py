"""Structured LLM request and agent turn schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, StrictFloat, StrictInt

from fpl_agent.guardrails.schemas import GuardrailResult


class AgentRankingRequest(BaseModel):
    """Strict request shape; guardrails validate the semantic values."""

    model_config = ConfigDict(extra="forbid", strict=True)

    action: str
    position: str
    max_price: StrictFloat | StrictInt | None = None
    top_n: StrictInt | StrictFloat = 5


class AgentTurn(BaseModel):
    """Traceable agent response, retaining input and validated outcome."""

    model_config = ConfigDict(extra="forbid")

    original_utterance: str
    request: AgentRankingRequest
    guardrail_result: GuardrailResult
    response: str