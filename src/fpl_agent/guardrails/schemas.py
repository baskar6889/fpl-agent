"""Structured guardrail outcomes and stable error codes."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from fpl_agent.tools.schemas import RankingResult


class GuardrailErrorCode(str, Enum):
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    INVALID_POSITION = "INVALID_POSITION"
    INVALID_PRICE = "INVALID_PRICE"
    INVALID_TOP_N = "INVALID_TOP_N"
    INVALID_DATA = "INVALID_DATA"
    NO_RESULTS = "NO_RESULTS"
    INVALID_RANKING_OUTPUT = "INVALID_RANKING_OUTPUT"


class GuardrailError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: GuardrailErrorCode
    message: str
    details: dict[str, Any] | None = None


class GuardrailResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: RankingResult | None = None
    error: GuardrailError | None = None

    @model_validator(mode="after")
    def require_exactly_one_outcome(self) -> "GuardrailResult":
        if (self.result is None) == (self.error is None):
            raise ValueError("Exactly one of result or error must be provided")
        return self