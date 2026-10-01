"""OpenAI Responses API implementation of the provider-neutral LLM interface."""

from __future__ import annotations

import json
import os
from typing import Any

from fpl_agent.guardrails.schemas import GuardrailError
from fpl_agent.tools.schemas import RankingResult

from .prompts import REQUEST_INTERPRETATION_PROMPT, RESPONSE_GENERATION_PROMPT
from .schemas import AgentRankingRequest


class OpenAIResponsesProvider:
    """Interpret requests and render results through the OpenAI Responses API."""

    def __init__(self, model: str, *, client: Any | None = None) -> None:
        if not model:
            raise ValueError("model must be a non-empty model identifier")
        self._model = model
        if client is not None:
            self._client = client
            return

        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("Set OPENAI_API_KEY before using the OpenAI provider")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install the OpenAI SDK with: pip install openai") from exc
        self._client = OpenAI(api_key=key)

    def interpret_request(self, utterance: str, prompt: str) -> AgentRankingRequest:
        response = self._client.responses.parse(
            model=self._model,
            input=[
                {"role": "system", "content": prompt or REQUEST_INTERPRETATION_PROMPT},
                {"role": "user", "content": utterance},
            ],
            text_format=AgentRankingRequest,
        )
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise ValueError("OpenAI did not return a structured ranking request")
        return AgentRankingRequest.model_validate(parsed)

    def generate_response(self, outcome: RankingResult | GuardrailError) -> str:
        if not isinstance(outcome, (RankingResult, GuardrailError)):
            raise TypeError("outcome must be a validated RankingResult or GuardrailError")
        payload = json.dumps(outcome.model_dump(mode="json"), ensure_ascii=False)
        response = self._client.responses.create(
            model=self._model,
            instructions=RESPONSE_GENERATION_PROMPT,
            input=payload,
        )
        text = getattr(response, "output_text", None)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("OpenAI did not return a response")
        return text.strip()
