"""FastAPI presentation adapter for existing FPL Agent capabilities."""

from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from fpl_agent.application import (
    ApplicationBackend,
    CURATED_DIR,
    RAW_DIR,
    build_backend,
    newest_curated_file,
    refresh_backend,
)
from fpl_agent.agents.schemas import AgentTurn
from fpl_agent.api.schemas import (
    APIError,
    ChatRequest,
    DataStatusResponse,
    HealthResponse,
    RefreshResponse,
)
from fpl_agent.guardrails.ranking import guarded_rank_players
from fpl_agent.guardrails.schemas import GuardrailResult
from fpl_agent.tools.schemas import RankingRequest


def _optional_backend() -> ApplicationBackend | None:
    path = newest_curated_file(CURATED_DIR)
    if path is None:
        return None
    try:
        return build_backend(path)
    except (OSError, ValidationError, ValueError, ImportError):
        # An invalid dataset is unavailable; health/status remain useful.
        return None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    app.state.backend = _optional_backend()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="FPL Agent API", lifespan=_lifespan)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/data/status", response_model=DataStatusResponse)
    def data_status() -> DataStatusResponse:
        backend = app.state.backend
        if backend is None:
            return DataStatusResponse(available=False)
        return DataStatusResponse(
            available=True,
            curated_filename=backend.dataset_path.name,
            player_count=len(backend.dataset.players),
            gameweeks_elapsed=backend.dataset.gameweeks_elapsed,
        )

    @app.post("/data/refresh", response_model=RefreshResponse)
    def data_refresh() -> RefreshResponse:
        try:
            updated = refresh_backend(raw_dir=RAW_DIR, curated_dir=CURATED_DIR)
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=APIError(
                    code="DATA_REFRESH_FAILED",
                    message=f"FPL data refresh failed: {exc}",
                ).model_dump(),
            ) from exc
        # Commit the complete, validated backend as one state replacement.
        app.state.backend = updated
        return RefreshResponse(
            curated_filename=updated.dataset_path.name,
            player_count=len(updated.dataset.players),
            gameweeks_elapsed=updated.dataset.gameweeks_elapsed,
        )

    @app.post("/rank", response_model=GuardrailResult)
    def rank(request: RankingRequest) -> GuardrailResult:
        backend = app.state.backend
        if backend is None:
            raise HTTPException(
                status_code=503,
                detail=APIError(
                    code="DATA_UNAVAILABLE",
                    message="No valid curated FPL dataset is available.",
                ).model_dump(),
            )
        return guarded_rank_players(
            backend.dataset,
            backend.projection_provider,
            action="rank_players",
            position=request.position,
            max_price=request.max_price,
            top_n=request.top_n,
        )

    @app.post("/chat", response_model=AgentTurn)
    def chat(request: ChatRequest) -> AgentTurn:
        backend = app.state.backend
        if backend is None:
            raise HTTPException(
                status_code=503,
                detail=APIError(
                    code="DATA_UNAVAILABLE",
                    message="No valid curated FPL dataset is available.",
                ).model_dump(),
            )
        if backend.agent is None:
            raise HTTPException(
                status_code=503,
                detail=APIError(
                    code="OPENAI_NOT_CONFIGURED",
                    message="Set OPENAI_MODEL and OPENAI_API_KEY to use chat.",
                ).model_dump(),
            )
        try:
            return backend.agent.respond(request.message)
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=APIError(code="CHAT_FAILED", message=f"Chat request failed: {exc}").model_dump(),
            ) from exc

    return app


app = create_app()

