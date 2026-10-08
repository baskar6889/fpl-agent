"""HTTP request and response schemas that are not domain contracts."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HealthResponse(BaseModel):
    status: str = "ok"


class DataStatusResponse(BaseModel):
    available: bool
    curated_filename: str | None = None
    player_count: int | None = None
    gameweeks_elapsed: int | None = None


class RefreshResponse(BaseModel):
    status: str = "ok"
    curated_filename: str
    player_count: int
    gameweeks_elapsed: int


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1)

    @field_validator("message")
    @classmethod
    def require_non_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must contain non-whitespace characters")
        return value


class APIError(BaseModel):
    code: str
    message: str

