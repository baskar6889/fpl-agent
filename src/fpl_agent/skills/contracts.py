"""Descriptive contracts for supported agent skills."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class SkillContract:
    """Describe a skill's capability boundary without executing it."""

    name: str
    purpose: str
    invocation_criteria: str
    logical_inputs: tuple[str, ...]
    input_constraints: tuple[str, ...]
    output_type: str
    supported_errors: tuple[str, ...]
    unsupported_capabilities: tuple[str, ...]


RANK_PLAYERS_CONTRACT: Final = SkillContract(
    name="rank_players",
    purpose="Rank eligible players at a supported position by supplied projected points.",
    invocation_criteria=(
        "Invoke when the user asks for a ranked list of defenders, midfielders, or "
        "forwards, optionally with a maximum price and requested result count."
    ),
    logical_inputs=("position", "max_price (optional)", "top_n (optional, default 5)"),
    input_constraints=(
        "position must be DEF, MID, or FWD",
        "max_price, when supplied, must be finite and non-negative",
        "top_n defaults to 5 and must be an integer from 1 through 20",
    ),
    output_type="RankingResult",
    supported_errors=(
        "OUT_OF_SCOPE",
        "INVALID_POSITION",
        "INVALID_PRICE",
        "INVALID_TOP_N",
        "INVALID_DATA",
        "NO_RESULTS",
        "INVALID_RANKING_OUTPUT",
    ),
    unsupported_capabilities=(
        "Selecting a best XI or building a squad",
        "Recommending transfers",
        "Recommending a captain",
        "Ranking goalkeepers",
    ),
)


SKILL_CONTRACTS: Final[dict[str, SkillContract]] = {
    RANK_PLAYERS_CONTRACT.name: RANK_PLAYERS_CONTRACT,
}
