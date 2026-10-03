"""Load and validate the descriptive MVP1 golden-case CSV."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Decision(str, Enum):
    INVOKE_SKILL = "INVOKE_SKILL"
    CLARIFY = "CLARIFY"
    REJECT = "REJECT"


@dataclass(frozen=True)
class GoldenCase:
    case_id: str
    category: str
    request: str
    decision: Decision
    expected_skill: str | None
    expected_parameters: dict[str, str | int | float]
    expected_outcome: str


CSV_COLUMNS = (
    "ID",
    "Category",
    "Request",
    "Decision",
    "ExpectedSkill",
    "ExpectedPosition",
    "ExpectedMaxPrice",
    "ExpectedTopN",
    "ExpectedOutcome",
)


def _parse_optional_number(value: str, *, integer: bool, field: str) -> int | float | None:
    if value == "":
        return None
    try:
        parsed = int(value) if integer else float(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be numeric or blank") from exc
    if integer and str(parsed) != value:
        raise ValueError(f"{field} must be an integer or blank")
    return parsed


def _case_from_row(row: dict[str, str], row_number: int) -> GoldenCase:
    case_id = row["ID"].strip()
    category = row["Category"].strip()
    request = row["Request"]
    outcome = row["ExpectedOutcome"].strip()
    try:
        decision = Decision(row["Decision"].strip())
    except ValueError as exc:
        raise ValueError(f"row {row_number}: invalid decision {row['Decision']!r}") from exc

    if not case_id or not category or not request.strip() or not outcome:
        raise ValueError(f"row {row_number}: ID, Category, Request, and ExpectedOutcome are required")

    skill = row["ExpectedSkill"].strip() or None
    parameters: dict[str, str | int | float] = {}
    position = row["ExpectedPosition"].strip()
    if position:
        parameters["position"] = position
    max_price = _parse_optional_number(
        row["ExpectedMaxPrice"].strip(), integer=False, field="ExpectedMaxPrice"
    )
    if max_price is not None:
        parameters["max_price"] = max_price
    top_n = _parse_optional_number(
        row["ExpectedTopN"].strip(), integer=True, field="ExpectedTopN"
    )
    if top_n is not None:
        parameters["top_n"] = top_n

    if decision is Decision.INVOKE_SKILL and skill is None:
        raise ValueError(f"row {row_number}: INVOKE_SKILL requires ExpectedSkill")
    if decision is Decision.CLARIFY and outcome != "CLARIFICATION_REQUIRED":
        raise ValueError(f"row {row_number}: CLARIFY requires CLARIFICATION_REQUIRED outcome")
    if decision is Decision.REJECT and skill is not None:
        raise ValueError(f"row {row_number}: REJECT must not specify ExpectedSkill")

    return GoldenCase(
        case_id=case_id,
        category=category,
        request=request,
        decision=decision,
        expected_skill=skill,
        expected_parameters=parameters,
        expected_outcome=outcome,
    )


def load_golden_cases(path: str | Path | None = None) -> tuple[GoldenCase, ...]:
    """Load and schema-check cases; defaults to the repository-root CSV."""
    csv_path = Path(path) if path is not None else Path(__file__).resolve().parents[3] / "golden_set.csv"
    with csv_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise ValueError(f"golden CSV columns must be exactly {CSV_COLUMNS!r}")
        cases = tuple(_case_from_row(row, row_number) for row_number, row in enumerate(reader, start=2))

    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("golden case IDs must be unique")
    return cases
