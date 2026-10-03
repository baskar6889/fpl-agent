import csv
from pathlib import Path

import pytest

from fpl_agent.evaluation.golden_cases import (
    CSV_COLUMNS,
    Decision,
    load_golden_cases,
)

GOLDEN_CSV = Path(__file__).parents[2] / "golden_set.csv"


def test_golden_csv_loads_and_preserves_agreed_cases():
    cases = load_golden_cases(GOLDEN_CSV)

    assert len(cases) == 16
    assert [case.case_id for case in cases] == [f"G{i:03}" for i in range(1, 17)]
    assert cases[11].request == "Top midfielders under -£5m"
    assert cases[12].request == "Give me the top 100 midfielders"
    assert cases[11].expected_parameters == {
        "position": "MID",
        "max_price": -5.0,
        "top_n": 5,
    }
    assert cases[12].expected_parameters == {"position": "MID", "top_n": 100}


def test_decision_types_and_clarification_are_separate_from_skill_invocation():
    cases = {case.case_id: case for case in load_golden_cases(GOLDEN_CSV)}

    assert cases["G001"].decision is Decision.INVOKE_SKILL
    assert cases["G001"].expected_skill == "rank_players"
    assert cases["G008"].decision is Decision.REJECT
    assert cases["G008"].expected_outcome == "OUT_OF_SCOPE"
    for case_id in ("G015", "G016"):
        case = cases[case_id]
        assert case.decision is Decision.CLARIFY
        assert case.expected_outcome == "CLARIFICATION_REQUIRED"
        assert case.expected_skill is None


def test_loader_rejects_invalid_decision(tmp_path):
    path = tmp_path / "invalid.csv"
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "ID": "X001",
                "Category": "Valid",
                "Request": "Top midfielders",
                "Decision": "RUN_TOOL",
                "ExpectedSkill": "rank_players",
                "ExpectedPosition": "MID",
                "ExpectedMaxPrice": "",
                "ExpectedTopN": "5",
                "ExpectedOutcome": "RANKING_SUCCESS",
            }
        )

    with pytest.raises(ValueError, match="invalid decision"):
        load_golden_cases(path)


def test_loader_rejects_schema_mismatch(tmp_path):
    path = tmp_path / "bad-columns.csv"
    path.write_text("ID,Request\nX001,hello\n", encoding="utf-8")

    with pytest.raises(ValueError, match="columns must be exactly"):
        load_golden_cases(path)
