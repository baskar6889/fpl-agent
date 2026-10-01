"""Transform FPL bootstrap payloads to canonical player records."""

from __future__ import annotations

from typing import Any

POSITION_BY_ID = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def transform_bootstrap(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Join elements to teams and emit canonical player fields."""
    try:
        elements = payload["elements"]
        teams = payload["teams"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Bootstrap payload must contain elements and teams") from exc
    if not isinstance(elements, list) or not isinstance(teams, list):
        raise ValueError("Bootstrap elements and teams must be lists")

    team_names: dict[int, str] = {}
    for team in teams:
        team_id = team.get("id")
        name = team.get("name")
        if team_id is None or not isinstance(name, str):
            raise ValueError("Each team must contain an id and name")
        team_names[team_id] = name

    players = []
    for element in elements:
        try:
            team_id = element["team"]
            position_id = element["element_type"]
            players.append(
                {
                    "player_id": element["id"],
                    "player_name": element["web_name"],
                    "team_id": team_id,
                    "team_name": team_names[team_id],
                    "position": POSITION_BY_ID[position_id],
                    "price": int(element["now_cost"]) / 10,
                    "total_points": int(element["total_points"]),
                    "minutes": int(element["minutes"]),
                    "status": element["status"],
                    "chance_of_playing_next_round": element[
                        "chance_of_playing_next_round"
                    ],
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid FPL player record: {element!r}") from exc
    return {"players": players}
