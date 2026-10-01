"""HTTP client for the official Fantasy Premier League API."""

from __future__ import annotations

import httpx

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"


def fetch_bootstrap_data(timeout: float = 20.0) -> dict:
    """Fetch bootstrap-static data from the official FPL API."""
    response = httpx.get(BOOTSTRAP_URL, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("FPL bootstrap response must be a JSON object")
    return payload
