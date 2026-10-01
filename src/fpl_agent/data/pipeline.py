"""Coordinate fetching, raw storage, transformation, validation, and output."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .client import fetch_bootstrap_data
from .schemas import CanonicalDataset
from .transform import transform_bootstrap


def run_pipeline(
    raw_dir: str | Path = "data/raw",
    curated_dir: str | Path = "data/curated",
    fetcher: Callable[[], dict] = fetch_bootstrap_data,
) -> Path:
    """Fetch and persist bootstrap JSON and the validated canonical dataset."""
    payload = fetcher()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = Path(raw_dir) / f"bootstrap-static-{stamp}.json"
    curated_path = Path(curated_dir) / f"players-{stamp}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    curated_path.parent.mkdir(parents=True, exist_ok=True)

    raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    canonical = CanonicalDataset.model_validate(transform_bootstrap(payload))
    curated_path.write_text(
        canonical.model_dump_json(indent=2), encoding="utf-8"
    )
    return curated_path
