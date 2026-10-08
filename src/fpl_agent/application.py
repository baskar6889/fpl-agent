"""Small shared helpers for constructing the FPL application backend."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from fpl_agent.agents import OpenAIResponsesProvider, RankingAgent
from fpl_agent.data.pipeline import run_pipeline
from fpl_agent.data.schemas import CanonicalDataset
from fpl_agent.projections.baseline import BaselineV1


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
CURATED_DIR = ROOT / "data" / "curated"


@dataclass
class ApplicationBackend:
    dataset: CanonicalDataset
    dataset_path: Path
    projection_provider: BaselineV1
    provider: OpenAIResponsesProvider | None
    agent: RankingAgent | None


def newest_curated_file(curated_dir: str | Path = CURATED_DIR) -> Path | None:
    """Return the newest curated players file, matching the Streamlit behavior."""
    candidates = list(Path(curated_dir).glob("players-*.json"))
    return max(candidates, key=lambda path: path.name) if candidates else None


def build_backend(
    data_path: str | Path | None = None,
    *,
    curated_dir: str | Path = CURATED_DIR,
    require_agent: bool = False,
) -> ApplicationBackend:
    """Load canonical data and construct the shared projection and agent backend."""
    selected_path = Path(data_path) if data_path is not None else newest_curated_file(curated_dir)
    if selected_path is None:
        raise FileNotFoundError("No curated players-*.json dataset was found.")

    dataset = CanonicalDataset.model_validate_json(
        selected_path.read_text(encoding="utf-8")
    )
    projection_provider = BaselineV1()
    model = os.environ.get("OPENAI_MODEL", "").strip()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    provider = None
    agent = None
    if model and api_key:
        try:
            provider = OpenAIResponsesProvider(model=model)
        except Exception:
            if require_agent:
                raise
        else:
            agent = RankingAgent(dataset, projection_provider, provider)
    elif require_agent:
        if not model:
            raise ValueError("Set OPENAI_MODEL to use the OpenAI agent provider.")
        raise ValueError("Set OPENAI_API_KEY before using the OpenAI provider.")

    return ApplicationBackend(
        dataset=dataset,
        dataset_path=selected_path,
        projection_provider=projection_provider,
        provider=provider,
        agent=agent,
    )


def refresh_backend(
    *,
    raw_dir: str | Path = RAW_DIR,
    curated_dir: str | Path = CURATED_DIR,
    require_agent: bool = False,
) -> ApplicationBackend:
    """Run the existing pipeline, then validate and build from its new output."""
    output_path = run_pipeline(raw_dir=raw_dir, curated_dir=curated_dir)
    return build_backend(output_path, curated_dir=curated_dir, require_agent=require_agent)

