"""Stable interface consumed by downstream ranking components."""

from __future__ import annotations

from typing import Protocol

from fpl_agent.data.schemas import CanonicalDataset

from .schemas import PlayerProjection


class ProjectionProvider(Protocol):
    def project(self, dataset: CanonicalDataset) -> list[PlayerProjection]:
        """Return one projection record for each player in the dataset."""
        ...