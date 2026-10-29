"""Eval-only gold mapping provider for pipeline run A."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datamate.config import PROJECT_ROOT
from datamate.io.profile import BusinessProfile
from datamate.io.raw import BusinessRaw
from datamate.mapping import Mapping, load_mapping_yaml
from datamate.systems.pipeline import PipelineSystem

GOLD_DIR = PROJECT_ROOT / "evals" / "gold_mappings"


class GoldMappingProvider:
    """Load `evals/gold_mappings/<business_id>.yaml`. Lives only under evals/."""

    def __init__(self, gold_dir: Path | None = None) -> None:
        self.gold_dir = gold_dir if gold_dir is not None else GOLD_DIR

    def __call__(
        self,
        split: str,
        business_id: str,
        raw: BusinessRaw,
        profile: BusinessProfile,
        client: Any,
    ) -> Mapping | None:
        del split, raw, profile, client
        path = self.gold_dir / f"{business_id}.yaml"
        if not path.is_file():
            return None
        return load_mapping_yaml(path)


def make_pipeline_gold() -> PipelineSystem:
    return PipelineSystem(name="pipeline_gold", mapping_provider=GoldMappingProvider())


def make_pipeline_llm() -> PipelineSystem:
    return PipelineSystem(name="pipeline_llm")
