from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_enabled_families() -> list[str]:
    return [
        "qty_sold",
        "top_products",
        "unsold_products",
        "products_in_min_orders",
        "price_on_date",
        "price_increase_effect",
        "customers_min_orders",
        "top_customers",
        "orders_min_amount",
        "days_min_revenue",
    ]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    data_root: Path = Path("D:/zaloaic")
    cache_dir: Path = Path("cache")
    runs_dir: Path = Path("runs")
    llm_base_url: str = "http://localhost:8000/v1"
    llm_model: str = "qwen3-4b"
    llm_api_key: str = "EMPTY"
    llm_timeout_s: float = 120.0
    llm_max_context: int = 8192
    llm_max_output_tokens: int = 1024
    llm_prompt_budget_tokens: int = 6000
    llm_cache_enabled: bool = True
    llm_disable_thinking: bool = False
    seed: int = 20261116
    parse_votes: int = 2
    direct_timeout_s: float = 30.0
    direct_repairs: int = 2
    direct_prompt_budget_tokens: int = 6600
    # Extension families that may answer when the main parser is out of scope.
    enabled_families: list[str] = Field(default_factory=_default_enabled_families)
    # Ext verifier (E4a-fix4): p_yes >= threshold to compute. Default 0.5 until
    # evals/ext_threshold.py picks a value from train/dev (never from public).
    ext_verify_threshold: float = 0.5
    # Verifier prompt with the balanced few-shot set (False = instruction only).
    ext_verify_few_shot: bool = True

    @field_validator("cache_dir", "runs_dir", mode="after")
    @classmethod
    def _resolve_project_dir(cls, value: Path) -> Path:
        if not value.is_absolute():
            return PROJECT_ROOT / value
        return value

    def split_dir(self, split: str) -> Path:
        path = self.data_root / split
        if not path.exists():
            raise FileNotFoundError(
                f"Split directory for {split!r} does not exist at {path} (data_root={self.data_root})"
            )
        return path


settings = Settings()
