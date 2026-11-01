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
        "debt_total",
        "top_debtors",
        "debt_aging",
        "debt_added",
        "debt_collected",
        "purchase_total",
        "top_supplier",
        "avg_purchase_cost",
        "stock_on_hand",
        "stock_value",
        "ingredient_revenue",
        "rx_share",
        "gross_profit",
        "top_employee",
        # Off until dev shows >= 90% (no dev question reached them yet; lot remaining also
        # needs purchase-line lots and returns): out_of_stock_count, lots_expiring,
        # nearest_expiry_lot, net_profit.
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
    # Stricter verifier bar when one ext vote said "none" and the other gave a spec.
    ext_verify_threshold_split: float = 0.9
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


def apply_runtime_paths(*, data_dir: Path, cache_dir: Path) -> str:
    """Point global settings at an offline split directory for predict / Docker.

    `data_dir` is the split root (questions.jsonl, data/, calendar.csv). Caches go under
    `cache_dir` (never under data_dir). Returns the split name used by IO helpers so that
    `settings.split_dir(name) == data_dir.resolve()`.
    """
    resolved_data = data_dir.resolve()
    if not resolved_data.is_dir():
        raise FileNotFoundError(f"data dir does not exist: {resolved_data}")
    resolved_cache = cache_dir.resolve()
    resolved_cache.mkdir(parents=True, exist_ok=True)
    # Mutable BaseSettings fields; keep IO helpers (split_dir / cache_dir) working as-is.
    settings.data_root = resolved_data.parent
    settings.cache_dir = resolved_cache
    return resolved_data.name
