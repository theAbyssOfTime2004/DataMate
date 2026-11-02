import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root (rules/, .env). An installed package lives in site-packages, so the Docker image
# sets DATAMATE_PROJECT_ROOT=/app.
PROJECT_ROOT = Path(os.environ.get("DATAMATE_PROJECT_ROOT") or Path(__file__).resolve().parents[2])


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
        "customer_debt_flow",
        "purchase_total",
        "top_supplier",
        "avg_purchase_cost",
        "stock_on_hand",
        "stock_value",
        "ingredient_revenue",
        "rx_share",
        "rx_revenue",
        "otc_revenue",
        "gross_profit",
        "top_employee",
        "employee_revenue",
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
    # A server that stops answering: after this many calls in a row get no response, calls
    # fail at once (LLM unavailable) instead of each waiting out its timeout; one call per
    # cooldown checks whether the server is back.
    llm_breaker_failures: int = 3
    llm_breaker_cooldown_s: float = 60.0
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
    # When set, pipeline loads this JSON as RuleConfig and skips rulebook.md LLM extract.
    rules_override_path: Path | None = None
    # Qwen3 thinking for the parse calls: "off"; "retry" = a second pass with thinking for
    # questions the fast pass abstained on (disagreement, parse error, out of scope,
    # verifier low); "always" = thinking for every parse (measurement only).
    # Retry on: dev 0.876 -> 0.911 (+61/-1), val 0.824 -> 0.857 (+25/-4), holdout +2/-0.
    parse_thinking: Literal["off", "retry", "always"] = "retry"
    thinking_max_tokens: int = 3072
    thinking_timeout_s: float = 300.0
    # Two votes that differ only in date ranges: ask which range the question means and
    # accept when both A/B orders agree at this probability (else disagreement abstain).
    # Off: on dev it picked 2 right and 5 wrong, all "bữa qua" (yesterday) misread.
    period_adjudicate: bool = False
    period_adjudicate_threshold: float = 0.9
    # E3 composed query plans (after main + ext both out of scope). Off: dev 4/4 but val 0/7
    # answered (the verifier accepts plans that drop conditions like "cancelled" or
    # "delivery batches" in misspelled questions).
    enable_plan_path: bool = False
    plan_verify_threshold: float = 0.9
    plan_sql_timeout_s: float = 5.0
    plan_enable_thinking: bool = False

    @field_validator("cache_dir", "runs_dir", "rules_override_path", mode="after")
    @classmethod
    def _resolve_project_dir(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
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
