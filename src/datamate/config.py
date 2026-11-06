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
        "delivery_count",
        # Off until dev shows >= 90% (no dev question reached them yet; lot remaining also
        # needs purchase-line lots and returns): out_of_stock_count, lots_expiring,
        # nearest_expiry_lot, net_profit, orders_min_deliveries (no train question).
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
    # Questions asking for an opinion the records cannot hold (taste, attitude, skill,
    # advice) answer "unanswerable" when the yes/no check reaches this probability.
    # On all 3000 train questions it fired on 7, all gold unanswerable (0 false positives).
    subjective_check: bool = True
    subjective_threshold: float = 0.9
    # A picked product that shares only some words with the question is accepted when a
    # yes/no check says the other words are synonyms (lợn / heo, rán / chiên). On 37
    # train questions rewritten with regional words: 4 -> 32 right, 0 wrong.
    product_synonym_check: bool = True
    product_synonym_threshold: float = 0.9
    # A question that asks about a ranking term the rulebook defines ("Khách trung thành
    # nhất") is answered by the rank block from that definition, and never as an opinion.
    # 176 train questions rewritten with invented terms: 168 right, 0 wrong; real
    # dev/val/holdout answers unchanged.
    defined_term_check: bool = True
    # Store each shop's inferred mapping on disk (keyed by its files, the model and the
    # mapping/ETL code), so a new run skips the two validation ETL passes per shop.
    mapping_cache_enabled: bool = True
    # Ext families over per-order event logs (delivery batches). Off restores the ext
    # prompt and schema exactly as they were before these families. Off until a
    # same-GPU comparison shows no new wrongs (the first run mixed GPU change and prompt).
    event_families: bool = False
    # Shops with a delivery log ask a separate event parser first (delivery batches);
    # the extension prompt stays unchanged. Same-GPU check on the 117 ext questions of
    # the delivery shops: dev +6, val +3, no new wrongs.
    event_path: bool = True
    # A question the pipeline abstained on that counts the shop's record unit as the
    # rulebook names it ("lượt xe", "phiếu") is answered by the unit block. On the 124
    # train abstains: 1 answered (right), no new wrongs.
    unit_terms: bool = True
    # After an abstain, shops whose sales book has extra text columns (vehicle, job, note)
    # get the attribute block: filter, count, distinct values or a share. No train abstain
    # reaches it; on an invented salon shop it added 3 right answers and no wrong one.
    attribute_path: bool = True
    # After an abstain: a question about a product group in a shop that records no groups
    # (none mapped, and no raw column of at most this many kinds holds the name) is
    # unanswerable, as the rulebook says. Never fires on train (every train shop has groups);
    # on public it fired on 4 abstains, all in shops whose files carry no group at all.
    # After an abstain the parser cannot resolve, parse once more from the question
    # rewritten in plain Vietnamese (datamate.parse.plain). Numbers and product names
    # must survive the rewrite. Off: train abstains dev 16/16, val 12 right + 1 wrong
    # ("bữa qua" rewritten as "vừa qua"); the rewrite can change a period.
    plain_retry: bool = False
    # A question on a customer attribute (gender, street, group) whose value is in no
    # raw cell and that no column records (model p below the max) is unanswerable.
    # Train abstains: 7 fired, 7 right (dev 6, val 1).
    missing_attribute_check: bool = True
    missing_attribute_max_p: float = 0.2
    missing_group_check: bool = True
    missing_group_kinds: int = 50
    defined_term_threshold: float = 0.9
    rank_verify_threshold: float = 0.9

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
