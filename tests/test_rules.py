from __future__ import annotations

import datetime

import pytest
from pydantic import ValidationError

from datamate.config import PROJECT_ROOT
from datamate.rules import RuleConfig, load_rules


def test_train_rules_match_the_rulebook_defaults() -> None:
    loaded = load_rules(PROJECT_ROOT / "rules" / "train.json")
    defaults = RuleConfig()
    assert loaded == defaults
    assert loaded.revenue_statuses == ["completed", "shipping"]
    assert loaded.exclude_orders_with_return is True
    assert loaded.cancel_rate_denominator == "all_orders"
    assert loaded.shifts["morning"] == (
        datetime.time(6, 0, 0),
        datetime.time(13, 59, 59),
    )
    assert loaded.shifts["afternoon"] == (
        datetime.time(14, 0, 0),
        datetime.time(21, 59, 59),
    )
    assert loaded.week_start == "monday"
    assert loaded.month_this_compare == "same_day_count"
    assert loaded.tet.start_lunar_month == 12
    assert loaded.tet.start_lunar_day == 23
    assert loaded.tet.start_year_offset == -1
    assert loaded.tet.end_lunar_month == 1
    assert loaded.tet.end_lunar_day == 10
    assert loaded.percent_decimals == 2
    assert loaded.rounding == "half_up"
    assert loaded.non_product_line_labels == ["Phí phục vụ"]


def test_rule_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        RuleConfig.model_validate({"extra_shop_rule": True})
