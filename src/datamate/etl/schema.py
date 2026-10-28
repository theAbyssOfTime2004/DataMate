from __future__ import annotations

ORDERS_COLUMNS: tuple[str, ...] = (
    "order_id",
    "order_date",
    "order_ts",
    "status",
    "status_raw",
    "has_return",
    "customer_key",
    "customer_id",
    "customer_name",
    "is_walk_in",
    "payment_method",
    "channel",
    "employee_id",
    "order_subtotal",
    "order_discount",
    "shipping_fee",
    "order_payable",
)

ORDER_LINES_COLUMNS: tuple[str, ...] = (
    "order_id",
    "line_no",
    "product_key",
    "product_id",
    "product_name",
    "qty",
    "unit",
    "qty_base",
    "unit_price",
    "line_discount",
    "line_amount",
    "cost_per_unit",
    "is_return_line",
    "src_file",
    "src_sheet",
    "src_row",
)

PRODUCTS_COLUMNS: tuple[str, ...] = (
    "product_key",
    "product_id",
    "product_name",
    "base_unit",
    "category",
    "unit_cost",
)

UNIT_CONVERSIONS_COLUMNS: tuple[str, ...] = (
    "product_key",
    "unit",
    "base_unit",
    "factor",
)

CUSTOMERS_COLUMNS: tuple[str, ...] = (
    "customer_key",
    "customer_id",
    "customer_name",
    "segment",
)

# DuckDB must not infer DECIMAL width from the first Python Decimal values.
DECIMAL_SQL_TYPE = "DECIMAL(18,4)"
DECIMAL_COLUMNS: frozenset[str] = frozenset({"qty", "qty_base", "factor"})

TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "orders": ORDERS_COLUMNS,
    "order_lines": ORDER_LINES_COLUMNS,
    "products": PRODUCTS_COLUMNS,
    "unit_conversions": UNIT_CONVERSIONS_COLUMNS,
    "customers": CUSTOMERS_COLUMNS,
}

ORDERS_DDL = """
CREATE TABLE orders (
    order_id VARCHAR,
    order_date DATE,
    order_ts TIMESTAMP,
    status VARCHAR,
    status_raw VARCHAR,
    has_return BOOLEAN,
    customer_key VARCHAR,
    customer_id VARCHAR,
    customer_name VARCHAR,
    is_walk_in BOOLEAN,
    payment_method VARCHAR,
    channel VARCHAR,
    employee_id VARCHAR,
    order_subtotal BIGINT,
    order_discount BIGINT,
    shipping_fee BIGINT,
    order_payable BIGINT
);
""".strip()

ORDER_LINES_DDL = """
CREATE TABLE order_lines (
    order_id VARCHAR,
    line_no INTEGER,
    product_key VARCHAR,
    product_id VARCHAR,
    product_name VARCHAR,
    qty DECIMAL(18,4),
    unit VARCHAR,
    qty_base DECIMAL(18,4),
    unit_price BIGINT,
    line_discount BIGINT,
    line_amount BIGINT,
    cost_per_unit BIGINT,
    is_return_line BOOLEAN,
    src_file VARCHAR,
    src_sheet VARCHAR,
    src_row INTEGER
);
""".strip()

PRODUCTS_DDL = """
CREATE TABLE products (
    product_key VARCHAR,
    product_id VARCHAR,
    product_name VARCHAR,
    base_unit VARCHAR,
    category VARCHAR,
    unit_cost BIGINT
);
""".strip()

UNIT_CONVERSIONS_DDL = """
CREATE TABLE unit_conversions (
    product_key VARCHAR,
    unit VARCHAR,
    base_unit VARCHAR,
    factor DECIMAL(18,4)
);
""".strip()

CUSTOMERS_DDL = """
CREATE TABLE customers (
    customer_key VARCHAR,
    customer_id VARCHAR,
    customer_name VARCHAR,
    segment VARCHAR
);
""".strip()

TABLE_DDLS: dict[str, str] = {
    "orders": ORDERS_DDL,
    "order_lines": ORDER_LINES_DDL,
    "products": PRODUCTS_DDL,
    "unit_conversions": UNIT_CONVERSIONS_DDL,
    "customers": CUSTOMERS_DDL,
}
