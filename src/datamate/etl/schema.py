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
    "amount_paid",
    "amount_due",
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
    "phone",
)

PAYMENTS_COLUMNS: tuple[str, ...] = (
    "payment_id",
    "order_id",
    "customer_key",
    "customer_id",
    "customer_name",
    "pay_date",
    "pay_ts",
    "amount",
    "method",
    "kind_raw",
    "note",
    "src_file",
    "src_sheet",
    "src_row",
)

DEBTS_COLUMNS: tuple[str, ...] = (
    "order_id",
    "customer_key",
    "customer_id",
    "customer_name",
    "order_date",
    "due_date",
    "amount_total",
    "amount_paid",
    "amount_remaining",
    "as_of_date",
    "src_file",
    "src_sheet",
    "src_row",
)

PURCHASES_COLUMNS: tuple[str, ...] = (
    "receipt_id",
    "receipt_date",
    "supplier_id",
    "supplier_name",
    "status_raw",
    "is_cancelled",
    "total_amount",
    "src_file",
    "src_sheet",
    "src_row",
)

PURCHASE_LINES_COLUMNS: tuple[str, ...] = (
    "receipt_id",
    "line_no",
    "receipt_date",
    "supplier_name",
    "product_key",
    "product_id",
    "product_name",
    "qty",
    "unit",
    "qty_base",
    "unit_cost",
    "line_amount",
    "lot_id",
    "expiry_date",
    "src_file",
    "src_sheet",
    "src_row",
)

SUPPLIERS_COLUMNS: tuple[str, ...] = (
    "supplier_id",
    "supplier_name",
    "src_file",
    "src_sheet",
    "src_row",
)

STOCK_SNAPSHOT_COLUMNS: tuple[str, ...] = (
    "product_key",
    "product_id",
    "product_name",
    "qty_on_hand",
    "unit",
    "as_of_date",
    "unit_cost",
    "src_file",
    "src_sheet",
    "src_row",
)

STOCK_MOVES_COLUMNS: tuple[str, ...] = (
    "product_key",
    "product_id",
    "product_name",
    "move_date",
    "move_ts",
    "qty_in",
    "qty_out",
    "balance",
    "unit",
    "ref_id",
    "kind_raw",
    "src_file",
    "src_sheet",
    "src_row",
)

LOTS_COLUMNS: tuple[str, ...] = (
    "lot_id",
    "product_key",
    "product_id",
    "product_name",
    "receipt_id",
    "received_date",
    "expiry_date",
    "qty_received",
    "unit",
    "unit_cost",
    "src_file",
    "src_sheet",
    "src_row",
)

SALE_LOTS_COLUMNS: tuple[str, ...] = (
    "order_id",
    "line_no",
    "product_key",
    "product_id",
    "product_name",
    "lot_id",
    "qty",
    "unit",
    "src_file",
    "src_sheet",
    "src_row",
)

PRICE_HISTORY_COLUMNS: tuple[str, ...] = (
    "product_key",
    "product_id",
    "product_name",
    "effective_date",
    "price",
    "unit",
    "src_file",
    "src_sheet",
    "src_row",
)

EXPENSES_COLUMNS: tuple[str, ...] = (
    "expense_date",
    "amount",
    "category",
    "note",
    "src_file",
    "src_sheet",
    "src_row",
)

EMPLOYEES_COLUMNS: tuple[str, ...] = (
    "employee_id",
    "employee_name",
    "alias",
    "start_date",
    "commission_pct",
    "src_file",
    "src_sheet",
    "src_row",
)

DELIVERIES_COLUMNS: tuple[str, ...] = (
    "order_id",
    "delivery_date",
    "batch_no",
    "qty",
    "amount",
    "status_raw",
    "src_file",
    "src_sheet",
    "src_row",
)

DRUG_INFO_COLUMNS: tuple[str, ...] = (
    "product_key",
    "product_id",
    "product_name",
    "active_ingredient",
    "is_prescription",
    "src_file",
    "src_sheet",
    "src_row",
)

# Text columns of the sales book that no canonical field takes (vehicle, job, note, ...),
# one row per order, column header and distinct value.
ORDER_ATTRIBUTES_COLUMNS: tuple[str, ...] = ("order_id", "name", "value")

# DuckDB must not infer DECIMAL width from the first Python Decimal values.
DECIMAL_SQL_TYPE = "DECIMAL(18,4)"
DECIMAL_COLUMNS: frozenset[str] = frozenset(
    {
        "qty",
        "qty_base",
        "factor",
        "qty_on_hand",
        "qty_in",
        "qty_out",
        "balance",
        "qty_received",
        "commission_pct",
    }
)

TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "orders": ORDERS_COLUMNS,
    "order_lines": ORDER_LINES_COLUMNS,
    "products": PRODUCTS_COLUMNS,
    "unit_conversions": UNIT_CONVERSIONS_COLUMNS,
    "customers": CUSTOMERS_COLUMNS,
    "payments": PAYMENTS_COLUMNS,
    "debts": DEBTS_COLUMNS,
    "purchases": PURCHASES_COLUMNS,
    "purchase_lines": PURCHASE_LINES_COLUMNS,
    "suppliers": SUPPLIERS_COLUMNS,
    "stock_snapshot": STOCK_SNAPSHOT_COLUMNS,
    "stock_moves": STOCK_MOVES_COLUMNS,
    "lots": LOTS_COLUMNS,
    "sale_lots": SALE_LOTS_COLUMNS,
    "price_history": PRICE_HISTORY_COLUMNS,
    "expenses": EXPENSES_COLUMNS,
    "employees": EMPLOYEES_COLUMNS,
    "deliveries": DELIVERIES_COLUMNS,
    "drug_info": DRUG_INFO_COLUMNS,
    "order_attributes": ORDER_ATTRIBUTES_COLUMNS,
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
    order_payable BIGINT,
    amount_paid BIGINT,
    amount_due BIGINT
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
    segment VARCHAR,
    phone VARCHAR
);
""".strip()

PAYMENTS_DDL = """
CREATE TABLE payments (
    payment_id VARCHAR,
    order_id VARCHAR,
    customer_key VARCHAR,
    customer_id VARCHAR,
    customer_name VARCHAR,
    pay_date DATE,
    pay_ts TIMESTAMP,
    amount BIGINT,
    method VARCHAR,
    kind_raw VARCHAR,
    note VARCHAR,
    src_file VARCHAR,
    src_sheet VARCHAR,
    src_row INTEGER
);
""".strip()

DEBTS_DDL = """
CREATE TABLE debts (
    order_id VARCHAR,
    customer_key VARCHAR,
    customer_id VARCHAR,
    customer_name VARCHAR,
    order_date DATE,
    due_date DATE,
    amount_total BIGINT,
    amount_paid BIGINT,
    amount_remaining BIGINT,
    as_of_date DATE,
    src_file VARCHAR,
    src_sheet VARCHAR,
    src_row INTEGER
);
""".strip()

PURCHASES_DDL = """
CREATE TABLE purchases (
    receipt_id VARCHAR,
    receipt_date DATE,
    supplier_id VARCHAR,
    supplier_name VARCHAR,
    status_raw VARCHAR,
    is_cancelled BOOLEAN,
    total_amount BIGINT,
    src_file VARCHAR,
    src_sheet VARCHAR,
    src_row INTEGER
);
""".strip()

PURCHASE_LINES_DDL = """
CREATE TABLE purchase_lines (
    receipt_id VARCHAR,
    line_no INTEGER,
    receipt_date DATE,
    supplier_name VARCHAR,
    product_key VARCHAR,
    product_id VARCHAR,
    product_name VARCHAR,
    qty DECIMAL(18,4),
    unit VARCHAR,
    qty_base DECIMAL(18,4),
    unit_cost BIGINT,
    line_amount BIGINT,
    lot_id VARCHAR,
    expiry_date DATE,
    src_file VARCHAR,
    src_sheet VARCHAR,
    src_row INTEGER
);
""".strip()

SUPPLIERS_DDL = """
CREATE TABLE suppliers (
    supplier_id VARCHAR,
    supplier_name VARCHAR,
    src_file VARCHAR,
    src_sheet VARCHAR,
    src_row INTEGER
);
""".strip()

STOCK_SNAPSHOT_DDL = """
CREATE TABLE stock_snapshot (
    product_key VARCHAR,
    product_id VARCHAR,
    product_name VARCHAR,
    qty_on_hand DECIMAL(18,4),
    unit VARCHAR,
    as_of_date DATE,
    unit_cost BIGINT,
    src_file VARCHAR,
    src_sheet VARCHAR,
    src_row INTEGER
);
""".strip()

STOCK_MOVES_DDL = """
CREATE TABLE stock_moves (
    product_key VARCHAR,
    product_id VARCHAR,
    product_name VARCHAR,
    move_date DATE,
    move_ts TIMESTAMP,
    qty_in DECIMAL(18,4),
    qty_out DECIMAL(18,4),
    balance DECIMAL(18,4),
    unit VARCHAR,
    ref_id VARCHAR,
    kind_raw VARCHAR,
    src_file VARCHAR,
    src_sheet VARCHAR,
    src_row INTEGER
);
""".strip()

LOTS_DDL = """
CREATE TABLE lots (
    lot_id VARCHAR,
    product_key VARCHAR,
    product_id VARCHAR,
    product_name VARCHAR,
    receipt_id VARCHAR,
    received_date DATE,
    expiry_date DATE,
    qty_received DECIMAL(18,4),
    unit VARCHAR,
    unit_cost BIGINT,
    src_file VARCHAR,
    src_sheet VARCHAR,
    src_row INTEGER
);
""".strip()

SALE_LOTS_DDL = """
CREATE TABLE sale_lots (
    order_id VARCHAR,
    line_no INTEGER,
    product_key VARCHAR,
    product_id VARCHAR,
    product_name VARCHAR,
    lot_id VARCHAR,
    qty DECIMAL(18,4),
    unit VARCHAR,
    src_file VARCHAR,
    src_sheet VARCHAR,
    src_row INTEGER
);
""".strip()

PRICE_HISTORY_DDL = """
CREATE TABLE price_history (
    product_key VARCHAR,
    product_id VARCHAR,
    product_name VARCHAR,
    effective_date DATE,
    price BIGINT,
    unit VARCHAR,
    src_file VARCHAR,
    src_sheet VARCHAR,
    src_row INTEGER
);
""".strip()

EXPENSES_DDL = """
CREATE TABLE expenses (
    expense_date DATE,
    amount BIGINT,
    category VARCHAR,
    note VARCHAR,
    src_file VARCHAR,
    src_sheet VARCHAR,
    src_row INTEGER
);
""".strip()

EMPLOYEES_DDL = """
CREATE TABLE employees (
    employee_id VARCHAR,
    employee_name VARCHAR,
    alias VARCHAR,
    start_date DATE,
    commission_pct DECIMAL(18,4),
    src_file VARCHAR,
    src_sheet VARCHAR,
    src_row INTEGER
);
""".strip()

DELIVERIES_DDL = """
CREATE TABLE deliveries (
    order_id VARCHAR,
    delivery_date DATE,
    batch_no VARCHAR,
    qty DECIMAL(18,4),
    amount BIGINT,
    status_raw VARCHAR,
    src_file VARCHAR,
    src_sheet VARCHAR,
    src_row INTEGER
);
""".strip()

DRUG_INFO_DDL = """
CREATE TABLE drug_info (
    product_key VARCHAR,
    product_id VARCHAR,
    product_name VARCHAR,
    active_ingredient VARCHAR,
    is_prescription BOOLEAN,
    src_file VARCHAR,
    src_sheet VARCHAR,
    src_row INTEGER
);
""".strip()

ORDER_ATTRIBUTES_DDL = """
CREATE TABLE order_attributes (
    order_id VARCHAR,
    name VARCHAR,
    value VARCHAR
);
""".strip()

TABLE_DDLS: dict[str, str] = {
    "orders": ORDERS_DDL,
    "order_lines": ORDER_LINES_DDL,
    "products": PRODUCTS_DDL,
    "unit_conversions": UNIT_CONVERSIONS_DDL,
    "customers": CUSTOMERS_DDL,
    "payments": PAYMENTS_DDL,
    "debts": DEBTS_DDL,
    "purchases": PURCHASES_DDL,
    "purchase_lines": PURCHASE_LINES_DDL,
    "suppliers": SUPPLIERS_DDL,
    "stock_snapshot": STOCK_SNAPSHOT_DDL,
    "stock_moves": STOCK_MOVES_DDL,
    "lots": LOTS_DDL,
    "sale_lots": SALE_LOTS_DDL,
    "price_history": PRICE_HISTORY_DDL,
    "expenses": EXPENSES_DDL,
    "employees": EMPLOYEES_DDL,
    "deliveries": DELIVERIES_DDL,
    "drug_info": DRUG_INFO_DDL,
    "order_attributes": ORDER_ATTRIBUTES_DDL,
}
