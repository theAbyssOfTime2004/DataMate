from __future__ import annotations

from datamate.mapping.model import Entity

FIELD_DOCS: dict[Entity, dict[str, str]] = {
    "orders": {
        "order_id": "unique order / bill / invoice identifier",
        "order_date": (
            "date the order was placed (not delivery date or payment date); "
            "the period of an order is decided by this field"
        ),
        "order_ts": (
            "timestamp (date + time of day) when the order was created, "
            "used for morning/afternoon shift or peak hour"
        ),
        "status": (
            "raw order status text (will be mapped to completed, cancelled, returned, shipping)"
        ),
        "customer_id": "customer code / identifier (only when the shop uses customer codes)",
        "customer_name": "customer name recorded on the order",
        "payment_method": "payment method recorded on the order (cash, transfer, COD, credit/debt)",
        "channel": "sales channel (in-store, phone, delivery app, e-commerce)",
        "employee_id": "salesperson / cashier identifier or name",
        "order_subtotal": (
            "total merchandise amount of the order BEFORE order-level discount and shipping fee"
        ),
        "order_discount": "order-level discount / promotion / shop voucher on the whole order",
        "shipping_fee": "delivery / shipping fee of the order",
        "order_payable": (
            "final amount customer must pay for the whole order "
            "(after order-level discount, including shipping fee)"
        ),
    },
    "order_lines": {
        "order_id": "order / bill / invoice identifier linking the line to its order",
        "line_no": "line sequence number within the order (if explicitly present in the table)",
        "product_id": "product / item code (if present)",
        "product_name": "product / item name recorded on the line",
        "qty": "quantity sold on the line (in the selling unit of the line)",
        "unit": "selling unit recorded on the line (if in a separate column from qty)",
        "qty_base": (
            "quantity already converted to the product's base unit "
            "(only when the table has a pre-converted column)"
        ),
        "unit_price": "unit selling price BEFORE line-level discount",
        "line_discount": "discount amount on this specific line",
        "line_amount": (
            "line total amount AFTER line-level discount "
            "(qty * unit_price - line_discount); shop revenue is the sum of this field"
        ),
        "cost_per_unit": (
            "cost of goods sold per 1 selling unit recorded on the line (used for gross profit)"
        ),
    },
    "products": {
        "product_id": "product / item code in the product catalog",
        "product_name": "product / item name in the product catalog",
        "base_unit": "smallest / base counting unit of the product",
        "category": "product category / group name",
        "unit_cost": "cost price per 1 base unit in the product catalog",
    },
    "unit_conversions": {
        "product_id": "product code (if the conversion table uses product codes)",
        "product_name": "product name (if the conversion table uses product names)",
        "unit": "larger unit name (e.g. box, carton, pack)",
        "base_unit": "base / smallest unit name (if present in the conversion table)",
        "factor": "number of base units in 1 larger unit (1 unit = factor base_units)",
    },
    "customers": {
        "customer_id": "unique customer code",
        "customer_name": "customer full name",
        "segment": "customer group / tier / segment",
    },
}

REQUIRED_FIELDS: dict[str, set[str]] = {
    "order_lines": {"line_amount"},
    "orders": {"order_date", "status"},
}
