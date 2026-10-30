from __future__ import annotations

from typing import Literal

from datamate.mapping.model import Entity

FieldType = Literal["date", "timestamp", "money", "qty", "decimal", "text", "bool", "id"]

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
            "order state only (done / cancelled / returned / delivering); "
            "leave null if there is no such column — do not use a free-text note"
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
        "amount_paid": (
            "amount the customer has paid so far on this order, ONLY when the table also "
            "has a separate order total; a lone 'Khách trả'/'Thanh toán' column is order_payable"
        ),
        "amount_due": (
            "only when a separate remaining-debt column exists (còn nợ / còn lại)"
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
        "phone": "customer phone number (SĐT / điện thoại)",
    },
    "payments": {
        "payment_id": "payment / receipt voucher id (mã phiếu thu / mã giao dịch)",
        "order_id": "related order / invoice id if the payment is tied to one order",
        "customer_id": "customer code on the payment",
        "customer_name": "customer name on the payment (khách hàng)",
        "pay_date": "date the money was received or refunded (ngày thu / ngày chi)",
        "pay_ts": "timestamp of the payment if present",
        "amount": "payment amount in VND (số tiền / thực thu)",
        "method": "payment method (tiền mặt / chuyển khoản / COD)",
        "kind_raw": (
            "raw payment kind text: collect debt / prepay / refund "
            "(thu nợ / trả trước / hoàn tiền)"
        ),
        "note": "free-text note on the payment",
    },
    "debts": {
        "order_id": "order id this debt row refers to",
        "customer_id": "customer code who owes",
        "customer_name": "customer name who owes (khách nợ)",
        "order_date": "order date associated with the debt",
        "due_date": "date the debt must be paid (hạn trả)",
        "amount_total": "original debt / order total",
        "amount_paid": "amount already paid toward the debt",
        "amount_remaining": (
            "money still owed; never a date or a number of days "
            "(còn nợ / dư nợ)"
        ),
        "as_of_date": "snapshot date of the debt balance",
    },
    "purchases": {
        "receipt_id": "purchase receipt / goods-receipt id (mã phiếu nhập)",
        "receipt_date": "date goods were received (ngày nhập)",
        "supplier_id": "supplier code (mã NCC)",
        "supplier_name": "supplier name (nhà cung cấp / NCC)",
        "status_raw": "raw purchase status (hoàn thành / hủy / nháp)",
        "total_amount": "total amount of the purchase receipt",
    },
    "purchase_lines": {
        "receipt_id": "purchase receipt id linking the line",
        "line_no": "line number within the receipt",
        "receipt_date": "receipt date repeated on the line if flat",
        "supplier_name": "supplier name repeated on the line if flat",
        "product_id": "purchased product code",
        "product_name": "purchased product name (tên hàng nhập)",
        "qty": "quantity purchased",
        "unit": "purchase unit",
        "qty_base": "quantity in base unit if pre-converted",
        "unit_cost": "cost per purchase unit (giá nhập)",
        "line_amount": "line total for the purchase line (thành tiền nhập)",
        "lot_id": "lot / batch id if recorded on the line",
        "expiry_date": "expiry date of the received lot (HSD)",
    },
    "suppliers": {
        "supplier_id": "supplier code",
        "supplier_name": "supplier full name (nhà cung cấp)",
    },
    "stock_snapshot": {
        "product_id": "product code in the stock snapshot",
        "product_name": "product name in the stock snapshot (tồn kho)",
        "qty_on_hand": "quantity on hand (số lượng tồn)",
        "unit": "unit of the on-hand quantity",
        "as_of_date": "date of the stock snapshot",
        "unit_cost": "unit cost used for stock valuation if present",
    },
    "stock_moves": {
        "product_id": "product code of the stock movement",
        "product_name": "product name of the stock movement",
        "move_date": "date of the stock in/out movement (ngày nhập/xuất kho)",
        "move_ts": "timestamp of the movement if present",
        "qty_in": "quantity received into stock",
        "qty_out": "quantity issued from stock",
        "balance": "running stock balance after the move",
        "unit": "unit of the moved quantity",
        "ref_id": "reference document id (phiếu liên quan)",
        "kind_raw": "raw movement kind (nhập / xuất / điều chỉnh)",
    },
    "lots": {
        "lot_id": "lot / batch identifier (mã lô)",
        "product_id": "product code of the lot",
        "product_name": "product name of the lot",
        "receipt_id": "purchase receipt that created the lot",
        "received_date": "date the lot was received",
        "expiry_date": "expiry / best-before date (hạn dùng / HSD)",
        "qty_received": "quantity received in the lot",
        "unit": "unit of qty_received",
        "unit_cost": "unit cost of the lot",
    },
    "sale_lots": {
        "order_id": "order id that sold from a lot",
        "line_no": "order line number",
        "product_id": "product code sold from the lot",
        "product_name": "product name sold from the lot",
        "lot_id": "lot id used for the sale",
        "qty": "quantity taken from the lot",
        "unit": "unit of the sold quantity",
    },
    "price_history": {
        "product_id": "product code whose price changed",
        "product_name": "product name whose price changed",
        "effective_date": "date the price becomes effective (ngày áp dụng giá)",
        "price": "selling price from that date (giá bán)",
        "unit": "unit the price applies to",
    },
    "expenses": {
        "expense_date": "date of the expense (ngày chi)",
        "amount": "expense amount in VND (số tiền chi)",
        "category": "expense category (loại chi / khoản mục)",
        "note": "free-text expense note",
    },
    "employees": {
        "employee_id": "employee / staff code (mã NV)",
        "employee_name": "employee full name (nhân viên)",
        "alias": "short name / nickname if present",
        "start_date": "employment start date",
        "commission_pct": "commission percent if recorded",
    },
    "deliveries": {
        "order_id": "order id being delivered",
        "delivery_date": "date of this delivery batch (ngày giao)",
        "batch_no": "delivery batch / trip number (đợt giao)",
        "qty": "quantity delivered in this batch",
        "amount": "amount associated with this delivery batch",
        "status_raw": "raw delivery status",
    },
    "drug_info": {
        "product_id": "product / drug code",
        "product_name": "product / drug name",
        "active_ingredient": "active ingredient (hoạt chất)",
        "is_prescription": "whether the drug requires a prescription (kê đơn / OTC)",
    },
}

FIELD_TYPES: dict[Entity, dict[str, FieldType]] = {
    "orders": {
        "order_id": "id",
        "order_date": "date",
        "order_ts": "timestamp",
        "status": "text",
        "customer_id": "id",
        "customer_name": "text",
        "payment_method": "text",
        "channel": "text",
        "employee_id": "id",
        "order_subtotal": "money",
        "order_discount": "money",
        "shipping_fee": "money",
        "order_payable": "money",
        "amount_paid": "money",
        "amount_due": "money",
    },
    "order_lines": {
        "order_id": "id",
        "line_no": "decimal",
        "product_id": "id",
        "product_name": "text",
        "qty": "qty",
        "unit": "text",
        "qty_base": "qty",
        "unit_price": "money",
        "line_discount": "money",
        "line_amount": "money",
        "cost_per_unit": "money",
    },
    "products": {
        "product_id": "id",
        "product_name": "text",
        "base_unit": "text",
        "category": "text",
        "unit_cost": "money",
    },
    "unit_conversions": {
        "product_id": "id",
        "product_name": "text",
        "unit": "text",
        "base_unit": "text",
        "factor": "decimal",
    },
    "customers": {
        "customer_id": "id",
        "customer_name": "text",
        "segment": "text",
        "phone": "text",
    },
    "payments": {
        "payment_id": "id",
        "order_id": "id",
        "customer_id": "id",
        "customer_name": "text",
        "pay_date": "date",
        "pay_ts": "timestamp",
        "amount": "money",
        "method": "text",
        "kind_raw": "text",
        "note": "text",
    },
    "debts": {
        "order_id": "id",
        "customer_id": "id",
        "customer_name": "text",
        "order_date": "date",
        "due_date": "date",
        "amount_total": "money",
        "amount_paid": "money",
        "amount_remaining": "money",
        "as_of_date": "date",
    },
    "purchases": {
        "receipt_id": "id",
        "receipt_date": "date",
        "supplier_id": "id",
        "supplier_name": "text",
        "status_raw": "text",
        "total_amount": "money",
    },
    "purchase_lines": {
        "receipt_id": "id",
        "line_no": "decimal",
        "receipt_date": "date",
        "supplier_name": "text",
        "product_id": "id",
        "product_name": "text",
        "qty": "qty",
        "unit": "text",
        "qty_base": "qty",
        "unit_cost": "money",
        "line_amount": "money",
        "lot_id": "id",
        "expiry_date": "date",
    },
    "suppliers": {
        "supplier_id": "id",
        "supplier_name": "text",
    },
    "stock_snapshot": {
        "product_id": "id",
        "product_name": "text",
        "qty_on_hand": "qty",
        "unit": "text",
        "as_of_date": "date",
        "unit_cost": "money",
    },
    "stock_moves": {
        "product_id": "id",
        "product_name": "text",
        "move_date": "date",
        "move_ts": "timestamp",
        "qty_in": "qty",
        "qty_out": "qty",
        "balance": "qty",
        "unit": "text",
        "ref_id": "id",
        "kind_raw": "text",
    },
    "lots": {
        "lot_id": "id",
        "product_id": "id",
        "product_name": "text",
        "receipt_id": "id",
        "received_date": "date",
        "expiry_date": "date",
        "qty_received": "qty",
        "unit": "text",
        "unit_cost": "money",
    },
    "sale_lots": {
        "order_id": "id",
        "line_no": "decimal",
        "product_id": "id",
        "product_name": "text",
        "lot_id": "id",
        "qty": "qty",
        "unit": "text",
    },
    "price_history": {
        "product_id": "id",
        "product_name": "text",
        "effective_date": "date",
        "price": "money",
        "unit": "text",
    },
    "expenses": {
        "expense_date": "date",
        "amount": "money",
        "category": "text",
        "note": "text",
    },
    "employees": {
        "employee_id": "id",
        "employee_name": "text",
        "alias": "text",
        "start_date": "date",
        "commission_pct": "decimal",
    },
    "deliveries": {
        "order_id": "id",
        "delivery_date": "date",
        "batch_no": "text",
        "qty": "qty",
        "amount": "money",
        "status_raw": "text",
    },
    "drug_info": {
        "product_id": "id",
        "product_name": "text",
        "active_ingredient": "text",
        "is_prescription": "bool",
    },
}

# Bold fields from docs/cursor/e1.md.
# drug_info: at least one of active_ingredient / is_prescription (infer + validate + ETL).
# debts: at least one of amount_remaining / due_date (same pattern).
SIMPLE_REQUIRED: dict[Entity, set[str]] = {
    "payments": {"pay_date", "amount"},
    "debts": set(),
    "purchases": {"receipt_id", "receipt_date"},
    "purchase_lines": {"qty"},
    "suppliers": {"supplier_name"},
    "stock_snapshot": {"qty_on_hand"},
    "stock_moves": {"move_date"},
    "lots": {"lot_id", "expiry_date"},
    "sale_lots": {"order_id", "lot_id"},
    "price_history": {"effective_date", "price"},
    "expenses": {"expense_date", "amount"},
    "employees": {"employee_name"},
    "deliveries": {"order_id", "delivery_date"},
    "drug_info": set(),
}

# Entities whose required set is empty but need at least one of these fields.
SIMPLE_AT_LEAST_ONE: dict[Entity, frozenset[str]] = {
    "debts": frozenset({"amount_remaining", "due_date"}),
    "drug_info": frozenset({"active_ingredient", "is_prescription"}),
}

# Order and customer fields added with the extended schema. Step B does not offer them:
# a longer field list changed how the model mapped columns it already knew ("Khách trả"
# became amount_paid, a sparse status column was dropped). They are mapped separately.
LATER_FIELDS: dict[Entity, frozenset[str]] = {
    "orders": frozenset({"amount_paid", "amount_due"}),
    "customers": frozenset({"phone"}),
}

REQUIRED_FIELDS: dict[str, set[str]] = {
    "order_lines": {"line_amount"},
    "orders": {"order_date"},
}

SIMPLE_ENTITIES: tuple[Entity, ...] = (
    "payments",
    "debts",
    "purchases",
    "purchase_lines",
    "suppliers",
    "stock_snapshot",
    "stock_moves",
    "lots",
    "sale_lots",
    "price_history",
    "expenses",
    "employees",
    "deliveries",
    "drug_info",
)
