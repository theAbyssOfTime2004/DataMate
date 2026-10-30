# Mapping chuẩn (gold) cho POC

**Chỉ dùng để đo.** Code lời giải (`src/`) không bao giờ được đọc thư mục này. Luật 4 cấm logic theo ID doanh nghiệp, và mapping chuẩn chính là logic theo ID doanh nghiệp.

Dùng cho hai việc:
1. **Lần chạy A:** pipeline đọc mapping chuẩn để ra trần điểm của kiến trúc.
2. **Đo bước gán cột:** so mapping do LLM tạo với mapping chuẩn, từng trường một.

## Trạng thái

Các file YAML ở đây là **bản nháp do Claude viết** từ README và từ việc thống kê dữ liệu thật. Trước khi dùng làm chuẩn, **bạn phải tự soát từng dòng**:
- `# ĐÃ KIỂM:` là chỗ Claude đã chạy script đối chiếu trên dữ liệu. Vẫn nên liếc qua.
- `# TODO kiểm:` là chỗ còn suy đoán. Bắt buộc mở file ra xem.

Soát xong một file thì đổi `status: draft` thành `status: verified` và ghi ngày.

Phần bảng mới (E1): `simple_status: draft | verified`. Phần đơn hàng (`status`) đã verified thì **không sửa**. Chỉ đổi `simple_status` sau khi đã soát purchases/payments/stock/... và các later fields (`amount_paid` / `amount_due` / `phone`).

## Schema chuẩn (bản POC)

Tên trường dùng trong mục `columns` của mỗi bảng:

| Thực thể | Trường | Ý nghĩa |
|---|---|---|
| `orders` | `order_id` | mã đơn |
| | `order_date` | ngày đặt (kỳ của đơn tính theo trường này, rulebook §1) |
| | `order_ts` | thời điểm tạo đơn (ngày + giờ), dùng cho ca sáng/chiều, giờ cao điểm |
| | `status` | trạng thái gốc, quy về 4 giá trị qua `status_map` |
| | `customer_id` | mã khách (nếu shop có mã khách) |
| | `customer_name` | tên khách ghi trên đơn |
| | `payment_method` | hình thức thanh toán ghi trên đơn |
| | `channel` | kênh bán |
| | `employee_id` | nhân viên bán |
| | `order_subtotal` | tiền hàng cả đơn, trước giảm giá đơn |
| | `order_discount` | giảm giá trên cả đơn |
| | `shipping_fee` | phí giao |
| | `order_payable` | số khách phải trả cả đơn (sau giảm giá đơn, gồm phí giao) |
| `order_lines` | `order_id`, `line_no` | khoá dòng |
| | `product_id` | mã hàng (nếu có) |
| | `product_name` | tên hàng ghi trên dòng |
| | `qty`, `unit` | số lượng và đơn vị ghi trên dòng |
| | `qty_base` | số lượng quy về đơn vị cơ bản (nếu file có sẵn) |
| | `unit_price` | đơn giá trước chiết khấu dòng |
| | `line_discount` | chiết khấu trên dòng |
| | `line_amount` | thành tiền dòng, **đã trừ** chiết khấu dòng. **Doanh thu = tổng trường này** (§1) |
| | `cost_per_unit` | giá vốn một đơn vị bán, ghi trên dòng (§5 Lãi gộp) |
| `products` | `product_id`, `product_name`, `base_unit`, `category`, `unit_cost` | danh mục hàng |
| `unit_conversions` | `product_id` hoặc `product_name`, `unit`, `base_unit`, `factor` | 1 `unit` = `factor` đơn vị cơ bản |
| `customers` | `customer_id`, `customer_name`, `segment` | danh sách khách |
| | `phone` | SĐT khách (later field, E1) |
| `orders` (later) | `amount_paid` | đã trả trên đơn, **chỉ khi** có tổng đơn riêng |
| | `amount_due` | còn nợ / còn lại trên đơn |

### Schema mở rộng (E1 — bảng đơn giản)

| Thực thể | Trường bắt buộc (giữ bảng) | Ghi chú |
|---|---|---|
| `payments` | `pay_date`, `amount` | phiếu thu / thanh toán |
| `debts` | `amount_remaining` | sổ công nợ |
| `purchases` | `receipt_id`, `receipt_date` | phiếu nhập; `purchase_status_map` nếu có `status_raw` |
| `purchase_lines` | `qty` | chi tiết nhập (có thể cùng sheet với purchases) |
| `suppliers` | `supplier_name` | danh mục NCC |
| `stock_snapshot` | `qty_on_hand` | tồn kho thời điểm |
| `stock_moves` | `move_date` | xuất nhập tồn / thẻ kho |
| `lots` | `lot_id`, `expiry_date` | lô / hạn dùng |
| `sale_lots` | `order_id`, `lot_id` | lô gắn dòng bán |
| `price_history` | `effective_date`, `price` | lịch sử giá |
| `expenses` | `expense_date`, `amount` | chi phí |
| `employees` | `employee_name` | nhân viên |
| `deliveries` | `order_id`, `delivery_date` | đợt giao |
| `drug_info` | ≥1 trong `active_ingredient` / `is_prescription` | hoạt chất / kê đơn |

**4 trạng thái chuẩn:** `completed` (hoàn thành), `cancelled` (huỷ), `returned` (trả hàng), `shipping` (đang giao).

**Trạng thái phiếu nhập:** `valid` | `cancelled` trong `purchase_status_map`.

## Định dạng file

```yaml
business_id: biz_xxx
status: draft | verified
data_end_date: YYYY-MM-DD          # README: "Dữ liệu có đến hết ..."
customer_identity: by_code | by_name   # rulebook §3 "Xác định khách"
walk_in_values: [...]              # giá trị nghĩa là khách lẻ / khách vãng lai
return_rule: status | negative_qty | return_table
status_map: {giá trị gốc: trạng thái chuẩn}
tables:
  <thực thể>:
    sources:
      - file: <tên file>
        sheet: <tên sheet> | "*" (mọi sheet) | null (csv)
        header_row: <số thứ tự dòng header, đếm từ 1>
    join: {...}                    # khi trường lấy từ bảng khác (ví dụ tên hàng lấy từ danh mục)
    columns: {trường chuẩn: tên cột gốc}
    row_filter: "..."              # dòng phải bỏ (dòng tổng, dòng ghi chú)
notes: [...]                       # điều cần biết khi ETL
```

## Checklist soát cho mỗi doanh nghiệp

1. Mở từng file/sheet: `header_row` có đúng không, có dòng tổng hay dòng ghi chú ở cuối không.
2. Mỗi trường trong `columns`: đọc README và nhìn 10 giá trị thật xem nghĩa có khớp không.
3. `status_map`: **phủ hết** mọi giá trị khác nhau của cột trạng thái, ở **mọi** sheet và file.
4. `walk_in_values`: phủ hết các cách ghi khách lẻ.
5. Tự tính tay **doanh thu một tháng** trong Excel (lọc đơn hoàn thành và đang giao, cộng `line_amount`), rồi đối chiếu với `answers.jsonl` nếu có câu hỏi về tháng đó. Đây là bài kiểm tra cuối cùng cho mapping.
