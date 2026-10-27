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

**4 trạng thái chuẩn:** `completed` (hoàn thành), `cancelled` (huỷ), `returned` (trả hàng), `shipping` (đang giao).

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
