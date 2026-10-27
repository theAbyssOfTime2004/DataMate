# DataMate — Kế hoạch triển khai

Zalo AI Challenge 2026, track DataMate. Agent trả lời câu hỏi tiếng Việt trên dữ liệu bảng bẩn của cửa hàng nhỏ, chạy offline, tổng tham số model ≤ 15B. Đúng **+1**, sai **−1**, bỏ qua **0**.

Kế hoạch gồm hai phần:
- **POC (28–31/10):** chạy xuyên suốt từ đầu đến cuối trên phạm vi hẹp, để trả lời câu hỏi "hướng này có ăn điểm không". Phần này viết chi tiết.
- **End-to-end (01–16/11):** mở rộng, đóng gói, nộp bài. Phần này chỉ phác ở mức stage, sẽ viết lại sau buổi quyết định ngày 31/10.

---

## 0. Mốc thời gian cứng

| Ngày | Việc | Ghi chú |
|---|---|---|
| 28–30/10 | POC | |
| **31/10** | **Họp quyết định** (đi tiếp / chỉnh hướng) | Theo tiêu chí ở mục 5 |
| 01–13/11 | End-to-end | Docker phải chạy được **từ khoảng 05/11** |
| 14/11 | Đóng băng tính năng | Chỉ sửa lỗi |
| 15/11 | Nộp Docker (sớm 1 ngày) | Hạn thật: **23:59 ngày 16/11**. Mỗi đội được nộp 2 lần, ban tổ chức chấm lần cuối |
| 20–23/11 | Private test | Chạy lời giải đã đóng băng, nộp kết quả |

Public leaderboard cho nộp 5 lần/ngày. Từ khoảng 03/11 trở đi nộp thường xuyên để lấy phản hồi.

---

## 1. Những gì đã biết từ dữ liệu (27/10)

**Quy mô:**
- Train: 30 doanh nghiệp, 3.000 câu hỏi, có đáp án.
- Public: 14 doanh nghiệp, 1.000 câu hỏi.
- Private: 10–20 doanh nghiệp, khoảng 500 câu hỏi, **có ngành mới**.

**Kiểu đáp án (train):**

| number | percent | list | date | string | unanswerable | boolean | table |
|---|---|---|---|---|---|---|---|
| 63% | 19% | 6% | 5% | 3% | **2,3%** | <1% | <1% |

Hệ quả:
- **82% là con số phải khớp tuyệt đối.**
- `unanswerable` hiếm, nên dùng nó rất thận trọng. Công cụ an toàn là `abstain`.

**Chủ đề:** khoảng 38% câu nhắc tới doanh thu. Tiếp theo là khách hàng, đơn huỷ, "hôm nay / tháng này / tuần", công nợ, nhập hàng.

**Câu hỏi bộ public diễn đạt khác hẳn bộ train:** sau khi bỏ dấu và thay số bằng ký hiệu, chỉ **51/1.000** câu public trùng mẫu với train. Không được dựa vào mẫu câu.

**Rulebook mỗi bộ một khác.** Bộ public so với train:
- **thêm:** khách trung thành, khách sộp, giờ cao điểm, nhà cung cấp chính, trung bình mỗi thứ X, topping ghi chung ô, báo cáo tiền mặt theo ca, ngành **Mẹ và bé**;
- **bớt:** vài luật của train.

Bộ private sẽ còn khác nữa. Hệ quả: **luật phải được đọc từ `rulebook.md` lúc chạy**, không viết cứng.

**Dữ liệu sinh theo khoảng 9 họ định dạng.** Ngành và mức độ bẩn thay đổi, nhưng cấu trúc file lặp lại:

| Họ định dạng | Doanh nghiệp train |
|---|---|
| CSV phẳng, mỗi dòng một dòng hàng | 101, 114, 125 |
| CSV quan hệ (đơn, chi tiết, khách, sản phẩm, thanh toán) | 111, 127, 129 |
| CSV chỉ có hoá đơn, không có dòng hàng | 110 |
| Excel phần mềm bán hàng kiểu `DanhSach*` | 106, 108, 119, 120, 128 |
| Excel `DonHang_ChiTiet` + `DanhSachQuyDoi` | 102, 103, 104, 121, 122 |
| Excel `So_chi_tiet_ban_hang` / báo cáo | 112, 115, 116, 118, 130 |
| Sổ ghi tay, mỗi tháng một sheet | 105, 117 |
| Sổ ghi tay, một sheet | 107, 113, 126 |
| SQLite của máy POS (kèm nguồn hỗn hợp) | 123, 124 |

**Các kiểu bẩn đã thấy:**
- header 2 tầng;
- dòng tiêu đề báo cáo trước bảng ("Từ ngày … đến ngày …");
- ngày lẫn số serial của Excel (`45689`), `01/02/2025`, `1 thg 6 2025`;
- tiền dạng `55.000đ`, `25k`, `"560,000"`;
- số lượng dính đơn vị (`1 tuýp`);
- trạng thái dùng từ đồng nghĩa (`đã lấy`, `đã xong`, `ok`, `Xong`, `completed`);
- khách lẻ ghi `lẻ` / `Khách lẻ`;
- tên hàng khác nhau hoa thường;
- dòng trả hàng ghi số lượng âm;
- ô trống ở các dòng sau trong cùng một đơn (rulebook §8).

**README:**
- thường ghi **ngày cuối có dữ liệu**, là căn cứ để trả lời `unanswerable` theo §7;
- ghi luật riêng của shop, ví dụ "dòng số lượng âm = đơn có trả hàng".

---

## 2. Kiến trúc

```
                      ┌─ rulebook.md ──► [Rule config]  (JSON có schema)
data/<biz>/ ──► [Raw reader] ──► raw tables + profile cards
                                   │
                       README ──► [Mapper (LLM)] ──► mapping JSON ──► [Validator]
                                   │
                                   ▼
                          [Canonical ETL]  → DuckDB: orders, order_lines, products, ...
                                   │
question + as_of ─► [Question parser (LLM)] ─► QuerySpec ─► [Time resolver] ─► khoảng ngày
                                   │
                         ┌─────────┴──────────┐
                  [Metric library]      [SQL codegen (LLM)]   ← đường dự phòng (end-to-end)
                         └─────────┬──────────┘
                                   ▼
                     [Verifier + abstain policy] ─► [Formatter] ─► submission.jsonl + logs/
```

**Nguyên tắc:**
1. **LLM quyết định *cái gì*, code quyết định *tính thế nào*.**
   - LLM: gán cột, hiểu câu hỏi, đọc luật mới.
   - Code: parse dữ liệu, tính khoảng ngày, áp luật, định dạng đáp án.
2. **Tiền xử lý mỗi doanh nghiệp một lần**, cache lại. Câu hỏi chỉ chạy trên các bảng chuẩn trong DuckDB.
3. **Không chắc thì `abstain`.** Một đáp án có xác suất đúng *p* có kỳ vọng điểm 2p − 1, nên chỉ trả lời khi p > 0,5.
4. **Không có logic theo ID doanh nghiệp hay ID câu hỏi** (luật 4). Mọi khác biệt giữa các shop phải đến từ mapping, config hoặc README đọc lúc chạy.
5. **Tất định:** temperature 0, seed cố định, cache theo nội dung đầu vào. Ban tổ chức chạy lại phải lệch ≤ 0,03 PA.

---

## 3. Model và phần cứng

- **Dev:** `Qwen/Qwen3-4B-Instruct-2507-FP8` trên vLLM ở máy local (RTX 5050 8 GB), dùng lại cấu hình docker-compose của Shiki.
- **Bài nộp:** **một** model cỡ khoảng 14B cho mọi bước, vì giới hạn 15B tính tổng mọi model. Ứng viên: dòng Qwen3 14B, bản FP8.
  - **Trước khi chốt:** kiểm tra ngày phát hành weights < 01/10/2026, giấy phép cho phép dùng, và vLLM trong base image của ban tổ chức chạy được model đó.
- **Thuê RTX 5090** theo giờ (RunPod / vast.ai) để: chạy model 14B trên tập validation, test Docker, đo thời gian. Máy chấm là 2 × 5090, nên thuê đúng loại card này để môi trường giống nhau.
- **Fine-tune:** chưa làm. Chỉ cân nhắc sau 10/11 nếu còn thời gian, và phải nộp kèm code train.

---

## 4. Phương pháp đo

- **Chỉ số chính:** PA trên tập validation, kèm theo:
  - **độ phủ** (tỉ lệ câu dám trả lời);
  - **độ chính xác trên số câu đã trả lời**;
  - PA tách theo kiểu đáp án và theo chủ đề câu hỏi.
- **Chia tập theo doanh nghiệp, không theo câu hỏi:**
  - giữ riêng 6 doanh nghiệp làm *validation*;
  - giữ riêng hẳn **một họ định dạng** để mô phỏng "doanh nghiệp kiểu mới".
  - Chia theo câu hỏi sẽ cho điểm đẹp hơn thực tế.
- **Chỉ số riêng cho bước gán cột:** số trường gán đúng trên tổng số trường, so với mapping chuẩn viết tay.
- **Thời gian mỗi câu** và thời gian tiền xử lý mỗi doanh nghiệp. Ngân sách ở máy chấm: khoảng 500 câu trong 6 giờ.
- Mỗi lần chạy ghi lại: commit, model, config, seed, và kết quả từng câu (`runs/<timestamp>/`).

---

## 5. POC (28–31/10)

**Phạm vi:**
- **5 doanh nghiệp, 5 họ định dạng khác nhau:**
  - `biz_101`: CSV phẳng;
  - `biz_105`: sổ ghi tay mỗi tháng một sheet, header 2 tầng, ngày lẫn kiểu;
  - `biz_106`: Excel POS `DanhSach*` nhiều file;
  - `biz_111`: CSV quan hệ;
  - `biz_123`: SQLite.
- **Nhóm câu hỏi:** doanh thu, số đơn, tỉ lệ huỷ, phần trăm thay đổi và chênh lệch giữa hai kỳ. Trên 5 doanh nghiệp này có khoảng 250/514 câu thuộc nhóm đó.

### P0. Repo và môi trường (0,5 ngày)
- `git init`.
- `.gitignore` chặn: dữ liệu, `runs/`, cache, `.env`.
- `pyproject.toml`. Thư viện chính: `duckdb`, `pandas`, `openpyxl`, `pydantic`, `httpx`, `pytest`, `ruff`.
- Module `config` (pydantic-settings): `DATA_ROOT=D:\zaloaic`, `LLM_BASE_URL`, `LLM_MODEL`, `CACHE_DIR`.
- **Quy ước bắt buộc:** code không bao giờ ghi vào thư mục dữ liệu (luật 5: `/data` chỉ đọc).

### P1. Bộ chấm và bộ chạy thử (0,5 ngày) — làm đầu tiên
- **`scoring.py`:** cài đúng luật so khớp ở §9 rulebook:
  - kiểu đáp án phải khớp;
  - số khớp tuyệt đối, phần trăm khớp sau khi làm tròn 2 chữ số;
  - chuỗi được chuẩn hoá (NFC, chữ thường, bỏ dấu câu, gộp khoảng trắng) rồi so với đáp án và các `alias`;
  - list phải đúng cả tập lẫn thứ tự;
  - table: so cột theo tên, không khớp tên thì theo vị trí; thứ tự dòng không quan trọng;
  - trả lời một câu `unanswerable` thì tính sai.
  - **Test:** dùng chính các ví dụ đúng/sai trong bảng §9.
- **`splits.py`:** chia tập theo doanh nghiệp, cố định bằng seed.
- **`run_eval.py`:** chạy một hệ thống trả lời trên một tập câu hỏi. Ghi `submission.jsonl`, log từng câu, rồi in PA, độ phủ, độ chính xác, tách theo kiểu đáp án.
- **Mốc so sánh đầu tiên:** hệ thống "luôn abstain" phải ra PA = 0. Đây là bài test cho chính bộ chấm.

### P2. Đọc file thô và thẻ mô tả bảng (0,5 ngày)
- Đọc mọi file `.csv`, `.xlsx` (từng sheet), `.sqlite` (từng bảng) thành bảng thô, **giữ nguyên chuỗi** chưa parse.
- **Tìm dòng header:**
  - bỏ dòng tiêu đề báo cáo và dòng trống ở đầu;
  - header 2 tầng thì lấy tầng có nhiều ô khác rỗng và khác nhau hơn.
  - Ghi lại đã bỏ những dòng nào.
- **Thẻ mô tả** cho mỗi bảng: tên file/sheet, số dòng, và với mỗi cột: tên, 5 giá trị mẫu, số giá trị khác nhau, tỉ lệ rỗng, kiểu đoán được (ngày / tiền / số / chuỗi).
- Gộp các sheet cùng header (sổ mỗi tháng một sheet) thành một bảng, kèm cột nguồn (tên sheet) vì §8 có luật "ngày không ghi năm thì lấy năm của sheet".

### P3. Schema chuẩn, bộ parse, và mapping chuẩn viết tay (1 ngày)
- **Schema chuẩn bản POC:**
  - `orders(order_id, order_ts, status, customer_key, payment_method, channel, ...)`
  - `order_lines(order_id, line_no, product_key, product_name, qty, unit, qty_base, unit_price, line_amount, is_service_fee)`
  - `products(product_key, name, base_unit, category)`
  - `unit_conversions(product_key, unit, factor)`
- **Bộ parse:** ngày (serial Excel, `dd/mm/yyyy`, `1 thg 6 2025`, có giờ hoặc không), tiền (`55.000đ`, `25k`, `"560,000"`), số lượng có đơn vị, trạng thái → 4 giá trị chuẩn (hoàn thành / huỷ / trả hàng / đang giao).
  - Bảng từ đồng nghĩa của trạng thái **không viết cứng theo shop**: bộ parse đề xuất, LLM gán ở bước P4.
  - Mỗi bộ parse có test riêng.
- **Luật §8 làm ở bước ETL:**
  - ô trống ở các dòng sau của đơn thì kế thừa từ dòng đầu;
  - hai dòng giống hệt nhau trong cùng một đơn thì chỉ tính một lần;
  - đơn có dòng số lượng âm là đơn trả hàng (khi README ghi như vậy).
- **Mapping chuẩn viết tay cho 5 doanh nghiệp**, đặt ở `evals/gold_mappings/`. **Chỉ dùng để đo**, không bao giờ được code lời giải đọc tới. Có test grep để chặn import từ thư mục này.

### P4. LLM tự gán cột và bộ kiểm tra mapping (0,5 ngày)
- **Prompt đầu vào:** README, các thẻ mô tả bảng, danh sách trường của schema chuẩn kèm mô tả từng trường. **Đầu ra:** mapping JSON theo schema Pydantic, gồm:
  - bảng nào ứng với thực thể nào;
  - cột nào ứng với trường nào;
  - từ trạng thái nào ứng với trạng thái chuẩn nào;
  - quy ước trả hàng, ngày cuối có dữ liệu.
- **Bộ kiểm tra (code):**
  - trường bắt buộc phải có;
  - parse được ít nhất 98% giá trị;
  - nếu có cả tổng đơn và dòng hàng, tổng `line_amount` phải khớp tiền hàng của đơn ở phần lớn các đơn;
  - khoảng ngày phải hợp lý.
  - Không qua thì gọi lại LLM kèm lỗi cụ thể, tối đa 2 lần.
- **Chỉ số:** độ chính xác của mapping so với mapping chuẩn, tính theo từng trường.

### P5. Rule config, tính khoảng thời gian, thư viện số đo (1 ngày)
- **Rule config (bản POC):** viết tay thành JSON theo schema, lấy từ `train/rulebook.md`, chỉ những luật POC cần:
  - đơn nào tính vào doanh thu;
  - doanh thu một đơn tính ra sao;
  - định nghĩa tuần, tháng này, so sánh với kỳ trước;
  - mẫu số của tỉ lệ huỷ;
  - cách làm tròn.
  - Ở end-to-end, LLM sẽ sinh config này từ rulebook (xem E2).
- **Tính khoảng thời gian (code thuần):** hôm nay, hôm qua, ngày X, tuần này, tuần trước, tháng M/Y, tháng M không ghi năm, quý, N ngày gần nhất, từ đầu năm, Tết năm Y (tra `calendar.csv`), "so với tháng trước cùng số ngày" (§2).
  - **Test:** ít nhất mỗi dòng của bảng §2 một ca, kèm các ca biên (tháng 2, cuối năm, Tết vắt qua hai năm dương lịch).
- **Thư viện số đo:** `revenue`, `order_count`, `cancel_rate`, `pct_change`, `diff`, nhận vào khoảng ngày, bộ lọc và rule config.
- **Xử lý dữ liệu thiếu:**
  - kỳ nằm **hoàn toàn** ngoài khoảng ngày có dữ liệu → `unanswerable`;
  - nằm một phần → tính trên phần có dữ liệu (§7).

### P6. Hiểu câu hỏi, định dạng đáp án, quy tắc bỏ qua (0,5 ngày)
- **LLM chuyển câu hỏi thành `QuerySpec` (JSON có schema).** Ví dụ:
  `{"metric": "revenue", "periods": [{"expr": "tháng này"}], "compare": null, "filters": {}, "answer_type": "number"}`.
  Prompt kèm few-shot lấy từ **các doanh nghiệp không nằm trong tập đang đo**.
- **Định dạng đáp án:** theo đúng bảng format §9. Tiền là số nguyên, làm tròn nửa lên. Phần trăm làm tròn 2 chữ số.
- **Quy tắc bỏ qua bản POC:**
  - `metric` nằm ngoài thư viện → `abstain`;
  - QuerySpec không qua kiểm tra schema → `abstain`;
  - hai lần parse (khác few-shot) cho hai QuerySpec khác nhau → `abstain`.

### P7. Mốc so sánh, các lần chạy A/B, báo cáo quyết định (0,5 ngày)
- **Mốc "LLM viết pandas thẳng":** đưa thẻ mô tả bảng, README, đoạn rulebook và câu hỏi cho LLM. LLM viết code pandas, chạy trong sandbox, sửa tối đa 2 lần nếu lỗi.
- **Chạy trên cùng nhóm câu POC:**
  - **A:** pipeline với mapping chuẩn → trần điểm của kiến trúc;
  - **B:** pipeline với mapping do LLM gán → điểm thật ở model 4B;
  - **C:** mốc "pandas thẳng";
  - **B-14B:** chạy riêng bước gán cột và hiểu câu hỏi bằng model 14B trên 5090 thuê (vài giờ).
- **Viết `docs/poc_report.md`:** bảng PA / độ phủ / độ chính xác cho A, B, C, B-14B; độ chính xác mapping; thời gian; top 10 kiểu lỗi.

### Tiêu chí quyết định ngày 31/10

| Câu hỏi | Ngưỡng để đi tiếp theo kiến trúc này | Nếu không đạt |
|---|---|---|
| A: trên số câu đã trả lời, đúng bao nhiêu? | ≥ 90% | Sửa thư viện số đo / schema chuẩn trước khi mở rộng |
| B hoặc B-14B: mapping đúng cho bao nhiêu doanh nghiệp? | ≥ 4/5 với 14B | Đổi sang heuristic đọc header + LLM, hoặc nghiêng về hướng C |
| PA của B-14B so với C | B-14B > C rõ ràng | Hướng C đơn giản hơn: đầu tư vào C kèm kiểm chứng |
| Thời gian | Đủ để ước tính 500 câu trong 6 giờ trên 2 × 5090 | Giảm số lần gọi LLM mỗi câu |

---

## 6. End-to-end (01–16/11) — phác thảo, viết lại sau 31/10

| Stage | Nội dung | Ngày |
|---|---|---|
| E1 | Mở rộng schema chuẩn: khách hàng, thanh toán và công nợ, phiếu nhập, tồn kho, lô và hạn dùng, giá vốn trên dòng bán, chi phí, bảng giá, nhân viên, giao hàng. Áp cho cả 9 họ định dạng, 30 doanh nghiệp | 01–04/11 |
| E2 | **Rulebook → config bằng LLM**, có schema và kiểm tra. Khái niệm nằm ngoài schema thì giữ nguyên đoạn văn để đưa cho đường dự phòng. Test: sinh config từ rulebook public phải nhận ra được các luật mới | 03–05/11 |
| E3 | **Đường dự phòng: LLM viết SQL** trên các bảng chuẩn, kèm đoạn rulebook liên quan và thẻ schema. Chạy trong sandbox, có giới hạn thời gian | 04–07/11 |
| E4 | Mở rộng thư viện số đo cho các chủ đề lớn tiếp theo: khách hàng, công nợ, nhập hàng, bán chạy, tồn kho, lãi gộp | 05–09/11 |
| E5 | **Docker:** base image theo hướng dẫn của ban tổ chức, `predict.sh`, chạy với mạng tắt, ghi log từng câu vào `/result/logs/`, kiểm tra tính tất định (chạy 2 lần, so PA). **Bản chạy được đầu tiên trước 05/11**, test trên 5090 thuê | 04–06/11, rồi cập nhật liên tục |
| E6 | **Kiểm chứng và chọn ngưỡng bỏ qua:** so thư viện với SQL; tự nhất quán giữa nhiều lần sinh; chọn ngưỡng cho PA cao nhất trên tập validation chia theo doanh nghiệp | 08–11/11 |
| E7 | Chạy model 14B trên toàn bộ tập validation, nộp public leaderboard, sửa theo nhóm lỗi | 09–13/11 |
| E8 | Đóng băng (14/11), tập dượt nộp, **nộp ngày 15/11** | 14–15/11 |

**Thứ tự ưu tiên theo số câu:** doanh thu và đơn hàng → khách hàng → công nợ → nhập hàng / tồn kho → phần đuôi dài. Phần đuôi dài giao cho đường SQL dự phòng và cho quy tắc bỏ qua.

---

## 7. Rủi ro

| Rủi ro | Giảm thiểu |
|---|---|
| Họ định dạng mới ở bộ private | Bước gán cột dựa trên README và thẻ mô tả, không dựa vào tên file; giữ riêng một họ định dạng khi đo |
| Luật mới ở rulebook private | Config sinh lúc chạy, có đường SQL dự phòng, quy tắc bỏ qua |
| Model 4B không đại diện cho 14B | Kết luận quan trọng phải đo trên 14B thuê |
| Không tái lập được (lệch > 0,03) | temperature 0, seed cố định, cache, chạy Docker 2 lần so PA trước khi nộp |
| Quá 6 giờ | Đo thời gian mỗi câu từ POC; ghi `submission.jsonl` dần sau mỗi câu; cắt bớt số lần gọi LLM |
| Docker hỏng sát hạn | Bản chạy được trước 05/11; nộp sớm ngày 15/11 |
| Rò rỉ dữ liệu test | Không train hay tinh chỉnh trên public test; few-shot chỉ lấy từ train |

---

## 8. Cách làm với Cursor

- Mỗi stage làm trên một nhánh. Mỗi prompt một lượt Agent. Đọc diff trước khi accept, test xanh rồi mới commit.
- **Phần lõi tự viết hoặc tự soát kỹ:** bộ chấm (P1), bộ parse (P3), tính khoảng thời gian (P5), thư viện số đo (P5), quy tắc bỏ qua (P6). Đây là những chỗ sai một dòng là mất điểm hàng loạt.
- **Không để Cursor đọc thư mục `evals/gold_mappings/` khi viết code lời giải**, để tránh vô tình viết cứng theo doanh nghiệp.
- **Điểm review:** sau P1, sau P4, sau P7 (báo cáo POC).
