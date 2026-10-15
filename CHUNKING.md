# LegalIR Preprocessing V8 — Hướng dẫn sử dụng

- `preprocess_legal_json.py`: tiền xử lý và chunking các file `context_*.json`.
- `check_preprocessing.py`: kiểm tra tính toàn vẹn của dữ liệu sau chunking so với context gốc.

---

## 0. Lệnh nhanh

### JSON — batch nhỏ/debug

```powershell
py preprocess_legal_json.py -i "H:\...\Nhom_2" -o "H:\...\Nhom_2_processed"
py check_preprocessing.py -i "H:\...\Nhom_2" -o "H:\...\Nhom_2_processed"
```

### JSONL — corpus lớn

```powershell
py preprocess_legal_json.py -i "H:\...\selected-contexts" -o "H:\...\selected-contexts_processed" --output-mode jsonl
py check_preprocessing.py -i "H:\...\selected-contexts" -o "H:\...\selected-contexts_processed"
```

---

## 1. Cấu hình chunking mặc định

Các ngưỡng ổn định được đặt trực tiếp trong `Config` của `preprocess_legal_json.py`, vì vậy khi chạy bình thường không cần truyền lại trên command line.

```python
@dataclass(frozen=True)
class Config:
    max_words: int = 1200
    hard_max_words: int = 1500
    short_context_words: int = 300
```

|         Cấu hình     | Mặc định | Chú thích                                                                                                 |
|----------------------|----------|-----------------------------------------------------------------------------------------------------------|
| `max_words`          |  `1200`  | Giới hạn thông thường khi gom các unit cấu trúc liên tiếp thành chunk.                                    |
| `hard_max_words`     |  `1500`  | Hard cap cuối khi compact cục bộ; đồng thời là ngưỡng xác định một Điều là quá dài và cần split nội bộ.   |
| `short_context_words`|  `300`   | Vùng `non_article` ngắn nằm sát một vùng Điều có thể được gắn vào chunk Điều nếu sau khi gắn vẫn không vượt hard cap. |

Ngoài ba giá trị trên, pipeline mặc định có các nguyên tắc cố định:

- **Không giới hạn cứng số Điều/chunk.** Số Điều được quyết định bởi legal context và kích thước chunk.
- **Không overlap.** Nội dung không được cố ý lặp giữa hai chunk.
- **Điều ≤ 1500 từ là đơn vị nguyên tử**, không cắt nhỏ chỉ để đạt kích thước 1200.
- **Điều > 1500 từ** mới được split nội bộ theo cấu trúc nhỏ hơn.
- Không deduplicate paragraph theo nội dung, vì hai đoạn giống nhau vẫn có thể là hai occurrence hợp lệ trong source.

> `word` ở các ngưỡng trên là đơn vị đếm bằng regex của chương trình để điều khiển chunking/audit, **không phải token của BGE, LLM hay tokenizer BM25**.

---

# 2. Cách chạy preprocessing

Di chuyển PowerShell tới thư mục chứa hai file source:

```powershell
cd "H:\Competition\UIT_Hackathon\task_1_LegaIIR\Source code\Preprocessing\v8"
```

## 2.1. Corpus nhỏ / đang debug → dùng JSON

Đây là mode mặc định. Mỗi logical chunk được lưu thành **một file `.json` riêng**, thuận tiện mở bằng mắt để kiểm tra.

Phù hợp khi:

- đang phát triển/debug thuật toán;
- chỉ kiểm tra một vài context hoặc một batch nhỏ;
- tổng số chunk chưa quá lớn và số file vật lý chưa gây bất tiện.

Lệnh:

```powershell
py preprocess_legal_json.py -i "H:\...\Nhom_2" -o "H:\...\Nhom_2_processed" --output-mode json
```

Cấu trúc output:

```text
Nhom_2_processed_v8/
├── chunks/
│   ├── context_21_chunk_0001_....json
│   ├── context_21_chunk_0002_....json
│   └── ...
└── errors/
    └── context_....json
```

---

## 2.2. Corpus lớn → dùng JSONL

```powershell
py preprocess_legal_json.py -i "H:\...\selected-contexts" -o "H:\...\selected-contexts_processed" --output-mode jsonl
```

Mặc định:

Cấu trúc output:

```text
selected-contexts_processed/
├── chunks/
│   ├── chunks_0001.jsonl
│   ├── chunks_0002.jsonl
│   ├── chunks_0003.jsonl
│   └── ...
└── errors/
    └── context_....json
```

JSONL chỉ giảm **số file vật lý**. Nó không làm giảm số logical chunk và không thay đổi nội dung chunk.

---

## 3. Overwrite

Mặc định chương trình **xóa output cũ trước khi chạy**:

Nếu muốn từ chối xóa output cũ:

```powershell
--no-overwrite
```

---

# 4. Khái quát cách `preprocess_legal_json.py` xử lý dữ liệu

Pipeline chính:

```text
context_*.json
      │
      ▼
1. Discover source files
      │
      ▼
2. Load + validate source
      │
      ▼
3. Normalize passage
      │
      ▼
4. Scan toàn bộ passage theo thứ tự
      │
      ▼
5. Chia thành Article / Non-Article segments
      │
      ├───────────────────────┐
      ▼                       ▼
6A. ARTICLE                6B. NON-ARTICLE
    │                         │
    ├─ Điều thường            ├─ cấu trúc/heading
    │  giữ nguyên             ├─ paragraph
    │                         ├─ sentence
    ├─ gộp Điều liên tiếp     └─ word-boundary fallback
    │  cùng context
    │
    └─ Điều >1500
       split nội bộ
      │                       │
      └───────────┬───────────┘
                  ▼
7. Pack + compact cục bộ
   max thường 1200
   hard cap 1500
                  │
                  ▼
8. Gắn short context <=300 khi phù hợp
                  │
                  ▼
9. Tạo schema chunk chung + content
                  │
                  ▼
10. Ghi JSON hoặc JSONL
                  │
                  ▼
11. In thống kê trên console
```

### 4.1. Normalize passage

Chương trình chuẩn hóa:

- Unicode NFC;
- `CRLF/CR` về `LF`;
- khoảng trắng/tab trong dòng;
- các dòng trống liên tiếp.

Mục đích là tạo representation ổn định để chunk và validator dùng cùng chuẩn.

### 4.2. Segment-level scanning

Toàn passage được quét tuần tự và mọi vùng đều được gán vào:

```text
article
```

hoặc:

```text
non_article
```

Nhờ vậy phần mở đầu, căn cứ, chữ ký, phụ lục, bảng, danh mục... không bị bỏ mất chỉ vì cùng file có các Điều.

### 4.3. Article chunking

- Nhận diện heading `Điều` thật, kể cả dạng hậu tố như `Điều 21a`, `Điều 21bis`.
- Tránh một số false-positive là câu tham chiếu tới Điều khác.
- Điều ≤ `1500` từ giữ nguyên.
- Các Điều liên tiếp cùng Phần/Chương có thể được gom.
- Không giới hạn cứng số Điều/chunk.
- Điều > `1500` được split theo cấu trúc nhỏ hơn để tránh outlier quá dài.

### 4.4. Non-article chunking

Thứ tự fallback:

```text
cấu trúc/heading
    ↓
paragraph
    ↓
sentence
    ↓
word boundary
```

Word-boundary chỉ được dùng khi các đơn vị tự nhiên vẫn quá dài.

### 4.5. Không overlap

Hai chunk kế nhau không cố ý chứa phần nội dung lặp. Điều này:

- giảm near-duplicate;
- giảm kích thước corpus;
- tránh tăng term frequency giả khi lập BM25;
- giúp kiểm tra `missing/extra token` rõ ràng.

---

# 5. Cách chạy checker

Checker tự nhận dạng output đang là JSON hay JSONL. Lệnh kiểm tra **giống nhau cho cả hai mode**.

## 5.1. Kiểm tra output JSON

```powershell
py check_preprocessing.py -i "H:\...\Nhom_2" -o "H:\...\Nhom_2_processed"
```

## 5.2. Kiểm tra output JSONL

```powershell
py check_preprocessing.py -i "H:\...\selected-contexts" -o "H:\...\selected-contexts_processed"
```

Checker:

1. ưu tiên tìm `output/chunks/`;
2. nếu có `.json` thì đọc mode JSON;
3. nếu không có `.json` thì đọc `.jsonl` streaming từng dòng;
4. không load toàn bộ JSONL vào RAM một lần.

**Không nên trộn cả chunk JSON và JSONL trong cùng một thư mục `chunks/`.** Nếu cả hai cùng tồn tại, checker ưu tiên JSON để tránh đếm trùng logical chunk.
---

# 6. Luồng xử lý của `check_preprocessing.py`

```text
CONTEXT GỐC                       CHUNKS
    │                                │
    ▼                                ▼
1. Discover sources            2. Detect JSON/JSONL
    │                                │
    ▼                                ▼
3. Validate source             4. Read chunk records
    │                                │
    └──────────────┬─────────────────┘
                   ▼
5. Validate schema + source mapping
                   │
                   ▼
6. Tách metadata header khỏi content
                   │
                   ▼
7. Sắp chunk theo chunk_id/index
                   │
                   ▼
8. Tokenize source và chunk bằng cùng chuẩn
                   │
        ┌──────────┼───────────┐
        ▼          ▼           ▼
   Missing      Extra      Sequence
    token       token       mismatch
        └──────────┼───────────┘
                   ▼
9. Thống kê độ dài chunk
                   │
                   ▼
10. PASS / FAIL trên console
```

Checker **không tạo file report**.

Các thông số quan trọng:

- `total_normalized_source_tokens`: tổng token audit của toàn bộ source sau normalization.
- `total_chunk_content_tokens`: tổng token audit trong payload các chunk.
- `missing_token_occurrences`: số occurrence có trong source nhưng bị thiếu ở chunk.
- `extra_token_occurrences`: số occurrence có trong chunk nhưng không có đủ ở source; thường chỉ dấu duplicate/overlap/rewrite.
- `sequence_mismatch_sources`: số source mà sequence token ghép từ chunk không giống source.
- `loss_rate`: `missing_token_occurrences / total_normalized_source_tokens`.

Kết quả lý tưởng:

```text
Pipeline integrity                 : PASS
Ready for BM25                    : YES

total_normalized_source_tokens     = total_chunk_content_tokens
missing_token_occurrences          = 0
extra_token_occurrences            = 0
sequence_mismatch_sources          = 0
loss_rate                          = 0%
```

---

# 7. Ý nghĩa thống kê P50 / P90 / P95

Ví dụ:

```text
P50 = 1061
P90 = 1222
P95 = 1367
max = 1500
```

Hiểu là:

- `P50`: 50% chunk có độ dài ≤ 1061 từ.
- `P90`: 90% chunk có độ dài ≤ 1222 từ.
- `P95`: 95% chunk có độ dài ≤ 1367 từ.
- `max`: chunk dài nhất là 1500 từ.

Các thống kê này dùng để quan sát distribution chunk trước khi xây retrieval.

