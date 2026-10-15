# UIT DATA SCIENCE CHALLENGE 2026 (TASK 1 - LEGALIR)

> **Cuộc thi:** UIT Data Science Challenge 2026  
> **Tác vụ:** Task 1 – LegalIR (Truy vấn văn bản pháp luật tiếng Việt)  
> **Dữ liệu:** ~8.500 văn bản hành chính/quy phạm pháp luật  
> **Kết quả thực tế đạt được:** 
>   | `Recall: 0.8323` (83.23% — Độ đo xếp hạng chính) 
>   | `Precision: 0.1788` (17.88% — Độ đo phụ)  
> **Ràng buộc:** Trả về tối đa 05 `document_id` cho mỗi câu hỏi.

## 1. PHÂN TÍCH KIẾN TRÚC THUẬT TOÁN CỦA DỰ ÁN

## 1.1. Hệ thống phân cấp thư mục algorithms

```text
Thư mục algorithms/: Hệ thống phân cấp
├── chunking/       # preprocessing/chunking đã có
├── embedding/      # BGE-M3
├── retrieval/      # BM25 + Dense + Weighted RRF
├── reranking/      # giữ chỗ, chưa dùng baseline
└── generation/     # tạo submission.json
```

Kiến trúc dự án được thiết kế theo mô hình **Hybrid Retrieval Pipeline** chuyên biệt cho văn bản pháp lý với tính mô-đun hóa cao:

```text
               [Kho dữ liệu ~8.500 Văn bản hành chính]
                                  │
                                  ▼
             [Legal-Specific Chunking Engine (V8)]
             - Giữ nguyên Điều/Khoản/Điểm (Atomic units)
             - Giới hạn: max_words 1200, hard_cap 1500
             - Bảo toàn 100% nội dung (Zero token loss)
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
       [Custom BM25 Index]              [BGE-M3 Dense Index]
       - Tokenizer nhận diện            - BAAI/bge-m3 (8.192 max tokens)
         mã số: 12/2023/NĐ-CP           - Streaming Memmap + FP16
       - Tách ký tự /, -, .             - Tối ưu VRAM trên RTX 3060
                  │                               │
             Top 200                         Top 200
                  └───────────────┬───────────────┘
                                  ▼
                   [Weighted RRF (Reciprocal Rank Fusion)]
                   - Tham số k = 60, weights = [1.0, 1.0]
                   - Loại bỏ chênh lệch phân phối điểm số
                                  │
                                  ▼
                   [Chunk-to-Document Aggregation]
                   - Gom cụm chunk theo source_id
                   - Max-pooling / Best-rank selection
                                  │
                                  ▼
                   [Top 5 Document IDs / Submission]
```

### 1.2. Tầng Tiền xử lý & Chunking (Điểm sáng kỹ thuật vượt trội)
* Khác biệt hoàn toàn với cách chunking cố định độ dài cơ bản (Fixed-size chunking như cắt mỗi 512 từ có overlap), dự án xây dựng bộ parser phân tích cú pháp phân cấp pháp luật Việt Nam (`algorithms/chunking/preprocess_chunking_legal.py`).
* **Đơn vị nguyên tử (Atomic Unit):** Coi toàn bộ một "Điều" là một đơn vị ngữ nghĩa hoàn chỉnh ($\le 1500$ từ không bị cắt xé). Chỉ split nội bộ theo Khoản/Điểm nếu Điều vượt ngưỡng quá dài. Điều này đảm bảo trọn vẹn ngữ cảnh pháp lý (chủ thể + điều kiện + chế tài).
* **Xử lý số hiệu pháp quy phức tạp:** Bắt chuẩn các dạng `Điều 21bis`, `Điều 12a`, xử lý ngắt dòng `Điều` / `20. Tiêu đề` và loại trừ các bẫy false-positive (ví dụ: *"theo Điều 7 của CISPR..."*).
* **Audit Pipeline:** Tích hợp kiểm định toàn vẹn dữ liệu, đảm bảo không bị thất thoát token (zero-loss guarantee).

### 1.3. Tầng Truy vấn lai (Hybrid Retrieval)
* **Sparse Search (BM25 tùy biến):** 
  * Tự cài đặt BM25 với cấu trúc Inverted Index tối ưu bằng NumPy.
  * Bộ Tokenizer đặc chế bằng Regular Expression để tách chuẩn các ký tự số hiệu văn bản (`/`, `-`, `.`), giải quyết triệt để vấn đề truy vấn theo số hiệu nghị định/thông tư mà các mô hình ngôn ngữ lớn hay bỏ sót.
* **Dense Search (BGE-M3 Embedding):**
  * Sử dụng mô hình SOTA BAAI/bge-m3 hỗ trợ context length lên tới **8.192 tokens**, mã hóa trọn vẹn cả các chunk dài mà không bị truncate ngữ nghĩa.
* **Tối ưu hóa tài nguyên phần cứng (Low-Resource Engineering):**
  * Tối ưu hóa để chạy mượt mà trên **GPU đơn dân dụng NVIDIA RTX 3060 (12GB VRAM)**.
  * Ứng dụng **NumPy Memory-Mapped file (`open_memmap`)** và **Streaming Corpus** để ghi trực tiếp vector embedding xuống ổ đĩa, xử lý hàng chục nghìn chunk mà không gây quá tải RAM hệ thống.
  * Tự động kích hoạt FP16 Half-precision khi có CUDA, cân đối batch size (batch 4 cho indexing, batch 8 cho inference).

### 1.4. Tầng Hợp nhất thứ hạng & Gom cụm (Fusion & Aggregation)
* **Weighted Reciprocal Rank Fusion (RRF, $k=60$):**
  * BM25 cho điểm số không giới hạn trên, trong khi Dense Retriever trả về Cosine Similarity trong khoảng $[-1, 1]$. Việc cộng điểm tuyến tính (Score fusion) sẽ làm lệch kết quả.
  * Dự án sử dụng RRF dựa trên thứ hạng (Rank-based), hoàn toàn miễn nhiễm với sự khác biệt về phân phối điểm số của hai bộ tìm kiếm.
* **Chunk-to-Document Aggregation:**
  * Tìm kiếm diễn ra ở tầng Chunk (độ đặc hiệu cao), nhưng đề bài yêu cầu nộp ID văn bản cha. Thuật toán tổng hợp điểm số (Max-pooling/Best-rank) gom cụm các chunk có cùng `source_id`, đảm bảo chọn ra đúng 5 văn bản có độ liên quan cao nhất mà không bị trùng lặp.

## 2. CÁC YẾU TỐ CỐT LÕI VÀ BÀI HỌC KINH NGHIỆM RÚT RA

1. **"Data Chunking quyết định trần chất lượng của hệ thống IR/RAG":**
   * Nếu chunking cắt đôi một Điều luật (làm tách rời điều kiện áp dụng và mức phạt), mô hình embedding tối tân đến đâu cũng không thể cứu vãn. Chunking theo cấu trúc tài liệu miền chuyên sâu (Domain-specific hierarchical chunking) chính là yếu tố tạo ra bước nhảy vọt về Recall.
2. **"Sparse và Dense là hai thành phần tương hỗ hoàn hảo":**
   * Người dùng tra cứu pháp luật thường có 2 nhóm hành vi:
     * Nhớ rõ từ khóa/số hiệu văn bản $\rightarrow$ BM25 chiếm ưu thế tuyệt đối.
     * Mô tả tình huống đời sống bằng ngôn ngữ tự nhiên $\rightarrow$ Dense Embedding bắt được ý nghĩa ngữ nghĩa.
   * Sự kết hợp của cả hai tạo nên độ phủ toàn diện cho hệ thống.
3. **"Tư duy tối ưu theo luật chơi (Metric-driven Optimization)":**
   * Nắm bắt chính xác bản chất của độ đo: Recall là thước đo chính, Precision là phụ; mỗi câu hỏi cho phép nộp tối đa 5 ID. Do đó, việc luôn trả về đủ Top 5 là chiến lược tối ưu nhất để cực đại hóa Recall.
4. **"Kỹ thuật tối ưu hóa trên phần cứng giới hạn":**
   * Trong thực tế doanh nghiệp, việc tối ưu hệ thống chạy trên phần cứng thương mại (1 GPU 12GB) thay vì đòi hỏi cụm máy chủ A100/H100 chứng tỏ năng lực kỹ thuật thực chiến rất cao (thông qua `memmap`, `streaming`, `fp16`, `vectorization`).
5. **"Tính ổn định và toàn vẹn dữ liệu (Production-readiness)":**
   * Sử dụng mã băm SHA256 để liên kết và kiểm tra tính nhất quán giữa file corpus thô, index BM25 và file npy embedding, ngăn ngừa hoàn toàn lỗi lệch chỉ mục (index mismatch).

## 3. KIẾN THỨC HỌC ĐƯỢC THÔNG QUA CUỘC THI

### 3.1. Kiến thức & Kỹ năng kỹ thuật cứng (Hard Skills):
* **Xử lý ngôn ngữ tự nhiên chuyên sâu (NLP & Text Processing):** Xây dựng parser phân tích tài liệu hành chính phức tạp dựa trên regex và state machine.
* **Information Retrieval (IR) & Search Engine:** Tự triển khai Inverted Index cho BM25; cấu hình và tinh chỉnh Dense Embedding (BGE-M3).
* **Kỹ thuật RAG & Ranking:** Làm chủ thuật toán Reciprocal Rank Fusion (RRF), Chunk-to-Document score aggregation.
* **Tối ưu hóa hiệu năng và bộ nhớ:** Sử dụng `numpy.memmap`, generators/streaming IO, mixed-precision FP16, batching trên PyTorch.
* **Đánh giá hệ thống IR:** Nắm vững và phân tích chuyên sâu các chỉ số Information Retrieval: Recall@K, Precision@K, MRR, NDCG.

### 3.2. Tư duy kỹ sư phần mềm & AI (Engineering Mindset):
* Kiến trúc hệ thống dạng mô-đun (Clean Architecture): tách biệt rõ rệt các tầng `chunking`, `embedding`, `retrieval`, `reranking`, `generation`.
* Tư duy thực nghiệm: Thiết lập quy trình đánh giá offline (`evaluate.py` trên tập train) trước khi áp dụng cho tập test.