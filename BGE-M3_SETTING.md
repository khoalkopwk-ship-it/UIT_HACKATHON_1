# UTE-Firelights - LegalIR Hybrid Retrieval

Phiên bản này giữ nguyên cấu trúc repository `UTE-Firelights` và bổ sung phần LegalIR còn thiếu:

```text
algorithms/
├── chunking/       # preprocessing/chunking đã có
├── embedding/      # BGE-M3
├── retrieval/      # BM25 + Dense + Weighted RRF
├── reranking/      # giữ chỗ, chưa dùng baseline
└── generation/     # tạo submission.json
```

Pipeline cuộc thi:

```text
chunks
   │
   ├───────────────┐
   ▼               ▼
 BM25            BGE-M3
   │               │
 Top 200         Top 200
   └───────┬───────┘
           ▼
      Weighted RRF
           ▼
     chunk -> source.id
           ▼
        Top 5
           ▼
   output/submission.json
```

Baseline dùng:

- BM25 top-k = 200
- BGE-M3 dense top-k = 200
- RRF k = 60
- BM25 weight = 1.0
- Dense weight = 1.0
- Final documents = 5
- BGE-M3 max length = 8192
- RTX 3060: FP16 tự bật khi CUDA khả dụng
- Batch BGE-M3 mặc định = 4 để phù hợp RTX 3060 12GB

## 1. Cấu trúc thư mục

```text
UTE-Firelights/
├── algorithms/
│   ├── chunking/
│   │   ├── preprocess_chunking_legal.py
│   │   └── check_preprocessing.py
│   ├── embedding/
│   │   ├── embedder.py
│   │   └── audit_bge_tokens.py
│   ├── retrieval/
│   │   ├── bm25.py
│   │   ├── dense_search.py
│   │   └── hybrid_search.py
│   ├── reranking/
│   │   └── reranker.py
│   └── generation/
│       └── rag.py
│
├── legal_ir/
│   ├── main.py
│   ├── submission.py
│   └── evaluate.py
│
├── data/
│   ├── raw/
│   ├── train/
│   │   └── train.json
│   ├── test/
│   │   └── public-official.json
│   └── processed/
│       └── chunks/
│
├── indexes/
├── output/
├── requirements.txt
├── README.md
└── .gitignore
```

`reranking/` được giữ đúng kiến trúc của project nhưng chưa gọi trong baseline. Không cần dùng reranker để chạy submission hiện tại.

## 2. Cài trên Windows + RTX 3060

Khuyến nghị Python 3.11.

PowerShell tại thư mục project:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Kiểm tra GPU:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Kết quả mong muốn:

```text
CUDA: True
GPU: NVIDIA GeForce RTX 3060
```

Nếu `CUDA: False`, BGE-M3 vẫn chạy CPU nhưng không nên dùng CPU cho lần build dense index lớn.

## 3. Dữ liệu chunks

Chunker hiện có của project tạo schema:

```json
{
  "chunk_id": "context_740_chunk_0001",
  "source": {
    "id": 740,
    "name": "...",
    "link": "...",
    "file": "..."
  },
  "content": "..."
}
```

Có thể dùng trực tiếp thư mục chunks đã preprocess.

Nếu cần chạy preprocessing:

```powershell
python -m legal_ir.main preprocess `
  --input-dir "data/raw/selected_contexts" `
  --output-dir "data/processed" `
  --output-mode json
```

Sau đó chunks nằm ở:

```text
data/processed/chunks/
```

Nếu preprocessing đã hoàn thành thì bỏ qua bước này.

## 4. Build index

Lần đầu chạy:

```powershell
python -m legal_ir.main build-index `
  --chunks "data/processed/chunks" `
  --index-dir "indexes/legalir_bge_m3_json" `
  --bge-batch-size 4 `
  --max-length 8192
```

Hệ thống tạo:

```text
indexes/legalir_bge_m3_json/
├── corpus.jsonl
├── corpus_meta.json
├── bm25_index.pkl
├── bm25_meta.json
├── dense_embeddings.npy
└── dense_meta.json
```

BGE-M3 sẽ được tải vào cache Hugging Face lần đầu. Những lần sau không cần tải lại nếu cache còn nguyên.

## 5. Tạo submission từ public-official.json

File nộp chính thức phải được sinh từ `data/test/public-official.json`, **không phải**
`data/train/train.json`.

Sau khi đã activate `.venv`, chạy nhanh bằng script:

```powershell
.\run_submission.ps1
```

Script mặc định sử dụng:

```text
Input:  data/test/public-official.json
Output: output/submission.json
ZIP:    output/submission.zip
```

Nếu PowerShell chặn script do Execution Policy, chạy lệnh Python tương đương:

```powershell
python -m legal_ir.main submit `
  --public "data/test/public-official.json" `
  --index-dir "indexes/legalir_bge_m3_json" `
  --output "output/submission.json" `
  --bm25-top-k 200 `
  --dense-top-k 200 `
  --rrf-k 60 `
  --bm25-weight 1.0 `
  --dense-weight 1.0 `
  --top-k-documents 5 `
  --query-batch-size 8 `
  --zip
```

Nếu chưa activate môi trường ảo, thay `python` ở đầu lệnh bằng
`.\.venv\Scripts\python.exe`.

Khi hoàn tất, log phải có `Processed 1000/1000 questions`, đồng thời thư mục
`output/` có cả `submission.json` và `submission.zip`.

## 6. Format submission

Mặc định phiên bản này giữ `question` cùng `answer`, phù hợp với format submission mẫu:

```json
{
  "38096": {
    "question": "Đề nghị xem xét lại quyết định...",
    "answer": [
      "277391",
      "198465",
      "170717",
      "217416",
      "217791"
    ]
  }
}
```

Mỗi query tối đa 5 document ID, không trùng.

Nếu hệ thống chấm của cuộc thi yêu cầu chỉ có `answer`, thêm:

```text
--answer-only
```

Ví dụ:

```powershell
python -m legal_ir.main submit `
  --public "data/test/public-official.json" `
  --index-dir "indexes/legalir_bge_m3_json" `
  --output "output/submission.json" `
  --answer-only `
  --zip
```

## 7. Tùy chọn: tạo prediction và kiểm tra Recall@5 trên train

Tập train chỉ dùng để đánh giá/tuning, không dùng để tạo
`output/submission.json`. Trước tiên tạo prediction riêng:

```powershell
python -m legal_ir.main submit `
  --public "data/train/train.json" `
  --index-dir "indexes/legalir_bge_m3_json" `
  --output "output/train_prediction.json" `
  --bm25-top-k 200 `
  --dense-top-k 200 `
  --rrf-k 60 `
  --bm25-weight 1.0 `
  --dense-weight 1.0 `
  --top-k-documents 5 `
  --query-batch-size 8
```

Sau đó kiểm tra Recall@5:

```powershell
python -m legal_ir.evaluate `
  --prediction "output/train_prediction.json" `
  --ground-truth "data/train/train.json"
```

Kết quả:

```text
Recall@5: 0.xxxx
```

Lưu ý: điểm `0.8813` trong submission mẫu của cuộc thi không thể được đảm bảo chỉ bằng code; nó phụ thuộc chính xác vào dataset/chunks, phiên bản preprocessing, index và tham số. Phiên bản này giữ nguyên công thức baseline BM25 + BGE-M3 + RRF để có thể tái tạo và sau đó tune.

## 8. Tuning sau baseline

Không cần build lại index khi chỉ thay đổi:

```text
bm25-top-k
dense-top-k
rrf-k
bm25-weight
dense-weight
top-k-documents
```

Có thể thử:

```text
BM25 top-k: 200 / 300 / 500
Dense top-k: 200 / 300 / 500
RRF k: 20 / 40 / 60 / 80 / 100
Dense weight: 0.8 / 1.0 / 1.2 / 1.5
BM25 weight: 0.8 / 1.0 / 1.2 / 1.5
```

Mục tiêu là tăng Recall@5 trên train trước khi tạo submission cuối.

## 9. Lưu ý RTX 3060

Mặc định:

```text
BGE-M3 batch = 4
max_length = 8192
FP16 = tự bật nếu CUDA=True
```

Nếu thiếu VRAM:

```powershell
--bge-batch-size 2
```

Nếu còn nhiều VRAM và chạy ổn:

```powershell
--bge-batch-size 6
```

Không nên tăng batch quá cao ngay từ đầu.

## 10. Kết quả cần giữ

Không cần lưu hàng loạt embedding/query/log trung gian vào GitHub.

Các artifact quan trọng:

```text
indexes/                  # local, không commit
output/train_prediction.json  # tùy chọn, chỉ dùng đánh giá trên train
output/submission.json        # 1.000 query từ public-official.json
output/submission.zip         # file dùng để nộp
```

`indexes/`, model cache và dataset lớn nên được bỏ qua bởi `.gitignore`.
