param(
    [string]$Public = "data\test\public-official.json",
    [string]$IndexDir = "indexes\legalir_bge_m3_json",
    [string]$Output = "output\submission.json"
)

python -m legal_ir.main submit `
  --public "$Public" `
  --index-dir "$IndexDir" `
  --output "$Output" `
  --bm25-top-k 200 `
  --dense-top-k 200 `
  --rrf-k 60 `
  --bm25-weight 1.0 `
  --dense-weight 1.0 `
  --top-k-documents 5 `
  --query-batch-size 8 `
  --zip
