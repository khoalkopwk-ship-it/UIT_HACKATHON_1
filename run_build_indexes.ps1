param(
    [Parameter(Mandatory=$true)][string]$Chunks,
    [string]$IndexDir = "indexes\legalir_bge_m3",
    [int]$BatchSize = 4,
    [int]$MaxLength = 8192
)

python -m legal_ir.main build-index `
  --chunks "$Chunks" `
  --index-dir "$IndexDir" `
  --bge-batch-size $BatchSize `
  --max-length $MaxLength
