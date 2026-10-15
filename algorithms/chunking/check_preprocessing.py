#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GENERIC LEGALIR CHUNK VALIDATOR - CONSOLE ONLY
==============================================

Dùng chung cho mọi thuật toán chunking nếu output JSON hoặc JSONL có schema chung:
  chunk_id
  source {id, name, link, file}
  document_section
  chapter
  article_range
  articles
  article_count
  content

Validator không biết/chấm thuật toán chunking. Nó chỉ kiểm tra:
- source/chunk mapping và schema;
- metadata cơ bản;
- coverage của source;
- token preservation (missing/extra);
- thứ tự token;
- phân bố độ dài chunk.

Không tạo report JSON. Kết quả chỉ in trên console.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import preprocess_chunking_legal as prep


REQUIRED_FIELDS = {
    "chunk_id",
    "source",
    "document_section",
    "chapter",
    "article_range",
    "articles",
    "article_count",
    "content",
}
SOURCE_FIELDS = {"id", "name", "link", "file"}
CHUNK_INDEX_RE = re.compile(r"_chunk_(\d+)$", re.IGNORECASE)
HEADER_SEPARATOR_RE = re.compile(r"\n[ \t]*\n")
EXAMPLE_LIMIT = 3
INVALID_INDEX = 10**18


@dataclass
class FailureTracker:
    """Đếm đầy đủ lỗi nhưng chỉ giữ vài ví dụ để console không bị tràn."""

    counts: Counter[str] = field(default_factory=Counter)
    examples: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    def add(self, category: str, example: Any | None = None) -> None:
        self.counts[category] += 1
        if example is None or len(self.examples[category]) >= EXAMPLE_LIMIT:
            return
        text = str(example)
        if text not in self.examples[category]:
            self.examples[category].append(text)

    def has_failures(self) -> bool:
        return bool(self.counts)


@dataclass
class ValidationResult:
    integrity_passed: bool
    source_files_found: int
    valid_source_files: int
    invalid_source_files: int
    invalid_source_files_saved: int
    chunk_files_found: int
    sources_with_chunks: int
    total_source_tokens: int
    total_chunk_tokens: int
    missing_occurrences: int
    sources_with_token_loss: int
    extra_occurrences: int
    sources_with_extra_tokens: int
    sequence_mismatch_sources: int
    loss_rate: float
    chunk_stats: dict[str, float | int]
    article_chunk_stats: dict[str, float | int]
    non_article_chunk_stats: dict[str, float | int]
    failures: FailureTracker


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def discover_chunk_storage(output_dir: Path) -> tuple[str, list[Path]]:
    """Tìm chunk JSON hoặc JSONL.

    Ưu tiên output/chunks/; nếu -o trỏ thẳng vào thư mục chunk thì đọc trực tiếp.
    Không trộn hai mode trong cùng một lần validate: nếu có JSON thì dùng JSON,
    nếu không mới dùng JSONL.
    """
    chunks_dir = output_dir / "chunks"
    search_dir = chunks_dir if chunks_dir.is_dir() else output_dir

    def allowed(path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            parents = path.relative_to(search_dir).parts[:-1]
        except ValueError:
            parents = ()
        return "errors" not in {part.lower() for part in parents}

    json_files = sorted(p for p in search_dir.rglob("*.json") if allowed(p))
    if json_files:
        return "json", json_files

    jsonl_files = sorted(p for p in search_dir.rglob("*.jsonl") if allowed(p))
    return "jsonl", jsonl_files


def iter_chunk_records(mode: str, files: list[Path]):
    """Yield (record, reference). JSONL được đọc streaming từng dòng."""
    if mode == "json":
        for path in files:
            try:
                yield load_json(path), path.name, None
            except Exception as exc:
                yield None, path.name, exc
        return

    for path in files:
        try:
            with path.open("r", encoding="utf-8-sig") as f:
                for line_no, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    ref = f"{path.name}:{line_no}"
                    try:
                        yield json.loads(line), ref, None
                    except Exception as exc:
                        yield None, ref, exc
        except Exception as exc:
            yield None, path.name, exc


def percentile(values: Sequence[int], p: float) -> float:
    """Tính percentile phục vụ riêng báo cáo của checker."""
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return float(ordered[lo])
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def stat(values: Sequence[int]) -> dict[str, float | int]:
    """Tính thống kê độ dài chunk; chỉ checker sử dụng."""
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0,
        }
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def normalized_tokens(text: str) -> list[str]:
    return [
        token.casefold()
        for token in prep.word_tokens(prep.normalize_passage(text))
    ]


def extract_payload(record: dict[str, Any]) -> tuple[str, bool]:
    """Lấy phần nội dung thật sau block metadata đầu `content`.

    Không hard-code `Chương:`/`Phạm vi STT:`/... vì các chunker có thể dùng
    metadata header khác nhau. Quy ước chung chỉ là metadata nằm trước dòng trống
    đầu tiên, payload nằm sau nó.
    """
    content = record.get("content")
    if not isinstance(content, str):
        return "", False

    content = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    parts = HEADER_SEPARATOR_RE.split(content, maxsplit=1)
    if len(parts) != 2:
        return "", False

    header, payload = parts[0].strip(), parts[1].strip()
    header_lines = [line.strip() for line in header.split("\n") if line.strip()]
    # Hai header nền tảng của schema content chung phải tồn tại; các header khác
    # được chấp nhận tự do và không tham gia token audit.
    header_ok = (
        any(line.startswith("Tên nguồn:") for line in header_lines)
        and any(line.startswith("Phần văn bản:") for line in header_lines)
    )
    return payload, header_ok


def parse_chunk_index(chunk_id: str) -> int | None:
    match = CHUNK_INDEX_RE.search(chunk_id)
    return int(match.group(1)) if match else None


def validate(args: argparse.Namespace) -> ValidationResult:
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    errors_dir = output_dir / "errors"

    source_files = prep.discover_files(input_dir, args.pattern)
    chunk_mode, chunk_storage_files = discover_chunk_storage(output_dir)
    failures = FailureTracker()

    # ------------------------------------------------------------------
    # 1. Sources
    # ------------------------------------------------------------------
    sources: dict[str, dict[str, Any]] = {}
    source_relative_by_id: dict[str, str] = {}
    invalid_source_paths: list[Path] = []

    for path in source_files:
        try:
            source = prep.load_source(path)
            sid = str(source["id"])
            if sid in sources:
                failures.add("duplicate_source_ids", sid)
                continue
            sources[sid] = source
            source_relative_by_id[sid] = str(path.relative_to(input_dir))
        except Exception:
            invalid_source_paths.append(path)

    invalid_saved = 0
    for path in invalid_source_paths:
        relative = path.relative_to(input_dir)
        if (errors_dir / relative).is_file():
            invalid_saved += 1
        else:
            failures.add("invalid_sources_not_saved", relative)

    # ------------------------------------------------------------------
    # 2. Chunks + schema/metadata
    # ------------------------------------------------------------------
    chunks_by_source: dict[str, list[tuple[int, str]]] = defaultdict(list)
    seen_chunk_ids: set[str] = set()
    all_lengths: list[int] = []
    article_lengths: list[int] = []
    non_article_lengths: list[int] = []

    chunk_records_found = 0
    for record, ref, load_error in iter_chunk_records(chunk_mode, chunk_storage_files):
        if load_error is not None:
            failures.add("invalid_chunk_json", ref)
            continue
        chunk_records_found += 1

        if not isinstance(record, dict):
            failures.add("chunk_schema_errors", ref)
            continue

        if REQUIRED_FIELDS - set(record):
            failures.add("chunk_schema_errors", record.get("chunk_id", ref))
            continue

        source_meta = record.get("source")
        if not isinstance(source_meta, dict) or SOURCE_FIELDS - set(source_meta):
            failures.add("chunk_schema_errors", record.get("chunk_id", ref))
            continue

        cid = str(record.get("chunk_id"))
        if cid in seen_chunk_ids:
            failures.add("duplicate_chunk_ids", cid)
        seen_chunk_ids.add(cid)

        index = parse_chunk_index(cid)
        if index is None:
            failures.add("invalid_chunk_id_format", cid)
            index = INVALID_INDEX

        sid = str(source_meta.get("id"))
        payload, header_ok = extract_payload(record)
        if not header_ok:
            failures.add("content_header_errors", cid)
        if not payload:
            failures.add("empty_chunk_content", cid)

        wc = prep.word_count(payload)
        all_lengths.append(wc)

        articles = record.get("articles")
        if not isinstance(articles, list):
            failures.add("article_metadata_errors", cid)
            articles = []
        else:
            if record.get("article_count") != len(articles):
                failures.add("article_metadata_errors", cid)

            expected_range = (
                {"start": articles[0], "end": articles[-1]}
                if articles
                else {"start": None, "end": None}
            )
            if record.get("article_range") != expected_range:
                failures.add("article_range_errors", cid)

        (article_lengths if articles else non_article_lengths).append(wc)

        if sid not in sources:
            failures.add("orphan_chunks", f"{cid} -> source {sid}")
        else:
            source = sources[sid]
            if source_meta.get("name") != source.get("name"):
                failures.add("source_metadata_mismatch", f"{cid}: name")
            if source_meta.get("link") != source.get("link"):
                failures.add("source_metadata_mismatch", f"{cid}: link")

            expected_relative = source_relative_by_id[sid]
            stored_file = str(source_meta.get("file", ""))
            if stored_file not in {expected_relative, Path(expected_relative).name}:
                failures.add("source_metadata_mismatch", f"{cid}: file")

        chunks_by_source[sid].append((index, payload))

    # ------------------------------------------------------------------
    # 3. Coverage + token/sequence preservation
    # ------------------------------------------------------------------
    valid_source_ids = set(sources)
    chunk_source_ids = set(chunks_by_source)

    for sid in sorted(valid_source_ids - chunk_source_ids):
        failures.add("missing_source_outputs", sid)

    total_source_tokens = 0
    total_chunk_tokens = 0
    missing_occurrences = 0
    extra_occurrences = 0
    sources_with_token_loss = 0
    sources_with_extra_tokens = 0
    sequence_mismatch_sources = 0

    for sid, source in sources.items():
        source_tokens = normalized_tokens(source["passage"])
        total_source_tokens += len(source_tokens)

        ordered = sorted(chunks_by_source.get(sid, []), key=lambda item: item[0])
        actual_indices = [index for index, _ in ordered if index != INVALID_INDEX]
        if actual_indices != list(range(1, len(actual_indices) + 1)):
            failures.add("chunk_index_sequence_errors", sid)

        chunk_tokens: list[str] = []
        for _, payload in ordered:
            chunk_tokens.extend(normalized_tokens(payload))
        total_chunk_tokens += len(chunk_tokens)

        source_counter = Counter(source_tokens)
        chunk_counter = Counter(chunk_tokens)
        missing = source_counter - chunk_counter
        extra = chunk_counter - source_counter

        if missing:
            sources_with_token_loss += 1
            missing_occurrences += sum(missing.values())
            failures.add("source_token_loss", sid)

        if extra:
            sources_with_extra_tokens += 1
            extra_occurrences += sum(extra.values())
            failures.add("duplicate_or_extra_tokens", sid)

        if source_tokens != chunk_tokens:
            sequence_mismatch_sources += 1
            # Chỉ tạo lỗi riêng nếu multiset token vẫn đúng nhưng thứ tự sai.
            if not missing and not extra:
                failures.add("token_order_errors", sid)

    integrity_passed = (
        not failures.has_failures()
        and missing_occurrences == 0
        and extra_occurrences == 0
        and sequence_mismatch_sources == 0
    )

    return ValidationResult(
        integrity_passed=integrity_passed,
        source_files_found=len(source_files),
        valid_source_files=len(sources),
        invalid_source_files=len(invalid_source_paths),
        invalid_source_files_saved=invalid_saved,
        chunk_files_found=chunk_records_found,
        sources_with_chunks=len(valid_source_ids & chunk_source_ids),
        total_source_tokens=total_source_tokens,
        total_chunk_tokens=total_chunk_tokens,
        missing_occurrences=missing_occurrences,
        sources_with_token_loss=sources_with_token_loss,
        extra_occurrences=extra_occurrences,
        sources_with_extra_tokens=sources_with_extra_tokens,
        sequence_mismatch_sources=sequence_mismatch_sources,
        loss_rate=(missing_occurrences / total_source_tokens if total_source_tokens else 0.0),
        chunk_stats=stat(all_lengths),
        article_chunk_stats=stat(article_lengths),
        non_article_chunk_stats=stat(non_article_lengths),
        failures=failures,
    )


def fmt_number(value: int | float, decimals: int = 2) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.{decimals}f}"


def fmt_stats(values: dict[str, float | int]) -> str:
    return (
        f"count={fmt_number(int(values['count']))}, "
        f"mean={fmt_number(float(values['mean']))}, "
        f"P50={fmt_number(float(values['p50']))}, "
        f"P90={fmt_number(float(values['p90']))}, "
        f"P95={fmt_number(float(values['p95']))}, "
        f"max={fmt_number(int(values['max']))}"
    )


def print_result(result: ValidationResult) -> None:
    line = "=" * 72
    print(line)
    print("PREPROCESSING VALIDATION")
    print(line)
    print(f"Pipeline integrity                 : {'PASS' if result.integrity_passed else 'FAIL'}")
    print(f"Ready for BM25                    : {'YES' if result.integrity_passed else 'NO'}")

    print("\n[SOURCES]")
    print(f"source_files_found                 : {fmt_number(result.source_files_found)}")
    print(f"valid_source_files                 : {fmt_number(result.valid_source_files)}")
    print(f"invalid_source_files               : {fmt_number(result.invalid_source_files)}")
    print(f"invalid_source_files_saved         : {fmt_number(result.invalid_source_files_saved)}")

    print("\n[CHUNKS]")
    print(f"chunk_files_found                  : {fmt_number(result.chunk_files_found)}")
    print(f"sources_with_chunks                : {fmt_number(result.sources_with_chunks)}")

    print("\n[TOKEN PRESERVATION]")
    print(f"total_normalized_source_tokens     : {fmt_number(result.total_source_tokens)}")
    print(f"total_chunk_content_tokens         : {fmt_number(result.total_chunk_tokens)}")
    print(f"missing_token_occurrences          : {fmt_number(result.missing_occurrences)}")
    print(f"sources_with_token_loss            : {fmt_number(result.sources_with_token_loss)}")
    print(f"extra_token_occurrences            : {fmt_number(result.extra_occurrences)}")
    print(f"sources_with_extra_tokens          : {fmt_number(result.sources_with_extra_tokens)}")
    print(f"sequence_mismatch_sources          : {fmt_number(result.sequence_mismatch_sources)}")
    print(f"loss_rate                          : {result.loss_rate:.6%}")

    print("\n[CHUNK LENGTH - WORDS]")
    print(f"all                                : {fmt_stats(result.chunk_stats)}")
    print(f"article                            : {fmt_stats(result.article_chunk_stats)}")
    print(f"non_article                        : {fmt_stats(result.non_article_chunk_stats)}")

    print("\n[FAILURES]")
    if not result.failures.counts:
        print("none")
    else:
        width = max(len(name) for name in result.failures.counts)
        for name in sorted(result.failures.counts):
            print(f"{name:<{width}} : {fmt_number(result.failures.counts[name])}")

        print(f"\n[EXAMPLES - MAX {EXAMPLE_LIMIT} / TYPE]")
        for name in sorted(result.failures.examples):
            examples = result.failures.examples[name]
            if examples:
                print(f"{name}: {', '.join(examples)}")

    print(line)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generic validator cho mọi chunker dùng chung LegalIR chunk schema; "
            "chỉ hiển thị kết quả trên console."
        )
    )
    parser.add_argument("-i", "--input-dir", required=True, help="Thư mục context gốc")
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Output chứa chunks/ và errors/, hoặc trực tiếp thư mục chunk",
    )
    parser.add_argument("--pattern", default="context_*.json")
    return parser


def main() -> None:
    result = validate(build_parser().parse_args())
    print_result(result)


if __name__ == "__main__":
    main()
