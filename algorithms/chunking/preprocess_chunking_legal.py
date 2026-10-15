#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SEGMENT-LEVEL HYBRID LEGAL PREPROCESSING V8 - UIT DSC 2026 TASK 1
===============================================================

MỤC ĐÍCH
--------
Chỉ thực hiện TIỀN XỬ LÝ + CHUNKING cho các file context_*.json.
KHÔNG xây BM25 index, KHÔNG tính BM25 score, KHÔNG truy vấn.

NGUYÊN TẮC CHUNKING
-------------------
1. Quét tuần tự toàn bộ passage, không quyết định cả document chỉ vì thấy một dòng "Điều".
2. Mọi phần của passage đều được đưa vào một trong hai loại segment:
   - article: một Điều pháp luật thực sự;
   - non_article: mở đầu, chữ ký, phụ lục, hướng dẫn, bảng/mục... không thuộc Điều.
3. Điều bình thường là đơn vị nguyên tử:
   - KHÔNG cắt nhỏ Điều <= hard_max_words (mặc định 1500 từ);
   - có thể gộp nhiều Điều LIÊN TIẾP, cùng Phần/Chương; mặc định không giới hạn cứng số Điều;
   - max_words=1200 là cap thông thường, hard_max_words=1500 là cap cuối khi compact cục bộ.
4. Điều > hard_max_words:
   - mặc định được cắt nội bộ theo Khoản/Điểm/đoạn/câu để tránh chunk cực dài;
   - metadata vẫn chỉ rõ tất cả subchunk thuộc cùng Điều.
5. non_article:
   - chunk theo cấu trúc -> paragraph -> sentence -> word-boundary fallback;
   - dùng cùng dải kích thước với article chunk để BM25 về sau ít lệch do document length.
6. Không overlap mặc định. Điều này giảm file, tránh near-duplicate và giúp audit token-loss rõ ràng.
7. Splitter không dùng deduplicate theo nội dung, vì hai paragraph giống nhau vẫn là hai nội dung hợp lệ.
8. Heading Điều được giữ NGUYÊN VĂN. Parser hỗ trợ Điều 21bis, 21a, 21b... và tránh một số
   false-positive kiểu "Điều 7 của CISPR ...".

FORMAT OUTPUT
-------------
Mọi chunk chỉ lưu schema chung:
  chunk_id, source, document_section, chapter, article_range,
  articles, article_count, content

Trên ổ đĩa chỉ tạo:
  output/chunks/*.json                     (mode json)
  hoặc output/chunks/chunks_*.jsonl        (mode jsonl)
  output/errors/*.json

Không tạo report/manifest/summary hay thống kê preprocessing. Khi hoàn tất chỉ in một dòng thông báo.

Python >= 3.10. Chỉ dùng thư viện chuẩn.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


# =============================================================================
# 1. REGEX / HẰNG SỐ
# =============================================================================

# Dùng để đếm kích thước chunk và audit, KHÔNG phải tokenizer BM25.
WORD_RE = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+", re.UNICODE)

# Hỗ trợ: Điều 1, Điều 12a, Điều 21bis, Điều 12ABC...
# Không ép dấu chấm để vẫn xử lý được tài liệu có format thiếu dấu.
ARTICLE_LINE_RE = re.compile(
    r"^\s*Điều\s+(?P<number>\d+(?:[A-Za-z]+)?)"
    r"(?P<separator>\s*[.:)]?)\s*(?P<title>.*)$",
    re.IGNORECASE,
)

# Trường hợp heading bị xuống dòng: "Điều" / dòng kế tiếp "20. Tiêu đề".
SPLIT_ARTICLE_NUMBER_RE = re.compile(
    r"^\s*(?P<number>\d+(?:[A-Za-z]+)?)(?P<separator>[.:)]?)\s*(?P<title>.*)$",
    re.IGNORECASE,
)

CHAPTER_RE = re.compile(
    r"^\s*Chương\s+(?P<number>[IVXLCDM]+|\d+)\s*(?P<separator>[.\-:]?)\s*(?P<title>.*)$",
    re.IGNORECASE,
)

NUMBERED_SECTION_RE = re.compile(
    r"^\s*(?P<label>PHẦN\s+(?:[IVXLCDM]+|\d+)|MỤC\s+(?:[IVXLCDM]+|\d+))"
    r"\s*[.\-:]?\s*(?P<title>.*)$",
    re.IGNORECASE,
)

FIXED_SECTION_RE = re.compile(
    r"^\s*(?P<label>QUYẾT\s+ĐỊNH|QUY\s+CHẾ|PHỤ\s+LỤC)\s*[.\-:]?\s*$",
    re.IGNORECASE,
)

# Các heading lớn có khả năng đánh dấu phần tài liệu kèm theo / vùng cấu trúc mới.
# Chỉ coi là major break khi dòng ngắn và có hình thức heading.
MAJOR_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"PHỤ\s+LỤC|HƯỚNG\s+DẪN|DANH\s+MỤC|KẾ\s+HOẠCH|QUY\s+CHẾ|"
    r"QUY\s+ĐỊNH|THÔNG\s+BÁO|MẪU\s+SỐ|BIỂU\s+MẪU|NỘI\s+DUNG\s+KÈM\s+THEO|"
    r"TÀI\s+LIỆU\s+KÈM\s+THEO|NƠI\s+NHẬN"
    r")\b",
    re.IGNORECASE,
)

# Khoản/Điểm/mục đánh số để chia Điều rất dài hoặc non-article.
SUBUNIT_RE = re.compile(
    r"^\s*(?:"
    r"\d+(?:\.\d+){0,5}[.)]?"      # 1. / 1) / 1.2 / 1.2.3
    r"|[a-zA-ZđĐ][.)]"              # a) / b. / đ)
    r"|[IVXLCDM]+[.)]"              # I. / II)
    r")\s+\S+",
    re.IGNORECASE,
)

GENERIC_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"PHẦN\b|MỤC\b|CHƯƠNG\b|PHỤ\s+LỤC\b|"
    r"QUYẾT\s+ĐỊNH\b|THÔNG\s+BÁO\b|NGHỊ\s+QUYẾT\b|"
    r"CHỈ\s+THỊ\b|CÔNG\s+VĂN\b|HƯỚNG\s+DẪN\b|"
    r"QUY\s+CHẾ\b|QUY\s+ĐỊNH\b|KẾ\s+HOẠCH\b|DANH\s+MỤC\b"
    r")",
    re.IGNORECASE,
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+(?=[A-ZÀ-ỸĐ0-9])")

# Nếu "Điều <số>" không có dấu chấm/hai chấm và ngay sau là các từ tham chiếu này,
# coi đây là câu nhắc tới một Điều khác chứ không phải heading Điều.
ARTICLE_REFERENCE_PREFIX_RE = re.compile(
    r"^(?:của|tại|theo|nêu\s+tại|quy\s+định\s+tại|được\s+quy\s+định|"
    r"và|hoặc|trong|thuộc|này\b)",
    re.IGNORECASE,
)


# =============================================================================
# 2. DATA CLASS
# =============================================================================

@dataclass(frozen=True)
class Config:
    """Ba ngưỡng thực sự tham gia quyết định chunk.

    - max_words: cap thông thường khi pack unit;
    - hard_max_words: hard cap cuối, đồng thời là ngưỡng bắt đầu split Điều dài;
    - short_context_words: non-article ngắn có thể gắn vào vùng Điều kế cận.

    Không overlap và không giới hạn cứng số Điều.
    """

    max_words: int = 1200
    hard_max_words: int = 1500
    short_context_words: int = 300

    def validate(self) -> None:
        if self.max_words <= 0:
            raise ValueError("max_words phải > 0")
        if self.hard_max_words < self.max_words:
            raise ValueError("hard_max_words phải >= max_words")
        if self.short_context_words < 0:
            raise ValueError("short_context_words phải >= 0")


@dataclass(frozen=True)
class LineInfo:
    """Một dòng trong normalized passage với offset ký tự."""

    start: int
    end: int
    text: str


@dataclass(frozen=True)
class ArticleMarker:
    """Heading Điều hợp lệ đã nhận diện."""

    start: int
    number_value: int | str
    document_section: str
    chapter_number: int | str | None
    chapter_title: str | None


@dataclass(frozen=True)
class SourceSegment:
    """Một vùng liên tục của passage. Các segment phủ toàn bộ passage theo thứ tự."""

    kind: str  # article | non_article
    text: str
    document_section: str
    chapter_number: int | str | None
    chapter_title: str | None
    article_number_value: int | str | None = None


@dataclass(frozen=True)
class ChunkCandidate:
    """Trạng thái nội bộ tối thiểu cần để tạo schema chunk cuối."""

    body: str
    document_section: str
    chapter_number: int | str | None
    chapter_title: str | None
    articles: tuple[int | str, ...]


# =============================================================================
# 3. NORMALIZATION / TOKEN COUNT
# =============================================================================

def word_tokens(text: str) -> list[str]:
    return WORD_RE.findall(text)


def word_count(text: str) -> int:
    return len(word_tokens(text))


def normalize_line(line: str) -> str:
    line = (
        line.replace("\u00a0", " ")
        .replace("\u200b", "")
        .replace("\ufeff", "")
        .replace("\u2028", "\n")
        .replace("\u2029", "\n")
    )
    return re.sub(r"[ \t]+", " ", line).strip()


def normalize_passage(text: str) -> str:
    """Chuẩn hóa hình thức nhưng không viết lại heading/nội dung."""
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    output: list[str] = []
    previous_blank = True

    for raw_line in text.split("\n"):
        # Sau NFC, mỗi raw_line vẫn chỉ là một dòng; xử lý thêm unicode line separator.
        pieces = raw_line.replace("\u2028", "\n").replace("\u2029", "\n").split("\n")
        for piece in pieces:
            line = normalize_line(piece)
            if not line:
                if output and not previous_blank:
                    output.append("")
                previous_blank = True
                continue
            output.append(line)
            previous_blank = False

    return "\n".join(output).strip()


# =============================================================================
# 4. PARSER HỖ TRỢ
# =============================================================================

def roman_to_int(value: str) -> int:
    table = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    previous = 0
    for char in reversed(value.upper()):
        current = table[char]
        if current < previous:
            total -= current
        else:
            total += current
            previous = current
    return total


def parse_number(value: str) -> int | str:
    value = value.strip()
    if value.isdigit():
        return int(value)
    if re.fullmatch(r"[IVXLCDM]+", value, flags=re.IGNORECASE):
        return roman_to_int(value)
    # Giữ nguyên suffix kiểu 21bis / 21a để không làm mất thông tin.
    return value


def normalize_section_label(label: str) -> str:
    compact = re.sub(r"\s+", " ", label.strip())
    upper = compact.upper()
    if upper == "QUYẾT ĐỊNH":
        return "Quyết định"
    if upper == "QUY CHẾ":
        return "Quy chế"
    if upper == "PHỤ LỤC":
        return "Phụ lục"
    if upper.startswith("PHẦN "):
        return "Phần " + compact.split(maxsplit=1)[1]
    if upper.startswith("MỤC "):
        return "Mục " + compact.split(maxsplit=1)[1]
    return compact


def chapter_title_from_line(number: int | str, inline_title: str) -> str:
    inline_title = inline_title.strip(" .-:")
    return f"Chương {number}. {inline_title}" if inline_title else f"Chương {number}."


def line_infos(text: str) -> list[LineInfo]:
    """Tạo line offsets trên chính normalized passage."""
    infos: list[LineInfo] = []
    cursor = 0
    for piece in text.splitlines(keepends=True):
        content = piece[:-1] if piece.endswith("\n") else piece
        infos.append(LineInfo(start=cursor, end=cursor + len(content), text=content))
        cursor += len(piece)

    # splitlines() trả [] khi chuỗi rỗng; pipeline không xử lý passage rỗng.
    if not infos and text:
        infos.append(LineInfo(start=0, end=len(text), text=text))
    return infos


def is_short_all_caps_heading(line: str) -> bool:
    tokens = word_tokens(line)
    if not tokens or len(tokens) > 24:
        return False
    letters = [ch for ch in line if ch.isalpha()]
    if len(letters) < 3:
        return False
    upper_ratio = sum(ch.isupper() for ch in letters) / len(letters)
    return upper_ratio >= 0.84


def match_article_heading(line: str) -> re.Match[str] | None:
    """Nhận diện heading Điều, có heuristic loại tham chiếu nội dung.

    Sửa lỗi 21bis:
      - number cho phép nhiều chữ cái suffix, không chỉ 1 ký tự;
      - heading output luôn giữ nguyên line nguồn, không tái tạo "Điều 21b. is".

    Tránh false-positive phổ biến:
      "Điều 7 của CISPR 16-2-1:2008" không có separator và title bắt đầu bằng "của".
    """
    stripped = line.strip()
    match = ARTICLE_LINE_RE.match(stripped)
    if not match:
        return None

    separator = match.group("separator").strip()
    title = match.group("title").strip()

    # Có dấu câu ngay sau số Điều: độ tin cậy cao.
    if separator in {".", ":", ")"}:
        return match

    # Heading chỉ gồm "Điều 21bis" vẫn hợp lệ.
    if not title:
        return match

    # Không có separator và phần sau rõ ràng là câu tham chiếu -> loại.
    if ARTICLE_REFERENCE_PREFIX_RE.match(title):
        return None

    # Heading pháp luật thường mở đầu bằng chữ hoa/số/ký hiệu mục.
    first_alpha = next((ch for ch in title if ch.isalpha()), "")
    if first_alpha and first_alpha.islower():
        return None

    return match


def match_chapter_heading(line: str) -> re.Match[str] | None:
    stripped = line.strip()
    match = CHAPTER_RE.match(stripped)
    if not match:
        return None
    separator = match.group("separator").strip()
    title = match.group("title").strip()
    if separator:
        return match
    if not title:
        return match
    if ARTICLE_REFERENCE_PREFIX_RE.match(title):
        return None
    first_alpha = next((ch for ch in title if ch.isalpha()), "")
    if first_alpha and first_alpha.islower():
        return None
    return match


def is_major_break_line(line: str) -> bool:
    """Boundary đủ mạnh để kết thúc một article region trước EOF/Điều kế tiếp."""
    stripped = line.strip()
    if not stripped:
        return False
    if match_chapter_heading(stripped):
        return True
    if FIXED_SECTION_RE.match(stripped) or NUMBERED_SECTION_RE.match(stripped):
        return True
    if MAJOR_HEADING_RE.match(stripped):
        # Hạn chế false-positive trong đoạn văn: chỉ heading ngắn / IN HOA / Nơi nhận.
        return len(word_tokens(stripped)) <= 24 and (
            is_short_all_caps_heading(stripped)
            or stripped.lower().startswith("nơi nhận")
            or stripped.upper().startswith("PHỤ LỤC")
        )
    return False


def is_structure_start(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return bool(
        match_article_heading(stripped)
        or match_chapter_heading(stripped)
        or NUMBERED_SECTION_RE.match(stripped)
        or FIXED_SECTION_RE.match(stripped)
        or GENERIC_HEADING_RE.match(stripped)
        or SUBUNIT_RE.match(stripped)
        or is_short_all_caps_heading(stripped)
    )


# =============================================================================
# 5. SEGMENT-LEVEL SCAN: PHỦ TOÀN BỘ PASSAGE
# =============================================================================

def scan_article_markers(passage: str) -> tuple[list[ArticleMarker], set[int]]:
    """Quét tuần tự line-by-line, thu article marker + major boundary positions.

    State Phần/Chương được cập nhật trong lúc quét nên metadata của Điều không phụ
    thuộc việc cả document có phải văn bản dạng Điều hay không.
    """
    infos = line_infos(passage)
    article_markers: list[ArticleMarker] = []
    major_break_positions: set[int] = set()

    # O(n): vị trí dòng không-rỗng kế tiếp. Tránh quét infos[i+1:] lặp lại
    # trên các tài liệu cực lớn.
    next_nonblank: list[int | None] = [None] * len(infos)
    next_idx: int | None = None
    for j in range(len(infos) - 1, -1, -1):
        next_nonblank[j] = next_idx
        if infos[j].text.strip():
            next_idx = j

    section = "Văn bản"
    chapter_number: int | str | None = None
    chapter_title: str | None = None

    for i, info in enumerate(infos):
        stripped = info.text.strip()
        if not stripped:
            continue

        # Heading bị tách dòng: "Điều" ở một dòng, số + title ở dòng không-rỗng kế tiếp.
        if stripped.casefold() == "điều":
            next_i = next_nonblank[i]
            next_info = infos[next_i] if next_i is not None else None
            if next_info is not None:
                split_match = SPLIT_ARTICLE_NUMBER_RE.match(next_info.text.strip())
                if split_match:
                    separator = split_match.group("separator").strip()
                    title = split_match.group("title").strip()
                    # Áp dụng heuristic tương tự heading một dòng.
                    looks_reference = (
                        not separator
                        and title
                        and ARTICLE_REFERENCE_PREFIX_RE.match(title) is not None
                    )
                    first_alpha = next((ch for ch in title if ch.isalpha()), "")
                    bad_lower_title = bool(
                        not separator and title and first_alpha and first_alpha.islower()
                    )
                    if not looks_reference and not bad_lower_title:
                        number_raw = split_match.group("number").strip()
                        article_markers.append(
                            ArticleMarker(
                                start=info.start,
                                number_value=parse_number(number_raw),
                                document_section=section,
                                chapter_number=chapter_number,
                                chapter_title=chapter_title,
                            )
                        )
                        continue

        fixed = FIXED_SECTION_RE.match(stripped)
        numbered = NUMBERED_SECTION_RE.match(stripped)
        if fixed or numbered:
            match = fixed or numbered
            assert match is not None
            section = normalize_section_label(match.group("label"))
            chapter_number = None
            chapter_title = None
            major_break_positions.add(info.start)
            continue

        chapter = match_chapter_heading(stripped)
        if chapter:
            chapter_number = parse_number(chapter.group("number"))
            inline_title = chapter.group("title").strip()

            # Nếu "Chương 2." không có title inline, lấy 1 dòng IN HOA ngắn kế tiếp
            # làm metadata, nhưng KHÔNG xóa dòng đó khỏi passage.
            if not inline_title and i + 1 < len(infos):
                next_line = infos[i + 1].text.strip()
                if next_line and is_short_all_caps_heading(next_line) and not match_article_heading(next_line):
                    inline_title = next_line

            chapter_title = chapter_title_from_line(chapter_number, inline_title)
            major_break_positions.add(info.start)
            continue

        article_match = match_article_heading(stripped)
        if article_match:
            number_raw = article_match.group("number").strip()
            article_markers.append(
                ArticleMarker(
                    start=info.start,
                    number_value=parse_number(number_raw),
                    document_section=section,
                    chapter_number=chapter_number,
                    chapter_title=chapter_title,
                )
            )
            continue

        if is_major_break_line(stripped):
            major_break_positions.add(info.start)

    return article_markers, major_break_positions


def segment_passage(passage: str) -> list[SourceSegment]:
    """Chia passage thành article/non_article segment và bảo toàn toàn bộ văn bản.

    Cách làm dùng character spans. Vì vậy mọi ký tự trong normalized passage thuộc
    đúng một vùng trước khi chunking; không còn vấn đề bỏ phần attachment chỉ vì
    document có một vài Điều ở đầu.
    """
    markers, major_breaks = scan_article_markers(passage)
    if not markers:
        return [
            SourceSegment(
                kind="non_article",
                text=passage,
                document_section=infer_section_at_start(passage),
                chapter_number=None,
                chapter_title=None,
            )
        ]

    # Tính range cho từng Điều: đến Điều kế tiếp, trừ khi có major break ở giữa.
    article_ranges: list[tuple[int, int, ArticleMarker]] = []
    text_len = len(passage)

    for idx, marker in enumerate(markers):
        next_article_start = markers[idx + 1].start if idx + 1 < len(markers) else text_len
        end = next_article_start

        # Major break nằm sau heading Điều và trước Điều kế tiếp sẽ tách phần đó
        # sang non_article segment. Boundary tại chính start của heading lớn.
        between = [pos for pos in major_breaks if marker.start < pos < next_article_start]
        if between:
            end = min(between)

        # Bảo vệ trường hợp heading ngay sau boundary bất thường.
        if end <= marker.start:
            end = next_article_start

        article_ranges.append((marker.start, end, marker))

    # Loại range bị chồng do một article marker xuất hiện sau phần đã bị major-break
    # của Điều trước. Range article sau vẫn độc lập và phần gap sẽ thành non_article.
    segments: list[SourceSegment] = []
    cursor = 0
    for start, end, marker in article_ranges:
        if start < cursor:
            # Có thể xảy ra nếu một marker false-positive nằm trong vùng đã đóng;
            # bỏ marker chồng để không duplicate text.
            continue

        if cursor < start:
            gap = passage[cursor:start].strip()
            if gap:
                sec, ch_num, ch_title = infer_context_before_offset(passage, cursor)
                segments.append(
                    SourceSegment(
                        kind="non_article",
                        text=gap,
                        document_section=sec,
                        chapter_number=ch_num,
                        chapter_title=ch_title,
                    )
                )

        article_text = passage[start:end].strip()
        if article_text:
            segments.append(
                SourceSegment(
                    kind="article",
                    text=article_text,
                    document_section=marker.document_section,
                    chapter_number=marker.chapter_number,
                    chapter_title=marker.chapter_title,
                    article_number_value=marker.number_value,
                )
            )
        cursor = max(cursor, end)

    if cursor < text_len:
        tail = passage[cursor:].strip()
        if tail:
            sec, ch_num, ch_title = infer_context_before_offset(passage, cursor)
            segments.append(
                SourceSegment(
                    kind="non_article",
                    text=tail,
                    document_section=sec,
                    chapter_number=ch_num,
                    chapter_title=ch_title,
                )
            )

    return segments


def infer_section_at_start(passage: str) -> str:
    for line in passage.split("\n")[:50]:
        stripped = line.strip()
        fixed = FIXED_SECTION_RE.match(stripped)
        numbered = NUMBERED_SECTION_RE.match(stripped)
        if fixed or numbered:
            match = fixed or numbered
            assert match is not None
            return normalize_section_label(match.group("label"))
    return "Văn bản"


def infer_context_before_offset(passage: str, offset: int) -> tuple[str, int | str | None, str | None]:
    """Suy ra state Phần/Chương tại vị trí gap; dùng scan nhỏ, ưu tiên correctness."""
    section = "Văn bản"
    chapter_number: int | str | None = None
    chapter_title: str | None = None
    prefix = passage[:offset]
    lines = prefix.split("\n")

    for i, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped:
            continue
        fixed = FIXED_SECTION_RE.match(stripped)
        numbered = NUMBERED_SECTION_RE.match(stripped)
        if fixed or numbered:
            match = fixed or numbered
            assert match is not None
            section = normalize_section_label(match.group("label"))
            chapter_number = None
            chapter_title = None
            continue
        chapter = match_chapter_heading(stripped)
        if chapter:
            chapter_number = parse_number(chapter.group("number"))
            inline_title = chapter.group("title").strip()
            chapter_title = chapter_title_from_line(chapter_number, inline_title)
    return section, chapter_number, chapter_title


# =============================================================================
# 6. LOSSLESS SPLITTER / PACKER
# =============================================================================

def split_paragraphs_lossless(text: str) -> list[str]:
    """Tách paragraph trên normalized text. Không dedup paragraph giống nhau."""
    text = text.strip()
    if not text:
        return []
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def split_sentences_lossless(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    pieces = [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]
    return pieces or [text]


def split_by_word_boundaries_lossless(text: str, max_words: int) -> list[str]:
    """Fallback cắt tại offset token, giữ nguyên punctuation/text substring."""
    text = text.strip()
    matches = list(WORD_RE.finditer(text))
    if len(matches) <= max_words:
        return [text] if text else []

    pieces: list[str] = []
    total = len(matches)
    start_token = 0
    while start_token < total:
        end_token = min(start_token + max_words, total)
        start_char = matches[start_token].start()
        end_char = matches[end_token].start() if end_token < total else len(text)
        piece = text[start_char:end_char].strip()
        if piece:
            pieces.append(piece)
        start_token = end_token
    return pieces


def extract_structural_units(text: str) -> list[str]:
    """Tách theo paragraph/heading; không deduplicate unit giống nhau."""
    paragraphs = split_paragraphs_lossless(text)
    if not paragraphs:
        return []

    units: list[str] = []
    for paragraph in paragraphs:
        current: list[str] = []

        def flush() -> None:
            if current:
                unit = "\n".join(current).strip()
                if unit:
                    units.append(unit)
                current.clear()

        for line in paragraph.split("\n"):
            stripped = line.strip()
            if is_structure_start(stripped) and current:
                flush()
            current.append(stripped)
        flush()
    return units


def ensure_units_within_max_lossless(
    units: Sequence[str],
    max_words: int,
) -> list[str]:
    """Bảo đảm từng unit <= max theo line -> câu -> word-boundary."""
    output: list[str] = []

    for unit in units:
        if word_count(unit) <= max_words:
            output.append(unit)
            continue

        subparts = [part.strip() for part in unit.split("\n") if part.strip()]
        if len(subparts) > 1 and all(word_count(part) <= max_words for part in subparts):
            output.extend(subparts)
            continue

        sentences = split_sentences_lossless(unit)
        if len(sentences) > 1:
            for sentence in sentences:
                if word_count(sentence) <= max_words:
                    output.append(sentence)
                else:
                    output.extend(split_by_word_boundaries_lossless(sentence, max_words))
            continue

        output.extend(split_by_word_boundaries_lossless(unit, max_words))

    return output


def _merge_adjacent_groups_lossless(
    groups: list[list[str]],
    hard_max_words: int,
) -> list[list[str]]:
    """Compact cục bộ các group kề nhau nếu tổng không vượt hard cap."""
    if len(groups) < 2:
        return groups

    output: list[list[str]] = []
    current = list(groups[0])
    for nxt in groups[1:]:
        combined = current + list(nxt)
        if word_count("\n\n".join(combined)) <= hard_max_words:
            current = combined
        else:
            output.append(current)
            current = list(nxt)
    output.append(current)
    return output


def pack_units_lossless(
    units: Sequence[str],
    max_words: int,
    hard_max_words: int | None = None,
) -> list[str]:
    """Pack tuần tự dưới max_words rồi compact cục bộ dưới hard_max_words."""
    cleaned = [unit.strip() for unit in units if unit and unit.strip()]
    if not cleaned:
        return []

    hard_max = hard_max_words if hard_max_words is not None else max_words
    if hard_max < max_words:
        raise ValueError("hard_max_words phải >= max_words")

    groups: list[list[str]] = []
    current: list[str] = []
    current_words = 0

    for unit in cleaned:
        wc = word_count(unit)
        if wc > max_words:
            raise ValueError("pack_units_lossless nhận unit vượt max_words")

        if current and current_words + wc > max_words:
            groups.append(current)
            current = []
            current_words = 0

        current.append(unit)
        current_words += wc

    if current:
        groups.append(current)

    groups = _merge_adjacent_groups_lossless(groups, hard_max)
    return ["\n\n".join(group).strip() for group in groups if group]


# =============================================================================
# 7. CHUNK ARTICLE SEGMENTS
# =============================================================================

def split_long_article_segment(segment: SourceSegment, config: Config) -> list[ChunkCandidate]:
    """Split Điều vượt hard_max_words theo cấu trúc; Điều thường luôn nguyên vẹn."""
    text = segment.text
    if "\n" in text:
        first_line, body = text.split("\n", 1)
    else:
        first_line, body = text, ""

    first_line = first_line.strip()
    body = body.strip()
    if not body:
        return [
            ChunkCandidate(
                body=text,
                document_section=segment.document_section,
                chapter_number=segment.chapter_number,
                chapter_title=segment.chapter_title,
                articles=(segment.article_number_value,),
            )
        ]

    units = extract_structural_units(body)
    safe_units = ensure_units_within_max_lossless(units, config.max_words)
    bodies = pack_units_lossless(safe_units, config.max_words)

    # Heading gốc chỉ nằm ở part đầu để toàn bộ payload vẫn là một partition lossless.
    if bodies:
        bodies[0] = f"{first_line}\n{bodies[0]}".strip()
    else:
        bodies = [text]

    return [
        ChunkCandidate(
            body=piece,
            document_section=segment.document_section,
            chapter_number=segment.chapter_number,
            chapter_title=segment.chapter_title,
            articles=(segment.article_number_value,),
        )
        for piece in bodies
    ]


def _article_group_candidate(group: Sequence[SourceSegment]) -> ChunkCandidate:
    first = group[0]
    return ChunkCandidate(
        body="\n\n".join(seg.text for seg in group).strip(),
        document_section=first.document_section,
        chapter_number=first.chapter_number,
        chapter_title=first.chapter_title,
        articles=tuple(seg.article_number_value for seg in group),
    )


def _compact_article_groups_local(
    groups: list[list[SourceSegment]],
    hard_max_words: int,
) -> list[list[SourceSegment]]:
    """Compact group Điều kề nhau nhưng không vượt Phần/Chương hoặc hard cap."""
    if len(groups) < 2:
        return groups

    output: list[list[SourceSegment]] = []
    current = list(groups[0])

    for nxt in groups[1:]:
        if not current or not nxt:
            continue
        first = current[0]
        next_first = nxt[0]
        same_context = (
            first.document_section == next_first.document_section
            and first.chapter_number == next_first.chapter_number
            and first.chapter_title == next_first.chapter_title
        )
        combined_words = word_count("\n\n".join(seg.text for seg in current + nxt))

        if same_context and combined_words <= hard_max_words:
            current.extend(nxt)
        else:
            output.append(current)
            current = list(nxt)

    output.append(current)
    return output


def build_article_run_candidates(
    run: Sequence[SourceSegment],
    config: Config,
) -> list[ChunkCandidate]:
    """Gom Điều tuần tự; chỉ split Điều riêng lẻ khi > hard_max_words."""
    candidates: list[ChunkCandidate] = []
    normal_groups: list[list[SourceSegment]] = []
    current: list[SourceSegment] = []
    current_words = 0

    def flush_current() -> None:
        nonlocal current, current_words
        if current:
            normal_groups.append(current)
            current = []
            current_words = 0

    def flush_normal_groups() -> None:
        nonlocal normal_groups
        if not normal_groups:
            return
        for group in _compact_article_groups_local(normal_groups, config.hard_max_words):
            candidates.append(_article_group_candidate(group))
        normal_groups = []

    for seg in run:
        wc = word_count(seg.text)

        if wc > config.hard_max_words:
            flush_current()
            flush_normal_groups()
            candidates.extend(split_long_article_segment(seg, config))
            continue

        if not current:
            current = [seg]
            current_words = wc
            continue

        same_boundary = (
            seg.document_section == current[0].document_section
            and seg.chapter_number == current[0].chapter_number
            and seg.chapter_title == current[0].chapter_title
        )
        would_exceed = current_words + wc > config.max_words

        if not same_boundary or would_exceed:
            flush_current()
            if not same_boundary:
                flush_normal_groups()
            current = [seg]
            current_words = wc
        else:
            current.append(seg)
            current_words += wc

    flush_current()
    flush_normal_groups()
    return candidates


# =============================================================================
# 8. CHUNK NON-ARTICLE SEGMENTS
# =============================================================================

def build_non_article_segment_candidates(
    segment: SourceSegment,
    config: Config,
) -> list[ChunkCandidate]:
    units = extract_structural_units(segment.text)
    safe_units = ensure_units_within_max_lossless(units, config.max_words)
    bodies = pack_units_lossless(
        safe_units,
        max_words=config.max_words,
        hard_max_words=config.hard_max_words,
    )

    return [
        ChunkCandidate(
            body=body,
            document_section=segment.document_section,
            chapter_number=segment.chapter_number,
            chapter_title=segment.chapter_title,
            articles=(),
        )
        for body in bodies
    ]


def _prepend_short_context(
    context_seg: SourceSegment,
    candidate: ChunkCandidate,
    hard_max_words: int,
) -> ChunkCandidate | None:
    combined_body = f"{context_seg.text}\n\n{candidate.body}".strip()
    if word_count(combined_body) > hard_max_words:
        return None
    return ChunkCandidate(
        body=combined_body,
        document_section=candidate.document_section,
        chapter_number=candidate.chapter_number,
        chapter_title=candidate.chapter_title,
        articles=candidate.articles,
    )


def _append_short_context(
    candidate: ChunkCandidate,
    context_seg: SourceSegment,
    hard_max_words: int,
) -> ChunkCandidate | None:
    if context_seg.document_section != candidate.document_section:
        return None
    if context_seg.chapter_number != candidate.chapter_number:
        return None
    combined_body = f"{candidate.body}\n\n{context_seg.text}".strip()
    if word_count(combined_body) > hard_max_words:
        return None
    return ChunkCandidate(
        body=combined_body,
        document_section=candidate.document_section,
        chapter_number=candidate.chapter_number,
        chapter_title=candidate.chapter_title,
        articles=candidate.articles,
    )


def build_candidates_from_segments(
    segments: Sequence[SourceSegment],
    config: Config,
) -> list[ChunkCandidate]:
    """Chunk tuần tự toàn passage; context ngắn dùng chung một quy tắc prefix/suffix."""
    output: list[ChunkCandidate] = []
    i = 0

    while i < len(segments):
        seg = segments[i]

        if (
            seg.kind == "non_article"
            and word_count(seg.text) <= config.short_context_words
            and i + 1 < len(segments)
            and segments[i + 1].kind == "article"
        ):
            context_seg = seg
            i += 1
            run: list[SourceSegment] = []
            while i < len(segments) and segments[i].kind == "article":
                run.append(segments[i])
                i += 1

            run_candidates = build_article_run_candidates(run, config)
            if run_candidates:
                merged = _prepend_short_context(
                    context_seg, run_candidates[0], config.hard_max_words
                )
                if merged is not None:
                    run_candidates[0] = merged
                else:
                    output.extend(build_non_article_segment_candidates(context_seg, config))

            if (
                run_candidates
                and i < len(segments)
                and segments[i].kind == "non_article"
                and word_count(segments[i].text) <= config.short_context_words
                and not (i + 1 < len(segments) and segments[i + 1].kind == "article")
            ):
                merged = _append_short_context(
                    run_candidates[-1], segments[i], config.hard_max_words
                )
                if merged is not None:
                    run_candidates[-1] = merged
                    i += 1

            output.extend(run_candidates)
            continue

        if seg.kind == "non_article":
            output.extend(build_non_article_segment_candidates(seg, config))
            i += 1
            continue

        run: list[SourceSegment] = []
        while i < len(segments) and segments[i].kind == "article":
            run.append(segments[i])
            i += 1
        run_candidates = build_article_run_candidates(run, config)

        if (
            run_candidates
            and i < len(segments)
            and segments[i].kind == "non_article"
            and word_count(segments[i].text) <= config.short_context_words
            and not (i + 1 < len(segments) and segments[i + 1].kind == "article")
        ):
            merged = _append_short_context(
                run_candidates[-1], segments[i], config.hard_max_words
            )
            if merged is not None:
                run_candidates[-1] = merged
                i += 1

        output.extend(run_candidates)

    return output


# =============================================================================
# 9. OUTPUT JSON - SCHEMA CHUNG, TỐI GIẢN
# =============================================================================

def ascii_slug(text: Any) -> str:
    value = str(text).replace("Đ", "D").replace("đ", "d")
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return value or "unknown"


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class JsonlShardWriter:
    """Ghi chunk theo JSONL shard để tránh tạo quá nhiều file nhỏ.

    Mỗi dòng là đúng một JSON object theo cùng schema với mode JSON.
    Writer chạy streaming nên không cần giữ toàn bộ chunk trong RAM.
    """

    def __init__(self, chunks_dir: Path, shard_size: int = 5000) -> None:
        if shard_size <= 0:
            raise ValueError("jsonl_shard_size phải > 0")
        self.chunks_dir = chunks_dir
        self.shard_size = shard_size
        self.shard_index = 0
        self.records_in_shard = 0
        self._handle = None

    def _open_next(self) -> None:
        if self._handle is not None:
            self._handle.close()
        self.shard_index += 1
        self.records_in_shard = 0
        path = self.chunks_dir / f"chunks_{self.shard_index:04d}.jsonl"
        self._handle = path.open("w", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        if self._handle is None or self.records_in_shard >= self.shard_size:
            self._open_next()
        assert self._handle is not None
        self._handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        self._handle.write("\n")
        self.records_in_shard += 1

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def build_content(
    source_name: str,
    section: str,
    chapter_title: str | None,
    body: str,
) -> str:
    """Tạo trường `content` theo schema chung.

    Chỉ các dòng header metadata ở đầu là nội dung tổng hợp:
      - Tên nguồn
      - Phần văn bản
      - Chương (nếu có)

    Sau dòng trống là nội dung được lấy từ normalized passage.
    Không lặp heading Điều ở các part sau của Điều dài, nhờ đó checker generic
    có thể khôi phục chính xác nội dung nguồn từ `content`.
    """
    header = [f"Tên nguồn: {source_name}", f"Phần văn bản: {section}"]
    if chapter_title:
        header.append(f"Chương: {chapter_title}")
    return "\n".join(header) + "\n\n" + body.strip()


def build_chunk_record(
    source: dict[str, Any],
    source_file: str,
    chunk_id: str,
    candidate: ChunkCandidate,
) -> dict[str, Any]:
    """Chuyển candidate nội bộ sang schema chunk chung.

    Schema cố ý chỉ giữ 8 trường đã thống nhất để mọi thuật toán chunking khác
    có thể tạo cùng format và dùng chung check_preprocessing.py.
    """
    article_list = list(candidate.articles)
    chapter = (
        {"number": candidate.chapter_number, "title": candidate.chapter_title}
        if candidate.chapter_number is not None
        else None
    )

    return {
        "chunk_id": chunk_id,
        "source": {
            "id": source["id"],
            "name": source["name"],
            "link": source["link"],
            "file": source_file,
        },
        "document_section": candidate.document_section,
        "chapter": chapter,
        "article_range": {
            "start": article_list[0] if article_list else None,
            "end": article_list[-1] if article_list else None,
        },
        "articles": article_list,
        "article_count": len(article_list),
        "content": build_content(
            source_name=source["name"],
            section=candidate.document_section,
            chapter_title=candidate.chapter_title,
            body=candidate.body,
        ),
    }


def chunk_filename(record: dict[str, Any]) -> str:
    """Tên file duy nhất trên toàn corpus, nhưng vẫn dễ đọc bằng mắt."""
    parts = [record["chunk_id"], ascii_slug(record["document_section"])]
    chapter = record.get("chapter")
    if isinstance(chapter, dict) and chapter.get("number") is not None:
        parts.extend(["chuong", str(chapter["number"])])

    articles = record.get("articles") or []
    if articles:
        parts.extend(["dieu", str(articles[0])])
        if len(articles) > 1:
            parts.append(str(articles[-1]))
    else:
        parts.append("ngoai_dieu")

    return "_".join(ascii_slug(p) for p in parts if str(p)) + ".json"


def save_error_source(source_path: Path, input_dir: Path, errors_dir: Path) -> Path:
    """Chỉ copy file nguồn lỗi; KHÔNG tạo reason.json hay report phụ."""
    try:
        relative = source_path.relative_to(input_dir)
    except ValueError:
        relative = Path(source_path.name)
    destination = errors_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    return destination


# =============================================================================
# 10. LOAD / DISCOVER
# =============================================================================

def load_source(path: Path) -> dict[str, Any]:
    """Đọc context gốc.

    Bắt buộc: `id`, `passage` không rỗng.
    `name` có thể thiếu trong một số TCVN; khi đó suy ra từ URL hoặc context_<id>.
    """
    from urllib.parse import urlparse

    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("JSON root phải là object")
    if "id" not in data:
        raise ValueError("Thiếu trường bắt buộc: id")
    if "passage" not in data:
        raise ValueError("Thiếu trường bắt buộc: passage")
    if not isinstance(data["passage"], str) or not data["passage"].strip():
        raise ValueError("passage phải là chuỗi không rỗng")

    link = data.get("link") if isinstance(data.get("link"), str) else ""
    name = data.get("name") if isinstance(data.get("name"), str) else ""
    if not name.strip():
        candidate = Path(urlparse(link).path).stem if link else ""
        name = candidate or f"context_{data['id']}"

    return {**data, "name": name.strip(), "link": link}


def discover_files(input_dir: Path, pattern: str = "context_*.json") -> list[Path]:
    """Tìm source đệ quy để không phụ thuộc corpus có thêm một thư mục con hay không."""
    return sorted(p for p in input_dir.rglob(pattern) if p.is_file())



# =============================================================================
# 11. PIPELINE CHÍNH
# =============================================================================

def preprocess(args: argparse.Namespace) -> None:
    """Đọc context, tạo chunk và lưu output; không thu thập thống kê báo cáo."""
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    chunks_dir = output_dir / "chunks"
    errors_dir = output_dir / "errors"

    config = Config()
    config.validate()

    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)

    # Chỉ hai loại dữ liệu được lưu:
    #   output/chunks/ -> chunk JSON riêng lẻ hoặc JSONL shard
    #   output/errors/ -> nguyên bản source không xử lý được
    chunks_dir.mkdir(parents=True, exist_ok=True)
    errors_dir.mkdir(parents=True, exist_ok=True)

    jsonl_writer = (
        JsonlShardWriter(chunks_dir, args.jsonl_shard_size)
        if args.output_mode == "jsonl"
        else None
    )

    seen_ids: set[str] = set()

    try:
        for source_path in discover_files(input_dir, args.pattern):
            relative = str(source_path.relative_to(input_dir))
            try:
                source = load_source(source_path)
                sid = str(source["id"])

                if sid in seen_ids:
                    raise ValueError(f"source.id bị trùng: {sid}")
                seen_ids.add(sid)

                passage = normalize_passage(source["passage"])
                segments = segment_passage(passage)
                if not segments:
                    raise ValueError("Không tạo được segment nào")

                candidates = build_candidates_from_segments(segments, config)
                if not candidates:
                    raise ValueError("Không tạo được chunk candidate")

                for chunk_index, candidate in enumerate(candidates, start=1):
                    chunk_id = f"context_{sid}_chunk_{chunk_index:04d}"
                    record = build_chunk_record(source, relative, chunk_id, candidate)
                    if jsonl_writer is not None:
                        jsonl_writer.write(record)
                    else:
                        save_json(record, chunks_dir / chunk_filename(record))

            except Exception:
                save_error_source(source_path, input_dir, errors_dir)
    finally:
        if jsonl_writer is not None:
            jsonl_writer.close()


# =============================================================================
# 12. CLI
# =============================================================================

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Legal preprocessing V8: chỉ chunk dữ liệu và lưu chunks/errors; "
            "không sinh báo cáo preprocessing."
        )
    )
    p.add_argument("-i", "--input-dir", required=True, help="Thư mục context gốc")
    p.add_argument("-o", "--output-dir", required=True, help="Thư mục output")
    p.add_argument("--pattern", default="context_*.json")
    p.add_argument(
        "--output-mode",
        choices=("json", "jsonl"),
        default="json",
        help=(
            "json: mỗi chunk một file (mặc định, dễ kiểm tra); "
            "jsonl: gom chunk thành shard cho corpus lớn."
        ),
    )
    p.add_argument(
        "--jsonl-shard-size",
        type=int,
        default=5000,
        help="Số chunk mỗi file JSONL khi --output-mode jsonl (mặc định 5000).",
    )
    p.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mặc định xóa output cũ trước khi chạy.",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    preprocess(args)
    print("Hoàn tất preprocessing.")


if __name__ == "__main__":
    main()
