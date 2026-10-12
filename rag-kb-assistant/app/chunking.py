"""Structure-aware chunking.

Split on headings first, then paragraphs, then sentences. Tables and code
blocks are never cut. Each chunk keeps its section path so it can be shown in
citations and prepended to the text that gets embedded.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    position: int
    section: str
    text: str

    def embed_text(self, title: str) -> str:
        head = title if not self.section else f"{title} > {self.section}"
        return f"{head}\n{self.text}"


@dataclass
class _Block:
    kind: str  # "para" or "atomic"
    text: str


def approx_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _sections(text: str) -> list[tuple[str, list[_Block]]]:
    """Return [(section_path, blocks)] in document order."""
    sections: list[tuple[str, list[_Block]]] = []
    stack: list[tuple[int, str]] = []
    blocks: list[_Block] = []
    para: list[str] = []
    table: list[str] = []
    fence: list[str] | None = None

    def current_path() -> str:
        return " > ".join(name for _, name in stack)

    def flush_para() -> None:
        if para:
            blocks.append(_Block("para", "\n".join(para).strip()))
            para.clear()

    def flush_table() -> None:
        if table:
            blocks.append(_Block("atomic", "\n".join(table)))
            table.clear()

    def close_section(path: str) -> None:
        flush_para()
        flush_table()
        if any(b.text for b in blocks):
            sections.append((path, list(blocks)))
        blocks.clear()

    path_for_blocks = ""
    for raw in text.replace("\r\n", "\n").split("\n"):
        line = raw.rstrip()
        if fence is not None:
            fence.append(line)
            if line.strip().startswith("```"):
                blocks.append(_Block("atomic", "\n".join(fence)))
                fence = None
            continue
        if line.strip().startswith("```"):
            flush_para()
            flush_table()
            fence = [line]
            continue
        heading = _HEADING.match(line)
        if heading:
            close_section(path_for_blocks)
            level, name = len(heading.group(1)), heading.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, name))
            path_for_blocks = current_path()
            continue
        if _TABLE_ROW.match(line):
            flush_para()
            table.append(line)
            continue
        flush_table()
        if not line.strip():
            flush_para()
        else:
            para.append(line)
    if fence is not None:  # unterminated code fence
        blocks.append(_Block("atomic", "\n".join(fence)))
    close_section(path_for_blocks)
    return sections


def _split_oversized(block: _Block, limit: int) -> list[_Block]:
    if approx_tokens(block.text) <= limit:
        return [block]
    if block.kind == "atomic":
        pieces, current = [], []
        for line in block.text.split("\n"):
            current.append(line)
            if approx_tokens("\n".join(current)) >= limit:
                pieces.append("\n".join(current))
                current = []
        if current:
            pieces.append("\n".join(current))
        return [_Block("atomic", p) for p in pieces if p.strip()]
    pieces, current = [], []
    for sentence in _SENTENCE.split(block.text):
        current.append(sentence)
        if approx_tokens(" ".join(current)) >= limit:
            pieces.append(" ".join(current))
            current = []
    if current:
        pieces.append(" ".join(current))
    return [_Block("para", p) for p in pieces if p.strip()]


def chunk_document(text: str, target_tokens: int = 450, overlap_tokens: int = 50) -> list[Chunk]:
    target_tokens = max(target_tokens, 50)
    overlap_tokens = max(0, min(overlap_tokens, target_tokens // 3))
    chunks: list[Chunk] = []

    for section, blocks in _sections(text):
        flat: list[_Block] = []
        for block in blocks:
            flat.extend(_split_oversized(block, target_tokens))

        current: list[_Block] = []
        size = 0
        for block in flat:
            tokens = approx_tokens(block.text)
            if current and size + tokens > target_tokens:
                chunks.append(Chunk(len(chunks), section, "\n\n".join(b.text for b in current)))
                carry: list[_Block] = []
                carried = 0
                for prev in reversed(current):
                    t = approx_tokens(prev.text)
                    if prev.kind == "atomic" or carried + t > overlap_tokens:
                        break
                    carry.insert(0, prev)
                    carried += t
                current, size = carry, carried
            current.append(block)
            size += tokens
        if current:
            chunks.append(Chunk(len(chunks), section, "\n\n".join(b.text for b in current)))

    # re-number across sections so positions are unique and ordered
    for i, chunk in enumerate(chunks):
        chunk.position = i
    return chunks


def chunk_id(doc_id: str, position: int, text: str) -> str:
    digest = hashlib.sha1(f"{doc_id}:{position}:{text}".encode("utf-8")).hexdigest()
    return digest[:16]
