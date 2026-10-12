"""Turn source files into clean Markdown-like text with a heading hierarchy."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".html", ".htm", ".pdf", ".docx"}


@dataclass
class ParsedDoc:
    title: str
    text: str


class ParseError(Exception):
    pass


def parse_file(path: Path) -> ParsedDoc:
    ext = path.suffix.lower()
    try:
        if ext in {".md", ".markdown", ".txt"}:
            text = _read_text(path)
            return ParsedDoc(_title_from_markdown(text, path), text)
        if ext in {".html", ".htm"}:
            return _parse_html(path)
        if ext == ".pdf":
            return _parse_pdf(path)
        if ext == ".docx":
            return _parse_docx(path)
    except ParseError:
        raise
    except Exception as exc:  # parsing libraries raise many different errors
        raise ParseError(f"{path.name}: {exc}") from exc
    raise ParseError(f"{path.name}: unsupported file type {ext}")


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _title_from_markdown(text: str, path: Path) -> str:
    match = re.search(r"^#\s+(.+?)\s*$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else path.stem.replace("_", " ").replace("-", " ")


def _parse_html(path: Path) -> ParsedDoc:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(_read_text(path), "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "form"]):
        tag.decompose()
    root = soup.body or soup
    lines: list[str] = []
    for el in root.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "tr"]):
        if el.name == "tr":
            cells = [c.get_text(" ", strip=True) for c in el.find_all(["th", "td"])]
            line = " | ".join(c for c in cells if c)
        else:
            line = el.get_text(" ", strip=True)
        if not line:
            continue
        if el.name and el.name.startswith("h") and len(el.name) == 2 and el.name[1].isdigit():
            line = "#" * int(el.name[1]) + " " + line
        elif el.name == "li":
            line = "- " + line
        if lines and lines[-1] == line:
            continue
        lines.append(line)
    text = "\n\n".join(lines)
    title = soup.title.get_text(strip=True) if soup.title else _title_from_markdown(text, path)
    return ParsedDoc(title or path.stem, text)


def _parse_pdf(path: Path) -> ParsedDoc:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[str] = []
    for number, page in enumerate(reader.pages, start=1):
        extracted = (page.extract_text() or "").strip()
        if extracted:
            pages.append(f"## Page {number}\n\n{extracted}")
    if not pages:
        raise ParseError(f"{path.name}: no extractable text (scanned PDF? OCR it first)")
    title = path.stem.replace("_", " ").replace("-", " ")
    meta_title = getattr(reader.metadata, "title", None) if reader.metadata else None
    return ParsedDoc((meta_title or title).strip(), "\n\n".join(pages))


def _parse_docx(path: Path) -> ParsedDoc:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(str(path))
    lines: list[str] = []
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = Paragraph(child, doc)
            text = para.text.strip()
            if not text:
                continue
            style = (para.style.name if para.style is not None else "") or ""
            match = re.match(r"Heading (\d)", style)
            if style == "Title":
                lines.append(f"# {text}")
            elif match:
                lines.append("#" * min(int(match.group(1)), 6) + " " + text)
            elif style.startswith("List"):
                lines.append(f"- {text}")
            else:
                lines.append(text)
        elif tag == "tbl":
            table = Table(child, doc)
            rows = []
            for row in table.rows:
                cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                rows.append("| " + " | ".join(cells) + " |")
            if rows:
                lines.append("\n".join(rows))
    text = "\n\n".join(lines)
    if not text.strip():
        raise ParseError(f"{path.name}: document is empty")
    core_title = (doc.core_properties.title or "").strip()
    return ParsedDoc(core_title or _title_from_markdown(text, path), text)
