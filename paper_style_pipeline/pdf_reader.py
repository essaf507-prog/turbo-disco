"""PDF text extraction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PageText:
    """Text extracted from one PDF page."""

    source: str
    page_number: int
    text: str


def discover_pdfs(path: str | Path) -> list[Path]:
    """Return sorted PDF paths from a single file or directory."""
    target = Path(path)
    if target.is_file():
        if target.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {target}")
        return [target]
    if target.is_dir():
        return sorted(item for item in target.rglob("*.pdf") if item.is_file())
    raise FileNotFoundError(f"PDF path does not exist: {target}")


def extract_pdf_pages(path: str | Path) -> list[PageText]:
    """Extract text from every page of a PDF file using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError(
            "PDF extraction requires the optional dependency: pip install pypdf"
        ) from exc

    pdf_path = Path(path)
    reader = PdfReader(str(pdf_path))
    pages: list[PageText] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(
                PageText(
                    source=str(pdf_path),
                    page_number=index,
                    text=normalize_pdf_text(text),
                )
            )
    return pages


def extract_corpus_text(path: str | Path) -> tuple[str, list[str]]:
    """Extract and concatenate text from all PDFs under path."""
    texts: list[str] = []
    sources: list[str] = []
    for pdf_path in discover_pdfs(path):
        page_texts = extract_pdf_pages(pdf_path)
        if page_texts:
            sources.append(str(pdf_path))
            page_blocks = [f"[Page {page.page_number}]\n{page.text}" for page in page_texts]
            texts.append(f"[Source: {pdf_path}]\n" + "\n\n".join(page_blocks))
    return "\n\n".join(texts), sources


def normalize_pdf_text(text: str) -> str:
    """Clean common PDF extraction artifacts while preserving paragraph boundaries."""
    lines = [line.strip() for line in text.replace("\r", "\n").split("\n")]
    cleaned: list[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if not previous_blank:
                cleaned.append("")
            previous_blank = True
            continue
        cleaned.append(line)
        previous_blank = False
    return "\n".join(cleaned).strip()
