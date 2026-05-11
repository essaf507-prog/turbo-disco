"""Paragraph-aware chunking for extracted paper text."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A chunk of text and lightweight source metadata."""

    text: str
    index: int
    source: str | None = None
    page_start: int | None = None
    page_end: int | None = None


_WORD_RE = re.compile(r"\S+")
_PARAGRAPH_RE = re.compile(r"\n\s*\n+")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def estimate_tokens(text: str) -> int:
    """Estimate tokens without model-specific tokenizers.

    Academic English averages roughly 0.75 words per token. The estimate is deliberately
    conservative so chunks stay below common model limits.
    """
    word_count = len(_WORD_RE.findall(text))
    return max(1, int(word_count / 0.75)) if text.strip() else 0


def chunk_text(
    text: str,
    *,
    max_tokens: int = 1200,
    overlap_tokens: int = 120,
    source: str | None = None,
) -> list[TextChunk]:
    """Split text into paragraph-first chunks with optional trailing overlap."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero.")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens cannot be negative.")
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens.")

    paragraphs = [part.strip() for part in _PARAGRAPH_RE.split(text) if part.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        if estimate_tokens(paragraph) <= max_tokens:
            units.append(paragraph)
        else:
            units.extend(_split_long_paragraph(paragraph, max_tokens))

    chunks: list[str] = []
    current: list[str] = []
    for unit in units:
        candidate = "\n\n".join([*current, unit]) if current else unit
        if current and estimate_tokens(candidate) > max_tokens:
            chunks.append("\n\n".join(current))
            current = _overlap_tail(current, overlap_tokens)
        current.append(unit)
    if current:
        chunks.append("\n\n".join(current))

    return [TextChunk(text=chunk, index=index, source=source) for index, chunk in enumerate(chunks)]


def _split_long_paragraph(paragraph: str, max_tokens: int) -> list[str]:
    sentences = [part.strip() for part in _SENTENCE_BOUNDARY_RE.split(paragraph) if part.strip()]
    if not sentences:
        return []
    parts: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        if estimate_tokens(sentence) > max_tokens:
            if current:
                parts.append(" ".join(current))
                current = []
            parts.extend(_split_by_words(sentence, max_tokens))
            continue
        candidate = " ".join([*current, sentence]) if current else sentence
        if current and estimate_tokens(candidate) > max_tokens:
            parts.append(" ".join(current))
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        parts.append(" ".join(current))
    return parts


def _split_by_words(text: str, max_tokens: int) -> list[str]:
    words = _WORD_RE.findall(text)
    # Convert max token estimate back into a conservative word budget.
    word_budget = max(1, int(max_tokens * 0.75))
    return [
        " ".join(words[index : index + word_budget])
        for index in range(0, len(words), word_budget)
    ]


def _overlap_tail(units: list[str], overlap_tokens: int) -> list[str]:
    if overlap_tokens == 0:
        return []
    tail: list[str] = []
    total = 0
    for unit in reversed(units):
        unit_tokens = estimate_tokens(unit)
        if tail and total + unit_tokens > overlap_tokens:
            break
        tail.insert(0, unit)
        total += unit_tokens
    return tail
