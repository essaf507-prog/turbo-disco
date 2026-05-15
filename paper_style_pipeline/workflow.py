"""High-level orchestration for training profiles and polishing drafts."""

from __future__ import annotations

from pathlib import Path

from paper_style_pipeline.chunker import chunk_text
from paper_style_pipeline.llm_client import ChatClient
from paper_style_pipeline.pdf_reader import extract_corpus_text
from paper_style_pipeline.polisher import polish_text
from paper_style_pipeline.style_analyzer import analyze_chunks
from paper_style_pipeline.style_profile import StyleProfile


def train_profile(
    pdf_path: str | Path,
    output_profile: str | Path,
    client: ChatClient,
    *,
    chunk_size: int = 1200,
    overlap: int = 120,
    corpus_name: str = "academic-paper-corpus",
) -> StyleProfile:
    """Extract PDFs, analyze style, save a profile, and return it."""
    text, sources = extract_corpus_text(pdf_path)
    if not text.strip():
        raise ValueError(f"No extractable text found in PDF path: {pdf_path}")
    chunks = chunk_text(text, max_tokens=chunk_size, overlap_tokens=overlap, source=str(pdf_path))
    profile = analyze_chunks(chunks, client, corpus_name=corpus_name, source_files=sources)
    profile.save(output_profile)
    return profile


def polish_with_profile(profile_path: str | Path, text: str, client: ChatClient) -> str:
    """Load a saved profile and polish text."""
    profile = StyleProfile.load(profile_path)
    return polish_text(text, profile, client)
