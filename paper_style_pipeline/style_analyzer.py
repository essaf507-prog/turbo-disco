"""Analyze chunk-level academic writing style and merge it into a profile."""

from __future__ import annotations

import json
from collections.abc import Iterable

from paper_style_pipeline.chunker import TextChunk
from paper_style_pipeline.llm_client import ChatClient
from paper_style_pipeline.style_profile import StyleProfile

_ANALYZE_SYSTEM_PROMPT = """You are an expert academic writing analyst.
Extract reusable English academic paper style patterns from the provided chunk.
Focus on expression habits rather than paper facts. Return strict JSON only."""

_MERGE_SYSTEM_PROMPT = """You are an expert editor building a reusable academic style guide.
Merge chunk analyses into one concise style profile. Return strict JSON only."""


def analyze_chunks(
    chunks: Iterable[TextChunk],
    client: ChatClient,
    *,
    corpus_name: str = "academic-paper-corpus",
    source_files: list[str] | None = None,
) -> StyleProfile:
    """Analyze text chunks with an LLM and merge them into a style profile."""
    chunk_list = list(chunks)
    if not chunk_list:
        raise ValueError("At least one chunk is required to analyze style.")

    analyses = [_analyze_one_chunk(chunk, client) for chunk in chunk_list]
    merged = _merge_analyses(analyses, client)
    profile = StyleProfile.from_dict(merged)
    profile.corpus_name = corpus_name
    profile.source_files = source_files or sorted(
        {chunk.source for chunk in chunk_list if chunk.source}
    )
    profile.chunk_count = len(chunk_list)
    return profile


def _analyze_one_chunk(chunk: TextChunk, client: ChatClient) -> dict[str, object]:
    schema = _profile_schema_description(include_summary=True)
    content = f"""Analyze this chunk as chunk #{chunk.index}. Do not summarize research findings.
Extract only reusable writing-style patterns.

Required JSON schema:
{schema}

Chunk:
{chunk.text}"""
    raw = client.complete(
        [
            {"role": "system", "content": _ANALYZE_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        temperature=0.1,
    )
    return _parse_json_object(raw)


def _merge_analyses(analyses: list[dict[str, object]], client: ChatClient) -> dict[str, object]:
    schema = _profile_schema_description(include_summary=True)
    raw = client.complete(
        [
            {"role": "system", "content": _MERGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Merge these chunk analyses into a concise reusable style profile. "
                    "Deduplicate generic items and keep the most actionable patterns.\n\n"
                    f"Required JSON schema:\n{schema}\n\n"
                    f"Chunk analyses:\n{json.dumps(analyses, ensure_ascii=False)}"
                ),
            },
        ],
        temperature=0.1,
    )
    return _parse_json_object(raw)


def _profile_schema_description(*, include_summary: bool) -> str:
    summary = '"summary": "2-4 sentence overview of the style",' if include_summary else ""
    return f"""{{
  {summary}
  "rhetorical_moves": ["how claims, gaps, methods, results, and implications are expressed"],
  "sentence_patterns": ["reusable sentence structure or grammar pattern"],
  "transitions": ["transition words/phrases and when to use them"],
  "academic_phrases": ["domain-neutral academic phrase templates"],
  "tone_rules": ["rules for cautious, precise, formal tone"],
  "do_not_rules": ["style mistakes to avoid"],
  "examples": ["short original-style example sentence templates"]
}}"""


def _parse_json_object(raw: str) -> dict[str, object]:
    """Parse JSON even if the model wrapped it in Markdown fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"LLM response did not contain a JSON object: {raw[:200]}")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("LLM JSON response must be an object.")
    return parsed
