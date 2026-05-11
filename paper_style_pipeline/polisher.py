"""Polish a draft using an extracted academic style profile."""

from __future__ import annotations

from paper_style_pipeline.llm_client import ChatClient
from paper_style_pipeline.style_profile import StyleProfile

_POLISH_SYSTEM_PROMPT = """You are an expert English academic editor.
Rewrite the user's draft so it follows the supplied corpus style profile.
Preserve meaning, claims, citations, equations, headings, and factual scope.
Do not invent references or results. Return only the polished document."""


def polish_text(text: str, profile: StyleProfile, client: ChatClient) -> str:
    """Return a polished version of `text` using the given style profile."""
    if not text.strip():
        raise ValueError("Input text cannot be empty.")
    profile_context = profile.to_prompt_context()
    prompt = f"""Style profile extracted from the paper corpus:
{profile_context}

User draft to polish:
{text}

Polish the draft in English academic paper style. Keep the original meaning and
structure unless a sentence-level change improves clarity."""
    return client.complete(
        [
            {"role": "system", "content": _POLISH_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
