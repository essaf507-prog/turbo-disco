"""Data structures for reusable academic writing style profiles."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class StyleProfile:
    """A compact, reusable representation of a paper corpus writing style."""

    corpus_name: str = "academic-paper-corpus"
    source_files: list[str] = field(default_factory=list)
    chunk_count: int = 0
    summary: str = ""
    rhetorical_moves: list[str] = field(default_factory=list)
    sentence_patterns: list[str] = field(default_factory=list)
    transitions: list[str] = field(default_factory=list)
    academic_phrases: list[str] = field(default_factory=list)
    tone_rules: list[str] = field(default_factory=list)
    do_not_rules: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StyleProfile:
        """Create a profile from a dictionary while tolerating missing keys."""
        known_fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in data.items() if key in known_fields})

    def save(self, path: str | Path) -> None:
        """Persist the profile as UTF-8 JSON."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> StyleProfile:
        """Load a profile from JSON."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Style profile JSON must contain an object at the top level.")
        return cls.from_dict(data)

    def to_prompt_context(self) -> str:
        """Render the profile into concise prompt context for polishing."""
        sections = [
            ("Corpus summary", [self.summary] if self.summary else []),
            ("Rhetorical moves", self.rhetorical_moves),
            ("Sentence patterns", self.sentence_patterns),
            ("Transitions", self.transitions),
            ("Reusable academic phrases", self.academic_phrases),
            ("Tone rules", self.tone_rules),
            ("Avoid", self.do_not_rules),
            ("Examples", self.examples),
        ]
        rendered: list[str] = []
        for title, values in sections:
            if values:
                rendered.append(f"{title}:\n" + "\n".join(f"- {value}" for value in values))
        return "\n\n".join(rendered)
