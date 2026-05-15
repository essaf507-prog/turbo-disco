from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from paper_style_pipeline.chunker import TextChunk
from paper_style_pipeline.polisher import polish_text
from paper_style_pipeline.style_analyzer import analyze_chunks
from paper_style_pipeline.style_profile import StyleProfile


class FakeClient:
    def complete(self, messages: list[dict[str, str]], *, temperature: float = 0.2) -> str:
        prompt = messages[-1]["content"]
        if "User draft to polish" in prompt:
            return "This study systematically examines the proposed approach."
        return """{
          "summary": "Formal, cautious, and contribution-oriented.",
          "rhetorical_moves": ["State the research gap before the contribution."],
          "sentence_patterns": ["This study + verb + object + qualifier."],
          "transitions": ["However, to address this limitation,"],
          "academic_phrases": ["the proposed approach"],
          "tone_rules": ["Use precise hedging for claims."],
          "do_not_rules": ["Avoid colloquial intensifiers."],
          "examples": ["This study examines X in the context of Y."]
        }"""


class ProfileWorkflowTests(unittest.TestCase):
    def test_profile_round_trip(self) -> None:
        profile = StyleProfile(summary="Formal style", transitions=["Moreover,"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            profile.save(path)
            loaded = StyleProfile.load(path)

        self.assertEqual(loaded.summary, "Formal style")
        self.assertEqual(loaded.transitions, ["Moreover,"])

    def test_analyze_chunks_uses_fake_client(self) -> None:
        profile = analyze_chunks([TextChunk("This paper proposes a method.", 0)], FakeClient())

        self.assertEqual(profile.chunk_count, 1)
        self.assertIn("Formal", profile.summary)
        self.assertIn("Use precise hedging for claims.", profile.tone_rules)

    def test_polish_text_uses_profile_context(self) -> None:
        result = polish_text(
            "This talks about our method.",
            StyleProfile(summary="Formal"),
            FakeClient(),
        )

        self.assertEqual(result, "This study systematically examines the proposed approach.")


if __name__ == "__main__":
    unittest.main()
