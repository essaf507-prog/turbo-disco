from __future__ import annotations

import unittest

from paper_style_pipeline.chunker import chunk_text, estimate_tokens


class ChunkerTests(unittest.TestCase):
    def test_estimate_tokens_handles_empty_text(self) -> None:
        self.assertEqual(estimate_tokens("   "), 0)

    def test_chunk_text_preserves_paragraph_content(self) -> None:
        text = "First paragraph has a clear claim.\n\nSecond paragraph develops the method."
        chunks = chunk_text(text, max_tokens=20, overlap_tokens=0, source="paper.pdf")

        self.assertGreaterEqual(len(chunks), 1)
        self.assertEqual(chunks[0].source, "paper.pdf")
        self.assertIn("First paragraph", "\n".join(chunk.text for chunk in chunks))
        self.assertIn("Second paragraph", "\n".join(chunk.text for chunk in chunks))

    def test_chunk_text_rejects_invalid_overlap(self) -> None:
        with self.assertRaises(ValueError):
            chunk_text("hello", max_tokens=10, overlap_tokens=10)


if __name__ == "__main__":
    unittest.main()
