"""Command line interface for the paper style pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from paper_style_pipeline.llm_client import LLMClient
from paper_style_pipeline.workflow import polish_with_profile, train_profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze PDF paper style and polish drafts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Build a reusable style profile from PDF papers.")
    train.add_argument("pdf_path", help="PDF file or directory containing PDF papers.")
    train.add_argument("--output", default="style_profile.json", help="Output JSON profile path.")
    train.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        help="Approximate maximum tokens per chunk.",
    )
    train.add_argument(
        "--overlap",
        type=int,
        default=120,
        help="Approximate overlap tokens between chunks.",
    )
    train.add_argument(
        "--corpus-name", default="academic-paper-corpus", help="Name stored in profile."
    )
    train.add_argument("--model", default=None, help="Override LLM_MODEL for this run.")

    polish = subparsers.add_parser("polish", help="Polish a draft using a saved style profile.")
    polish.add_argument("--profile", required=True, help="Style profile JSON path.")
    polish.add_argument("--input", required=True, help="Draft text file path.")
    polish.add_argument(
        "--output", default=None, help="Optional output file; prints to stdout if omitted."
    )
    polish.add_argument("--model", default=None, help="Override LLM_MODEL for this run.")

    serve = subparsers.add_parser("serve", help="Run the HTTP API server.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        client = LLMClient(model=args.model)
        profile = train_profile(
            args.pdf_path,
            args.output,
            client,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            corpus_name=args.corpus_name,
        )
        print(f"Saved style profile to {args.output} ({profile.chunk_count} chunks).")
        return 0

    if args.command == "polish":
        client = LLMClient(model=args.model)
        draft = Path(args.input).read_text(encoding="utf-8")
        polished = polish_with_profile(args.profile, draft, client)
        if args.output:
            Path(args.output).write_text(polished, encoding="utf-8")
            print(f"Saved polished text to {args.output}.")
        else:
            print(polished)
        return 0

    if args.command == "serve":
        try:
            import uvicorn
        except ImportError as exc:  # pragma: no cover - depends on optional dependency
            raise RuntimeError(
                "Serving requires optional dependencies: pip install .[api]"
            ) from exc
        uvicorn.run(
            "paper_style_pipeline.api:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
        return 0

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
