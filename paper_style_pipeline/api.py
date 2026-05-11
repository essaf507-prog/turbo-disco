"""FastAPI interface for training style profiles and polishing drafts."""

from __future__ import annotations

from pydantic import BaseModel, Field

from paper_style_pipeline.llm_client import LLMClient
from paper_style_pipeline.workflow import polish_with_profile, train_profile


class TrainRequest(BaseModel):
    """Request body for profile training."""

    pdf_path: str
    output_profile: str = "style_profile.json"
    chunk_size: int = Field(default=1200, gt=0)
    overlap: int = Field(default=120, ge=0)
    corpus_name: str = "academic-paper-corpus"
    model: str | None = None


class TrainResponse(BaseModel):
    """Response body after profile training."""

    profile_path: str
    chunk_count: int
    source_files: list[str]


class PolishRequest(BaseModel):
    """Request body for draft polishing."""

    profile_path: str
    text: str
    model: str | None = None


class PolishResponse(BaseModel):
    """Response body containing the polished text."""

    polished_text: str


def create_app():
    """Create and configure the FastAPI application lazily."""
    try:
        from fastapi import FastAPI
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError("HTTP API requires optional dependencies: pip install .[api]") from exc

    app = FastAPI(title="Paper Style Pipeline", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/train", response_model=TrainResponse)
    def train(request: TrainRequest) -> TrainResponse:
        client = LLMClient(model=request.model)
        profile = train_profile(
            request.pdf_path,
            request.output_profile,
            client,
            chunk_size=request.chunk_size,
            overlap=request.overlap,
            corpus_name=request.corpus_name,
        )
        return TrainResponse(
            profile_path=request.output_profile,
            chunk_count=profile.chunk_count,
            source_files=profile.source_files,
        )

    @app.post("/polish", response_model=PolishResponse)
    def polish(request: PolishRequest) -> PolishResponse:
        client = LLMClient(model=request.model)
        polished = polish_with_profile(request.profile_path, request.text, client)
        return PolishResponse(polished_text=polished)

    return app


app = create_app()
