from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .backend import DeterministicBackend
from .models import GenerationRequest
from .service import InferenceService


class GenerateBody(BaseModel):
    prompt: str = Field(min_length=1, max_length=20_000)
    max_tokens: int = Field(default=64, ge=1, le=4096)


def create_app(service: InferenceService | None = None) -> FastAPI:
    service = service or InferenceService(DeterministicBackend())

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await service.close()

    app = FastAPI(title="InferenceLab", version="1.0.0", lifespan=lifespan)

    @app.post("/v1/generate")
    async def generate(body: GenerateBody) -> dict[str, object]:
        result = await service.generate(GenerationRequest(body.prompt, body.max_tokens))
        return result.__dict__

    return app


app = create_app()
