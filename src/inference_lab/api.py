from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Header
from pydantic import BaseModel, Field

from .backend import DeterministicBackend
from .models import GenerationRequest
from .service import InferenceService


class GenerateBody(BaseModel):
    prompt: str = Field(min_length=1, max_length=20_000)
    max_tokens: int = Field(default=64, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


def create_app(service: InferenceService | None = None) -> FastAPI:
    service = service or InferenceService(DeterministicBackend())

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await service.close()

    app = FastAPI(title="InferenceLab", version="1.0.0", lifespan=lifespan)

    @app.post("/v1/generate")
    async def generate(body: GenerateBody, x_tenant: str = Header("default")) -> dict[str, object]:
        # x-tenant only partitions the cache. In a deployment, the authenticating gateway in front of
        # this service sets it; it is not an authorisation decision.
        result = await service.generate(GenerationRequest(body.prompt, body.max_tokens, body.temperature, x_tenant))
        return result.__dict__

    return app


app = create_app()
