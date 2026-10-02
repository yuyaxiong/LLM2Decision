from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import APIRouter, FastAPI, HTTPException, Request

from ..core.config import Config, UnknownModelError
from ..core.readout import ReadoutError
from ..core.schema import SystemOneRequest, SystemOneResponse
from ..llm.client import ChatClient, ChatError
from ..llm.service import SystemOneService, TooManyCandidatesError
from .debug import router as debug_router

logger = logging.getLogger("llm2decision")

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict:
    config: Config = request.app.state.config
    return {
        "status": "ok",
        "default_model": config.default_model,
        "models": config.model_names(),
        "api_key_configured": all(bool(settings.api_key) for settings in config.models.values()),
    }


@router.get("/v1/models")
async def list_models(request: Request) -> dict:
    """List the routable model routes, for upstreams to wire up routing."""
    config: Config = request.app.state.config
    return {
        "default": config.default_model,
        "models": [
            {
                "name": name,
                "model": settings.model,
                "provider": settings.provider,
                "base_url": settings.base_url,
                "temperature_scale": settings.temperature_scale,
                "max_tokens": settings.max_tokens,
                "disable_thinking": settings.disable_thinking,
                "logit_bias_enabled": settings.logit_bias_enabled,
                "prompt_version": settings.prompt_version,
            }
            for name, settings in config.models.items()
        ],
    }


async def _decide(payload: SystemOneRequest, request: Request) -> SystemOneResponse:
    service: SystemOneService = request.app.state.service
    try:
        return await service.decide(payload)
    except (UnknownModelError, TooManyCandidatesError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ReadoutError as exc:
        logger.warning("logprob readout failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"logprob readout failed: {exc}") from exc
    except ChatError as exc:
        logger.warning("model call failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"model call failed: {exc}") from exc


@router.post("/v1/decide", response_model=SystemOneResponse)
async def decide(payload: SystemOneRequest, request: Request) -> SystemOneResponse:
    """Native endpoint: one request returns decisions and probability distributions for several questions."""
    return await _decide(payload, request)


@router.post("/v1/systemone", response_model=SystemOneResponse)
async def systemone(payload: SystemOneRequest, request: Request) -> SystemOneResponse:
    """Compatibility endpoint: same name and shape as Jev's `/v1/systemone`, to ease migration for existing callers.

    LLM2Decision is not affiliated with TypeSafe AI or its Jev product, and is not endorsed by them.
    """
    return await _decide(payload, request)


def create_app(config: Optional[Config] = None) -> FastAPI:
    """Build the app.

    Pass `config` to inject configuration (tests, embedded use); otherwise it reads the config file at
    startup in the order `LLM2DECISION_CONFIG` → `llm2decision.yaml`.
    """

    @asynccontextmanager
    async def lifespan(instance: FastAPI):
        loaded = config if config is not None else Config.load()
        clients = {name: ChatClient(settings) for name, settings in loaded.models.items()}
        instance.state.config = loaded
        instance.state.service = SystemOneService(clients, loaded)
        logger.info("Loaded model routes %s, default route %s", loaded.model_names(), loaded.default_model)
        try:
            yield
        finally:
            for client in clients.values():
                await client.aclose()

    api = FastAPI(title="LLM2Decision", version="0.1.0", lifespan=lifespan)
    api.include_router(router)
    api.include_router(debug_router)
    return api


# uvicorn llm2decision.api.main:app
app = create_app()
