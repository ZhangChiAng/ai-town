"""OpenAI Responses assembly for the shared Pydantic AI Direct backend."""

import logging
from contextlib import suppress

import httpx
from openai import AsyncOpenAI
from pydantic_ai.models.openai import (
    OpenAIResponsesModel,
    OpenAIResponsesModelSettings,
)
from pydantic_ai.providers.openai import OpenAIProvider

from app.model_backends.contracts import ModelBackendSettings
from app.model_backends.pydantic_ai_backend import (
    PydanticAIBackend,
    create_request_capture_client,
)
from app.structured_logging import log_event

LOGGER = logging.getLogger(__name__)


async def create_openai_responses_backend(
    settings: ModelBackendSettings,
    /,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> PydanticAIBackend:
    """Create one stateless OpenAI Responses model using Direct requests."""
    http_client: httpx.AsyncClient | None = None
    try:
        http_client = create_request_capture_client(transport=transport)
        client = AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=60.0,
            max_retries=0,
            http_client=http_client,
        )
        provider = OpenAIProvider(openai_client=client)
        direct_model = OpenAIResponsesModel(settings.model, provider=provider)
        model_settings = OpenAIResponsesModelSettings(
            openai_reasoning_effort="max",
            openai_reasoning_summary="auto",
            openai_reasoning_context="current_turn",
            openai_store=False,
            openai_send_reasoning_ids=False,
        )
        return PydanticAIBackend(
            model=settings.model,
            direct_model=direct_model,
            model_settings=model_settings,
            http_client=http_client,
            provider="openai",
        )
    except BaseException as error:
        log_event(
            LOGGER,
            logging.ERROR,
            "model.provider_initialization.failed",
            "OpenAI provider initialization failed.",
            exception=error,
            model=settings.model,
            provider="openai",
        )
        if http_client is not None:
            # Cleanup errors must not replace the sanitized constructor error.
            with suppress(Exception):
                await http_client.aclose()
        if not isinstance(error, Exception):
            raise

    # Raising outside the handler drops sensitive constructor context.
    raise RuntimeError("OpenAI Responses client creation failed") from None
