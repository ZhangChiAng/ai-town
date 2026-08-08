"""Anthropic assembly for the shared Pydantic AI Direct backend."""

import logging
from contextlib import suppress

import httpx
from anthropic import AsyncAnthropic
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider

from app.model_backends.contracts import ModelBackendSettings
from app.model_backends.pydantic_ai_backend import (
    PydanticAIBackend,
    create_request_capture_client,
)
from app.structured_logging import log_event

_MAX_TOKENS = 1024
LOGGER = logging.getLogger(__name__)


async def create_anthropic_messages_backend(
    settings: ModelBackendSettings,
    /,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> PydanticAIBackend:
    """Create one bounded Anthropic model using the shared Direct backend."""
    http_client: httpx.AsyncClient | None = None
    try:
        http_client = create_request_capture_client(transport=transport)
        client = AsyncAnthropic(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=60.0,
            max_retries=0,
            http_client=http_client,
        )
        provider = AnthropicProvider(anthropic_client=client)
        direct_model = AnthropicModel(settings.model, provider=provider)
        model_settings = AnthropicModelSettings(
            max_tokens=_MAX_TOKENS,
            anthropic_cache_instructions="5m",
            anthropic_cache_messages="5m",
        )
        return PydanticAIBackend(
            model=settings.model,
            direct_model=direct_model,
            model_settings=model_settings,
            http_client=http_client,
            provider="anthropic",
        )
    except BaseException as error:
        log_event(
            LOGGER,
            logging.ERROR,
            "model.provider_initialization.failed",
            "Anthropic provider initialization failed.",
            exception=error,
            model=settings.model,
            provider="anthropic",
        )
        if http_client is not None:
            # Cleanup errors must not replace the sanitized constructor error.
            with suppress(Exception):
                await http_client.aclose()
        if not isinstance(error, Exception):
            raise

    # Raising outside the handler drops sensitive constructor context.
    raise RuntimeError("Anthropic Messages client creation failed") from None
