"""Shared Pydantic AI Direct implementation of the model backend port."""

import json
import logging
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, cast

import httpx
from pydantic_ai import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
)
from pydantic_ai.direct import model_request
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from app.model_backends.contracts import (
    JsonObject,
    ModelConversation,
    ModelGeneration,
    ModelReasoning,
    ModelUsage,
)

LOGGER = logging.getLogger(__name__)

_SUPPORTED_PROVIDERS = frozenset({"anthropic", "openai"})
_REQUEST_TIMEOUT_SECONDS = 60.0


@dataclass(slots=True)
class _RequestCaptureState:
    """Mutable result slot owned by exactly one active generation call."""

    count: int = 0
    snapshot: JsonObject | None = None
    invalid: bool = False

    def record(self, value: object) -> None:
        """Record one decoded body without retaining an earlier valid value."""
        self.count += 1
        if self.count != 1 or type(value) is not dict:
            self.invalid = True
            self.snapshot = None
            return

        try:
            # Reject non-finite floats and other values JSON would distort.
            json.dumps(value, allow_nan=False)
        except TypeError, ValueError:
            self.invalid = True
            self.snapshot = None
            return
        self.snapshot = cast(JsonObject, value)

    def require_snapshot(self) -> JsonObject:
        """Return the sole valid body or fail without exposing its contents."""
        if self.count != 1 or self.invalid or self.snapshot is None:
            raise ValueError("invalid model request snapshot")
        return self.snapshot


_ACTIVE_REQUEST_CAPTURE: ContextVar[_RequestCaptureState | None] = ContextVar(
    "active_model_request_capture",
    default=None,
)


def _reject_json_constant(value: str) -> None:
    """Reject non-standard JSON constants such as NaN and Infinity."""
    del value
    raise ValueError("invalid JSON constant")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build an object while rejecting ambiguous duplicate member names."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object member")
        result[key] = value
    return result


def _log_request_failure(provider: str, model: str, exc: BaseException) -> None:
    """Record request-level failures without leaking URL/auth/credentials.

    Pydantic AI's ModelHTTPError already strips transport credentials and
    surfaces a decoded body; its traceback only reflects the raise site and
    does not carry URLs or auth headers, so it is safe to include. For any
    other exception we deliberately avoid the message and traceback, since
    upstream SDK errors may embed request URLs, bodies, or credentials.
    """
    if isinstance(exc, ModelHTTPError):
        LOGGER.exception(
            "model request failed: provider=%s model=%s status=%s "
            "provider_model=%s retry_after=%r body=%r",
            provider,
            model,
            exc.status_code,
            exc.model_name,
            exc.retry_after,
            exc.body,
        )
        return
    LOGGER.warning(
        "model request failed: provider=%s model=%s exc_type=%s",
        provider,
        model,
        type(exc).__name__,
    )


def _summarize_response(response: ModelResponse) -> dict[str, object]:
    """Project a ModelResponse into a safe, log-friendly summary.

    Exposes only response-side fields: part types and visible/reasoning text.
    Never includes request bodies, URLs, or credentials.
    """
    parts_summary: list[dict[str, object]] = []
    for index, part in enumerate(response.parts):
        entry: dict[str, object] = {"index": index, "kind": type(part).__name__}
        if isinstance(part, TextPart):
            entry["content"] = part.content
        elif isinstance(part, ThinkingPart):
            content = part.content
            if isinstance(content, str):
                # Reasoning can be very long; keep logs bounded.
                entry["content"] = (
                    content if len(content) <= 512 else content[:512] + "..."
                )
        parts_summary.append(entry)
    usage = response.usage
    return {
        "parts": parts_summary,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
    }


def _log_projection_failure(
    provider: str,
    model: str,
    response: ModelResponse,
    exc: BaseException,
) -> None:
    """Record projection/validation failures with the actual model output."""
    LOGGER.warning(
        "model output could not be projected: provider=%s model=%s "
        "exc_type=%s message=%s response=%s",
        provider,
        model,
        type(exc).__name__,
        str(exc),
        _summarize_response(response),
        exc_info=True,
    )


async def _capture_request_body(request: httpx.Request) -> None:
    """Capture only the serialized JSON body for the active model call."""
    state = _ACTIVE_REQUEST_CAPTURE.get()
    if state is None:
        return

    try:
        body = await request.aread()
        value = json.loads(
            body,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except TypeError, ValueError, UnicodeDecodeError:
        state.count += 1
        state.invalid = True
        state.snapshot = None
        return
    state.record(value)


def create_request_capture_client(
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.AsyncClient:
    """Create the shared 60-second HTTP client with a body-only hook."""
    return httpx.AsyncClient(
        timeout=_REQUEST_TIMEOUT_SECONDS,
        transport=transport,
        event_hooks={"request": [_capture_request_body]},
    )


class PydanticAIBackend:
    """Map neutral conversations through one Pydantic AI Direct model."""

    def __init__(
        self,
        *,
        model: str,
        direct_model: Model,
        model_settings: ModelSettings,
        http_client: httpx.AsyncClient,
    ) -> None:
        """Bind one configured model and its owned shared HTTP client."""
        if direct_model.model_name != model:
            raise ValueError(
                "Pydantic AI model identity does not match settings"
            )
        if direct_model.system not in _SUPPORTED_PROVIDERS:
            raise ValueError("unsupported Pydantic AI provider")

        self._model = model
        self._direct_model = direct_model
        self._model_settings = cast(ModelSettings, dict(model_settings))
        self._http_client = http_client
        self._provider = direct_model.system
        self._closed = False

    @property
    def model(self) -> str:
        """Return the exact configured model identity."""
        return self._model

    async def generate(
        self,
        conversation: ModelConversation,
    ) -> ModelGeneration:
        """Perform one Direct request and project its captured safe result."""
        if self._closed:
            raise RuntimeError("model backend is closed")

        messages = _build_messages(conversation)
        capture = _RequestCaptureState()
        token = _ACTIVE_REQUEST_CAPTURE.set(capture)
        try:
            response: ModelResponse | None = None
            try:
                response = await model_request(
                    self._direct_model,
                    messages,
                    model_settings=self._model_settings,
                    instrument=False,
                )
            except Exception as exc:
                # Provider exceptions can contain URLs, bodies, and credentials;
                # surface their safe fields to the console before dropping them.
                _log_request_failure(self._provider, self._model, exc)
                request_failed = True
            else:
                request_failed = False

            if request_failed:
                # Raising after the handler also drops sensitive context.
                raise RuntimeError("Pydantic AI model request failed") from None

            try:
                snapshot = capture.require_snapshot()
                assert response is not None
                return _project_response(response, self._provider, snapshot)
            except Exception as exc:
                # The request succeeded but the response cannot be
                # projected; log the actual model output so invalid
                # formats are diagnosable, then re-raise unchanged.
                _log_projection_failure(
                    self._provider, self._model, response, exc
                )
                raise
        finally:
            # A later call must never inherit this call's body or failure state.
            _ACTIVE_REQUEST_CAPTURE.reset(token)

    async def aclose(self) -> None:
        """Close the owned shared HTTP client at most once."""
        if self._closed:
            return
        self._closed = True
        await self._http_client.aclose()


def _build_messages(
    conversation: ModelConversation,
) -> list[ModelRequest | ModelResponse]:
    """Map complete confirmed history and append the sole current request."""
    messages: list[ModelRequest | ModelResponse] = []
    for turn in conversation.turns:
        messages.extend(
            (
                ModelRequest.user_text_prompt(turn.input),
                ModelResponse(parts=[TextPart(turn.output)]),
            )
        )
    messages.append(
        ModelRequest.user_text_prompt(
            conversation.current_input,
            instructions=conversation.system_prompt,
        )
    )
    return messages


def _project_response(
    response: ModelResponse,
    provider: str,
    snapshot: JsonObject,
) -> ModelGeneration:
    """Accept one visible text plus provider-approved readable reasoning."""
    text_parts: list[TextPart] = []
    thinking_parts: list[ThinkingPart] = []
    for part in response.parts:
        if isinstance(part, TextPart):
            text_parts.append(part)
        elif isinstance(part, ThinkingPart):
            thinking_parts.append(part)
        else:
            raise ValueError("unexpected Pydantic AI response part")

    if len(text_parts) != 1:
        raise ValueError("expected exactly one visible text part")
    content = text_parts[0].content
    if type(content) is not str or not content.strip():
        raise ValueError("visible model text must not be blank")

    return ModelGeneration(
        content=content,
        reasoning=_project_reasoning(thinking_parts, provider),
        usage=_project_usage(response),
        request_snapshot=snapshot,
    )


def _project_reasoning(
    parts: list[ThinkingPart],
    provider: str,
) -> tuple[ModelReasoning, ...]:
    """Expose only documented readable fields for the configured provider."""
    reasoning: list[ModelReasoning] = []
    for part in parts:
        content = part.content
        if type(content) is str and content.strip():
            reasoning.append(
                ModelReasoning(
                    type="thinking"
                    if provider == "anthropic"
                    else "summary_text",
                    text=content,
                )
            )

        if provider != "openai" or type(part.provider_details) is not dict:
            continue
        raw_content = part.provider_details.get("raw_content")
        if type(raw_content) is not list or not all(
            type(value) is str for value in raw_content
        ):
            continue
        reasoning.extend(
            ModelReasoning(type="reasoning_text", text=value)
            for value in raw_content
            if value.strip()
        )
    return tuple(reasoning)


def _project_usage(response: ModelResponse) -> ModelUsage:
    """Partition inclusive Pydantic AI input usage into project buckets."""
    usage = response.usage
    total_input = _token_count(usage.input_tokens)
    output = _token_count(usage.output_tokens)
    cache_write = _token_count(usage.cache_write_tokens)
    cache_read = _token_count(usage.cache_read_tokens)
    uncached_input = total_input - cache_write - cache_read
    if uncached_input < 0:
        raise ValueError("cache usage exceeds total model input")
    return ModelUsage(
        input_tokens=uncached_input,
        output_tokens=output,
        cache_creation_input_tokens=cache_write,
        cache_read_input_tokens=cache_read,
    )


def _token_count(value: Any) -> int:
    """Require an exact non-negative integer token count."""
    if type(value) is not int or value < 0:
        raise ValueError("invalid Pydantic AI token count")
    return value
