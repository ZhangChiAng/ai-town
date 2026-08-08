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
from app.structured_logging import bind_log_context, log_event

LOGGER = logging.getLogger(__name__)

_SUPPORTED_PROVIDERS = frozenset({"anthropic", "openai"})
_REQUEST_TIMEOUT_SECONDS = 60.0


@dataclass(slots=True)
class _RequestCaptureState:
    """Mutable result slot owned by exactly one active generation call."""

    count: int = 0
    snapshot: JsonObject | None = None
    invalid: bool = False
    wire_requests: list[dict[str, object]] | None = None

    def record(
        self,
        value: object,
        wire_request: dict[str, object],
    ) -> None:
        """Record one decoded body and its error-only wire metadata."""
        self.count += 1
        if self.wire_requests is None:
            self.wire_requests = []
        self.wire_requests.append(wire_request)
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

    def record_invalid(self, wire_request: dict[str, object]) -> None:
        """Retain malformed wire data for logs while rejecting the snapshot."""
        self.count += 1
        if self.wire_requests is None:
            self.wire_requests = []
        self.wire_requests.append(wire_request)
        self.invalid = True
        self.snapshot = None

    def requests_for_logging(self) -> list[dict[str, object]]:
        """Return every actual HTTP request observed for this call."""
        return list(self.wire_requests or [])

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


def _log_request_failure(
    provider: str,
    model: str,
    conversation: ModelConversation,
    capture: _RequestCaptureState,
    exc: BaseException,
) -> None:
    """Record untruncated request, provider error, and exception details."""
    log_event(
        LOGGER,
        logging.ERROR,
        "model.provider_request.failed",
        "Provider request failed.",
        exception=exc,
        provider=provider,
        model=model,
        conversation=conversation,
        serialized_requests=capture.requests_for_logging(),
        provider_error=exc,
        provider_http_body=(
            exc.body if isinstance(exc, ModelHTTPError) else None
        ),
    )


def _log_projection_failure(
    provider: str,
    model: str,
    conversation: ModelConversation,
    capture: _RequestCaptureState,
    response: ModelResponse,
    exc: BaseException,
) -> None:
    """Record full request, response, raw details, and projection stack."""
    log_event(
        LOGGER,
        logging.ERROR,
        "model.projection.failed",
        "Provider response could not be projected.",
        exception=exc,
        provider=provider,
        model=model,
        conversation=conversation,
        serialized_requests=capture.requests_for_logging(),
        provider_response=response,
    )


async def _capture_request_body(request: httpx.Request) -> None:
    """Capture browser-safe JSON plus error-only URL, headers, and body."""
    state = _ACTIVE_REQUEST_CAPTURE.get()
    if state is None:
        return

    body = await request.aread()
    wire_request = {
        "method": request.method,
        "url": str(request.url),
        "headers": _headers_for_logging(request.headers),
        "body": body.decode("utf-8", errors="replace"),
    }
    try:
        value = json.loads(
            body,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except TypeError, ValueError, UnicodeDecodeError:
        state.record_invalid(wire_request)
        return
    state.record(value, wire_request)


def _headers_for_logging(headers: httpx.Headers) -> dict[str, object]:
    """Preserve repeated provider headers in a redactable mapping."""
    grouped: dict[str, object] = {}
    for name, value in headers.multi_items():
        current = grouped.get(name)
        if current is None:
            grouped[name] = value
        elif isinstance(current, list):
            current.append(value)
        else:
            grouped[name] = [current, value]
    return grouped


def create_request_capture_client(
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.AsyncClient:
    """Create the client with a safe snapshot and error wire capture hook."""
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
        provider: str | None = None,
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
        self._provider_family = direct_model.system
        self._provider = provider or direct_model.system
        self._closed = False

    @property
    def model(self) -> str:
        """Return the exact configured model identity."""
        return self._model

    @property
    def provider(self) -> str:
        """Return the configured upstream provider identity for logs."""
        return self._provider

    async def generate(
        self,
        conversation: ModelConversation,
    ) -> ModelGeneration:
        """Perform one Direct request and project its captured safe result."""
        if self._closed:
            raise RuntimeError("model backend is closed")

        with bind_log_context(model=self._model, provider=self._provider):
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
                    _log_request_failure(
                        self._provider,
                        self._model,
                        conversation,
                        capture,
                        exc,
                    )
                    request_failed = True
                else:
                    request_failed = False

                if request_failed:
                    # Raise after logging so browser errors retain no provider
                    # exception, body, URL, or authentication data.
                    raise RuntimeError(
                        "Pydantic AI model request failed"
                    ) from None

                try:
                    snapshot = capture.require_snapshot()
                    assert response is not None
                    return _project_response(
                        response,
                        self._provider_family,
                        snapshot,
                    )
                except Exception as exc:
                    assert response is not None
                    _log_projection_failure(
                        self._provider,
                        self._model,
                        conversation,
                        capture,
                        response,
                        exc,
                    )
                    raise
            finally:
                # A later call must never inherit this call's wire details.
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
