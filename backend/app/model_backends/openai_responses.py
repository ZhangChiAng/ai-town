"""OpenAI Responses protocol adapter for protocol-neutral model calls."""

from collections.abc import Callable, Mapping
from typing import Any, Protocol

from app.model_backends.contracts import (
    JsonValue,
    ModelBackendSettings,
    ModelConversation,
    ModelGeneration,
    ModelReasoning,
    ModelUsage,
    PreparedModelRequest,
)

_MAX_OUTPUT_TOKENS = 2048
_MISSING = object()
_NO_DEFAULT = object()


class _ResponsesResource(Protocol):
    """Minimal Responses resource used by the adapter."""

    def create(self, **kwargs: Any) -> Any:
        """Create one model response."""


class _ResponsesClient(Protocol):
    """Minimal injectable OpenAI client surface."""

    responses: _ResponsesResource

    def close(self) -> None:
        """Release resources owned by the client."""


class OpenAIResponsesBackend:
    """Prepare and execute one configured OpenAI Responses model."""

    def __init__(self, model: str, client: _ResponsesClient) -> None:
        """Bind one SDK client to its exact configured model name."""
        self._client = client
        self._model = model

    @property
    def model(self) -> str:
        """Return the case-sensitive configured model name."""
        return self._model

    def prepare(
        self,
        conversation: ModelConversation,
    ) -> PreparedModelRequest:
        """Build one stateless, credential-free Responses payload."""
        response_input: list[JsonValue] = []
        for turn in conversation.turns:
            response_input.extend(
                (
                    _input_message("user", turn.input),
                    _input_message("assistant", turn.output),
                )
            )
        response_input.append(
            _input_message("user", conversation.current_input)
        )

        return PreparedModelRequest(
            payload={
                "model": self.model,
                "instructions": conversation.system_prompt,
                "input": response_input,
                # Private reasoning shares the provider's output allowance.
                "max_output_tokens": _MAX_OUTPUT_TOKENS,
                "store": False,
            }
        )

    def generate(
        self,
        prepared: PreparedModelRequest,
    ) -> ModelGeneration:
        """Call Responses once and strictly normalize the completed result."""
        try:
            response = self._client.responses.create(**prepared.payload)
        except Exception:
            # Do not expose provider response bodies or credentials upstream.
            request_failed = True
        else:
            request_failed = False

        if request_failed:
            raise RuntimeError("OpenAI Responses request failed")

        return _parse_response(response)

    def close(self) -> None:
        """Close the underlying SDK client."""
        self._client.close()


def create_openai_responses_backend(
    settings: ModelBackendSettings,
    /,
    client_factory: Callable[..., _ResponsesClient] | None = None,
) -> OpenAIResponsesBackend:
    """Create a no-retry Responses backend from resolved settings."""
    if client_factory is None:
        # Keep the concrete SDK dependency inside its adapter boundary.
        from openai import OpenAI

        client_factory = OpenAI

    try:
        client = client_factory(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=60.0,
            max_retries=0,
        )
    except Exception:
        client_creation_failed = True
    else:
        client_creation_failed = False

    if client_creation_failed:
        raise RuntimeError("OpenAI Responses client creation failed")
    return OpenAIResponsesBackend(settings.model, client)


def _input_message(role: str, text: str) -> dict[str, JsonValue]:
    """Build the sole text message shape used by Responses history."""
    return {
        "role": role,
        "content": [{"type": "input_text", "text": text}],
    }


def _parse_response(response: Any) -> ModelGeneration:
    """Extract one visible assistant result from a completed response."""
    if _field(response, "status") != "completed":
        raise ValueError("response was not completed")

    output = _field(response, "output")
    if type(output) is not list or not output:
        raise ValueError("expected response output items")

    messages: list[Any] = []
    reasoning: list[ModelReasoning] = []
    for item in output:
        item_type = _field(item, "type")
        if item_type == "message":
            messages.append(item)
        elif item_type == "reasoning":
            reasoning.extend(_parse_reasoning_item(item))
        else:
            raise ValueError("unexpected response output item")

    if len(messages) != 1 or _field(messages[0], "role") != "assistant":
        raise ValueError("expected exactly one assistant message")

    content = _field(messages[0], "content")
    if type(content) is not list or len(content) != 1:
        raise ValueError("expected exactly one output content block")
    output_text = content[0]
    if _field(output_text, "type") != "output_text":
        raise ValueError("expected one output_text block")

    text = _field(output_text, "text")
    if type(text) is not str or not text.strip():
        raise ValueError("visible text must not be blank")

    return ModelGeneration(
        content=text.strip(),
        reasoning=tuple(reasoning),
        usage=_parse_usage(_field(response, "usage")),
    )


def _parse_reasoning_item(item: Any) -> list[ModelReasoning]:
    """Expose readable reasoning without inspecting encrypted provider state."""
    summary = _field(item, "summary", [])
    content = _field(item, "content", None)
    if type(summary) is not list or (
        content is not None and type(content) is not list
    ):
        raise ValueError("invalid reasoning content")

    reasoning: list[ModelReasoning] = []
    reasoning.extend(_parse_reasoning_blocks(summary, "summary_text"))
    reasoning.extend(_parse_reasoning_blocks(content or [], "reasoning_text"))
    return reasoning


def _parse_reasoning_blocks(
    blocks: list[Any],
    expected_type: str,
) -> list[ModelReasoning]:
    """Validate one provider reasoning block collection in order."""
    reasoning: list[ModelReasoning] = []
    for block in blocks:
        if _field(block, "type") != expected_type:
            raise ValueError("unexpected reasoning block")
        text = _field(block, "text")
        if type(text) is not str:
            raise ValueError("invalid reasoning text")
        if text.strip():
            reasoning.append(ModelReasoning(type=expected_type, text=text))
    return reasoning


def _parse_usage(usage: Any) -> ModelUsage:
    """Partition total Responses input usage into neutral cache counts."""
    total_input = _token_count(_field(usage, "input_tokens"))
    output_tokens = _token_count(_field(usage, "output_tokens"))
    details = _field(usage, "input_tokens_details", _MISSING)
    if details is None or (
        details is not _MISSING and not _is_field_container(details)
    ):
        raise ValueError("invalid input token details")

    cache_write = _compatible_usage_count(
        usage,
        details,
        "cache_write_tokens",
    )
    cache_read = _compatible_usage_count(
        usage,
        details,
        "cached_tokens",
    )
    if cache_write + cache_read > total_input:
        raise ValueError("cache usage exceeds total input")

    total_tokens = _field(usage, "total_tokens", _MISSING)
    if total_tokens is not _MISSING and (
        _token_count(total_tokens) != total_input + output_tokens
    ):
        raise ValueError("total usage is inconsistent")

    return ModelUsage(
        input_tokens=total_input - cache_write - cache_read,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_write,
        cache_read_input_tokens=cache_read,
    )


def _compatible_usage_count(usage: Any, details: Any, name: str) -> int:
    """Read a cache count from either compatible location without conflict."""
    values: list[int] = []
    for container in (details, usage):
        if container is _MISSING:
            continue
        raw_value = _field(container, name, _MISSING)
        if raw_value is not _MISSING and raw_value is not None:
            values.append(_token_count(raw_value))
    if len(set(values)) > 1:
        raise ValueError("conflicting cache token usage")
    return values[0] if values else 0


def _token_count(value: Any) -> int:
    """Require a non-negative integer without bool or numeric coercion."""
    if type(value) is not int or value < 0:
        raise ValueError("invalid token count")
    return value


def _is_field_container(value: Any) -> bool:
    """Return whether fields can safely be read from a mapping or SDK model."""
    if isinstance(value, Mapping):
        return True
    try:
        vars(value)
    except AttributeError, TypeError:
        return False
    except Exception:
        raise ValueError("invalid response field container") from None
    return True


def _field(value: Any, name: str, default: Any = _NO_DEFAULT) -> Any:
    """Read one mapping or SDK-model field without coercing its value."""
    try:
        if isinstance(value, Mapping):
            if name in value:
                return value[name]
        else:
            return getattr(value, name)
    except AttributeError:
        pass
    except Exception:
        # Response attribute access is parsing, never another upstream call.
        raise ValueError("invalid response field") from None
    if default is not _NO_DEFAULT:
        return default
    raise ValueError(f"missing response field: {name}")
