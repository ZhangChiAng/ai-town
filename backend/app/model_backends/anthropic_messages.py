"""Anthropic Messages adapter for the protocol-neutral model backend port."""

from collections.abc import Callable
from typing import Any, Protocol

from app.model_backends.contracts import (
    ModelBackendSettings,
    ModelConversation,
    ModelGeneration,
    ModelReasoning,
    ModelUsage,
    PreparedModelRequest,
)

_CACHE_CONTROL = {"type": "ephemeral", "ttl": "5m"}
_MAX_TOKENS = 1024
_MISSING = object()


class _MessagesResource(Protocol):
    """Minimal Anthropic Messages resource used by the adapter."""

    def create(self, **kwargs: Any) -> Any:
        """Create one upstream message."""


class _AnthropicClient(Protocol):
    """Minimal injectable Anthropic client surface."""

    messages: _MessagesResource

    def close(self) -> None:
        """Release resources owned by the SDK client."""


_ClientFactory = Callable[..., _AnthropicClient]


class AnthropicMessagesBackend:
    """Translate neutral conversations to Anthropic Messages calls."""

    def __init__(self, model: str, client: _AnthropicClient) -> None:
        """Store one immutable model binding and its owned client."""
        self._model = model
        self._client = client

    @property
    def model(self) -> str:
        """Return the configured model name exactly."""
        return self._model

    def prepare(
        self,
        conversation: ModelConversation,
    ) -> PreparedModelRequest:
        """Build a credential-free Anthropic payload without performing I/O."""
        messages: list[dict[str, Any]] = []
        for turn in conversation.turns:
            messages.extend(
                (
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": turn.input}],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": turn.output}],
                    },
                )
            )

        if messages:
            # Advance the rolling cache boundary to the last confirmed output.
            messages[-1]["content"][-1]["cache_control"] = dict(_CACHE_CONTROL)
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": conversation.current_input}
                ],
            }
        )

        return PreparedModelRequest(
            payload={
                "model": self.model,
                "max_tokens": _MAX_TOKENS,
                "system": [
                    {
                        "type": "text",
                        "text": conversation.system_prompt,
                        "cache_control": dict(_CACHE_CONTROL),
                    }
                ],
                "messages": messages,
            }
        )

    def generate(
        self,
        prepared: PreparedModelRequest,
    ) -> ModelGeneration:
        """Make exactly one upstream call and validate its safe result."""
        try:
            response = self._client.messages.create(**prepared.payload)
        except Exception:
            # Provider exceptions can contain request bodies, URLs, or secrets.
            request_failed = True
        else:
            request_failed = False

        if request_failed:
            raise RuntimeError("Anthropic Messages request failed")
        return _parse_response(response)

    def close(self) -> None:
        """Close the owned Anthropic SDK client."""
        self._client.close()


def create_anthropic_messages_backend(
    settings: ModelBackendSettings,
    /,
    client_factory: _ClientFactory | None = None,
) -> AnthropicMessagesBackend:
    """Create an Anthropic Messages backend with bounded client behavior."""
    if client_factory is None:
        # Keep the provider SDK isolated to this concrete adapter module.
        from anthropic import Anthropic

        client_factory = Anthropic

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
        raise RuntimeError("Anthropic Messages client creation failed")
    return AnthropicMessagesBackend(settings.model, client)


def _parse_response(response: Any) -> ModelGeneration:
    """Extract one visible text, safe thinking, and strict token usage."""
    content = _field(response, "content")
    if type(content) is not list or not content:
        raise ValueError("invalid Anthropic response content")

    allowed_types = {"text", "thinking", "redacted_thinking"}
    block_types = [_field(block, "type") for block in content]
    if any(
        type(block_type) is not str or block_type not in allowed_types
        for block_type in block_types
    ):
        raise ValueError("unexpected Anthropic response block")

    text_blocks = [
        block
        for block, block_type in zip(content, block_types, strict=True)
        if block_type == "text"
    ]
    if len(text_blocks) != 1:
        raise ValueError("expected exactly one Anthropic text block")

    visible_text = _field(text_blocks[0], "text")
    if type(visible_text) is not str or not visible_text.strip():
        raise ValueError("invalid Anthropic visible text")

    reasoning: list[ModelReasoning] = []
    for block, block_type in zip(content, block_types, strict=True):
        if block_type != "thinking":
            continue
        thinking = _field(block, "thinking")
        if type(thinking) is not str:
            raise ValueError("invalid Anthropic thinking text")
        if thinking.strip():
            # Deliberately ignore signatures and all redacted block contents.
            reasoning.append(ModelReasoning(type="thinking", text=thinking))

    usage = _field(response, "usage")
    return ModelGeneration(
        content=visible_text.strip(),
        reasoning=tuple(reasoning),
        usage=ModelUsage(
            input_tokens=_token_count(_field(usage, "input_tokens")),
            output_tokens=_token_count(_field(usage, "output_tokens")),
            cache_creation_input_tokens=_token_count(
                _field(usage, "cache_creation_input_tokens", 0)
            ),
            cache_read_input_tokens=_token_count(
                _field(usage, "cache_read_input_tokens", 0)
            ),
        ),
    )


def _token_count(value: Any) -> int:
    """Require an exact non-negative integer token count."""
    if type(value) is not int or value < 0:
        raise ValueError("invalid Anthropic token count")
    return value


def _field(value: Any, name: str, default: Any = _MISSING) -> Any:
    """Read one SDK object or dictionary field without coercion."""
    if type(value) is dict and name in value:
        return value[name]
    try:
        return getattr(value, name)
    except AttributeError:
        if default is not _MISSING:
            return default
        raise ValueError("invalid Anthropic response shape") from None
    except Exception:
        # Treat hostile or malformed SDK attribute access as parse failure.
        raise ValueError("invalid Anthropic response shape") from None
