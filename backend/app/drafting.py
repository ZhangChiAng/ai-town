"""Model request construction and browser-only two-layer draft generation."""

import hashlib
import hmac
import json
import logging
from collections.abc import Callable
from copy import deepcopy
from typing import Any, Literal, Protocol
from uuid import UUID, uuid4

from app.config import ModelSettings
from app.models import (
    AgentId,
    ConfirmLayerRequest,
    InnerTurn,
    InvalidLayerOutputError,
    Layer,
    LayerDraftResponse,
    ModelReasoningBlock,
    ModelRequestPreviewResponse,
    OuterTurn,
    Scene,
    SceneConflictError,
    TokenUsage,
    build_inner_input,
    build_outer_input,
    confirm_inner_turn,
    confirm_outer_turn,
    get_agent,
    parse_addressed_message,
)

LOGGER = logging.getLogger(__name__)
CACHE_CONTROL = {"type": "ephemeral", "ttl": "5m"}
MAX_TOKENS = 1024
RESPONSES_MAX_OUTPUT_TOKENS = 2048
ModelProtocol = Literal["anthropic", "responses"]
_NO_DEFAULT = object()
_MISSING = object()


class MessagesResource(Protocol):
    """Minimal Anthropic Messages API surface used by the service."""

    def create(self, **kwargs: Any) -> Any:
        """Create one model message."""


class ResponsesResource(Protocol):
    """Minimal OpenAI Responses API surface used by the service."""

    def create(self, **kwargs: Any) -> Any:
        """Create one model response."""


class ModelClient(Protocol):
    """Shared lifecycle surface for either provider SDK client."""

    def close(self) -> None:
        """Close network resources."""


class AnthropicClient(ModelClient, Protocol):
    """Minimal injectable Anthropic client surface."""

    messages: MessagesResource


class ResponsesClient(ModelClient, Protocol):
    """Minimal injectable OpenAI Responses client surface."""

    responses: ResponsesResource


class DraftGenerationError(RuntimeError):
    """Raised when an upstream response cannot produce a safe draft."""


def select_model_protocol(model: str) -> ModelProtocol:
    """Select the provider protocol solely from the configured model name."""
    if "claude" in model.casefold():
        return "anthropic"
    return "responses"


def create_anthropic_client(
    settings: ModelSettings,
    client_factory: Callable[..., AnthropicClient] | None = None,
) -> AnthropicClient:
    """Create an Anthropic client with bounded requests and no retries."""
    if client_factory is None:
        # Delay SDK initialization so isolated tests can use a tiny fake.
        from anthropic import Anthropic

        client_factory = Anthropic

    return client_factory(
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=60.0,
        max_retries=0,
    )


def create_responses_client(
    settings: ModelSettings,
    client_factory: Callable[..., ResponsesClient] | None = None,
) -> ResponsesClient:
    """Create an OpenAI client with bounded requests and no retries."""
    if client_factory is None:
        # The SDK appends ``/responses`` to this unmodified API root.
        from openai import OpenAI

        client_factory = OpenAI

    return client_factory(
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=60.0,
        max_retries=0,
    )


class LayerDraftService:
    """Build, generate, preview, and confirm isolated persona-layer calls."""

    def __init__(
        self,
        client: AnthropicClient | ResponsesClient,
        model: str,
    ) -> None:
        """Initialize the service with one concrete model and SDK client."""
        self._client = client
        self._model = model
        self._protocol = select_model_protocol(model)

    @property
    def model(self) -> str:
        """Return the concrete model served by this immutable service."""
        return self._model

    @property
    def protocol(self) -> ModelProtocol:
        """Return the upstream protocol served by this service."""
        return self._protocol

    def preview(
        self,
        scene: Scene,
        agent_id: AgentId,
        layer: Layer,
    ) -> ModelRequestPreviewResponse:
        """Return the exact next provider payload without calling the model."""
        event_id, _input = build_layer_input(scene, agent_id, layer)
        return ModelRequestPreviewResponse(
            layer=layer,
            event_id=event_id,
            request=build_model_request(
                scene,
                agent_id,
                layer,
                self._model,
            ),
        )

    def generate(
        self,
        scene: Scene,
        agent_id: AgentId,
        layer: Layer,
    ) -> LayerDraftResponse:
        """Call the model exactly once and return an unpersisted draft."""
        event_id, _input = build_layer_input(scene, agent_id, layer)
        model_request = build_model_request(
            scene,
            agent_id,
            layer,
            self._model,
        )

        try:
            if self._protocol == "anthropic":
                response = self._client.messages.create(**model_request)
            else:
                response = self._client.responses.create(**model_request)
        except Exception as error:
            # Provider bodies and credentials must not escape this boundary.
            raise DraftGenerationError("Model request failed.") from error

        try:
            if self._protocol == "anthropic":
                content, reasoning, usage = _parse_anthropic_response(response)
            else:
                content, reasoning, usage = _parse_responses_response(response)
            if layer == "outer":
                parse_addressed_message(content, agent_id)
        except (AttributeError, TypeError, ValueError, KeyError) as error:
            raise DraftGenerationError(
                f"Model returned an invalid {layer} draft."
            ) from error

        result = LayerDraftResponse(
            layer=layer,
            call_id=uuid4(),
            event_id=event_id,
            content=content,
            reasoning=reasoning,
            usage=usage,
            request_snapshot=deepcopy(model_request),
            state_token=request_state_token(model_request),
        )
        _log_usage(scene.id, agent_id, layer, self._model, usage)
        return result

    def confirm(
        self,
        scene: Scene,
        agent_id: AgentId,
        layer: Layer,
        confirmation: ConfirmLayerRequest,
    ) -> Scene:
        """Persist a confirmed draft without issuing a model request."""
        event_id, actual_input = build_layer_input(scene, agent_id, layer)
        if confirmation.event_id != event_id:
            raise SceneConflictError("The draft event is no longer current.")

        current_request = build_model_request(
            scene,
            agent_id,
            layer,
            self._model,
        )
        current_token = request_state_token(current_request)
        if not hmac.compare_digest(
            confirmation.state_token,
            current_token,
        ):
            raise SceneConflictError(
                "The scene changed after this draft was generated."
            )

        if layer == "inner":
            return confirm_inner_turn(
                scene,
                agent_id,
                confirmation,
                actual_input,
            )

        try:
            parse_addressed_message(confirmation.content, agent_id)
        except ValueError as error:
            raise InvalidLayerOutputError(
                "Outer output must be one non-self 'To X: body' line."
            ) from error
        return confirm_outer_turn(
            scene,
            agent_id,
            confirmation,
            actual_input,
        )


def build_layer_input(
    scene: Scene,
    agent_id: AgentId,
    layer: Layer,
) -> tuple[UUID, str]:
    """Build the selected layer's exact current event ID and user text."""
    if layer == "inner":
        return build_inner_input(scene, agent_id)
    return build_outer_input(scene, agent_id)


def build_model_request(
    scene: Scene,
    agent_id: AgentId,
    layer: Layer,
    model: str,
) -> dict[str, Any]:
    """Build the deterministic full-history request for the model protocol."""
    if select_model_protocol(model) == "anthropic":
        return build_anthropic_request(scene, agent_id, layer, model)
    return build_responses_request(scene, agent_id, layer, model)


def _build_layer_context(
    scene: Scene,
    agent_id: AgentId,
    layer: Layer,
) -> tuple[str, list[InnerTurn | OuterTurn], str]:
    """Return only the selected persona's prompt, history, and next input."""
    agent = get_agent(scene, agent_id)
    _event_id, current_input = build_layer_input(scene, agent_id, layer)
    if layer == "inner":
        system_prompt = agent.inner_context.system_prompt
        turns: list[InnerTurn | OuterTurn] = list(agent.inner_context.turns)
    else:
        system_prompt = agent.outer_context.system_prompt
        turns = list(agent.outer_context.turns)
    return system_prompt, turns, current_input


def build_anthropic_request(
    scene: Scene,
    agent_id: AgentId,
    layer: Layer,
    model: str,
) -> dict[str, Any]:
    """Build an Anthropic request with two five-minute cache breakpoints."""
    system_prompt, turns, current_input = _build_layer_context(
        scene,
        agent_id,
        layer,
    )

    messages: list[dict[str, Any]] = []
    for turn in turns:
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
        # Each layer advances its own rolling immutable history breakpoint.
        messages[-1]["content"][-1]["cache_control"] = dict(CACHE_CONTROL)
    messages.append(
        {
            "role": "user",
            "content": [{"type": "text", "text": current_input}],
        }
    )

    return {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": dict(CACHE_CONTROL),
            }
        ],
        "messages": messages,
    }


def build_responses_request(
    scene: Scene,
    agent_id: AgentId,
    layer: Layer,
    model: str,
) -> dict[str, Any]:
    """Build a stateless Responses request without Anthropic metadata."""
    system_prompt, turns, current_input = _build_layer_context(
        scene,
        agent_id,
        layer,
    )
    response_input: list[dict[str, Any]] = []
    for turn in turns:
        response_input.extend(
            (
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": turn.input}],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "input_text", "text": turn.output}],
                },
            )
        )
    response_input.append(
        {
            "role": "user",
            "content": [{"type": "input_text", "text": current_input}],
        }
    )
    return {
        "model": model,
        "instructions": system_prompt,
        "input": response_input,
        # Private reasoning consumes the same Responses output budget.
        "max_output_tokens": RESPONSES_MAX_OUTPUT_TOKENS,
        "store": False,
    }


def request_state_token(request: dict[str, Any]) -> str:
    """Hash the exact request so confirmation can detect stale browser state."""
    serialized = json.dumps(
        request,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _parse_anthropic_response(
    response: Any,
) -> tuple[str, list[ModelReasoningBlock], TokenUsage]:
    """Extract visible text, readable thinking, and Anthropic usage."""
    content = _field(response, "content")
    if not isinstance(content, list) or not content:
        raise ValueError("expected response blocks")

    allowed_types = {"text", "thinking", "redacted_thinking"}
    if any(
        _field(block, "type", None) not in allowed_types for block in content
    ):
        raise ValueError("unexpected response block")
    text_blocks = [
        block for block in content if _field(block, "type", None) == "text"
    ]
    if len(text_blocks) != 1:
        raise ValueError("expected exactly one visible text block")

    content_text = _validate_visible_text(_field(text_blocks[0], "text"))
    reasoning = _parse_anthropic_reasoning(content)
    usage = _parse_anthropic_usage(_field(response, "usage"))
    return content_text, reasoning, usage


def _parse_responses_response(
    response: Any,
) -> tuple[str, list[ModelReasoningBlock], TokenUsage]:
    """Extract one strict assistant output from a completed Response."""
    if _field(response, "status") != "completed":
        raise ValueError("response was not completed")

    output = _field(response, "output")
    if not isinstance(output, list) or not output:
        raise ValueError("expected response output items")
    if any(
        _field(item, "type", None) not in {"reasoning", "message"}
        for item in output
    ):
        raise ValueError("unexpected response output item")

    messages = [
        item for item in output if _field(item, "type", None) == "message"
    ]
    if len(messages) != 1 or _field(messages[0], "role", None) != "assistant":
        raise ValueError("expected exactly one assistant message")

    content = _field(messages[0], "content")
    if not isinstance(content, list) or len(content) != 1:
        raise ValueError("expected exactly one output content block")
    output_text = content[0]
    if _field(output_text, "type", None) != "output_text":
        raise ValueError("expected one output_text block")

    content_text = _validate_visible_text(_field(output_text, "text"))
    reasoning = _parse_responses_reasoning(output)
    usage = _parse_responses_usage(_field(response, "usage"))
    return content_text, reasoning, usage


def _validate_visible_text(value: Any) -> str:
    """Require one non-blank provider-visible text response."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("visible text must not be blank")
    return value.strip()


def _parse_anthropic_reasoning(
    content: list[Any],
) -> list[ModelReasoningBlock]:
    """Extract thinking without exposing signatures or redacted data."""
    reasoning: list[ModelReasoningBlock] = []
    for block in content:
        if _field(block, "type", None) != "thinking":
            continue
        text = _field(block, "thinking")
        if not isinstance(text, str):
            raise ValueError("invalid thinking text")
        if text.strip():
            reasoning.append(ModelReasoningBlock(type="thinking", text=text))
    return reasoning


def _parse_responses_reasoning(
    output: list[Any],
) -> list[ModelReasoningBlock]:
    """Extract readable reasoning summaries and text in provider order."""
    reasoning: list[ModelReasoningBlock] = []
    for item in output:
        if _field(item, "type", None) != "reasoning":
            continue

        summary = _field(item, "summary", [])
        content = _field(item, "content", None)
        if not isinstance(summary, list) or (
            content is not None and not isinstance(content, list)
        ):
            raise ValueError("invalid reasoning content")

        for block in [*summary, *(content or [])]:
            block_type = _field(block, "type", None)
            if block_type not in {"summary_text", "reasoning_text"}:
                raise ValueError("unexpected reasoning block")
            text = _field(block, "text")
            if not isinstance(text, str):
                raise ValueError("invalid reasoning text")
            if text.strip():
                reasoning.append(
                    ModelReasoningBlock(type=block_type, text=text)
                )
    return reasoning


def _parse_anthropic_usage(usage: Any) -> TokenUsage:
    """Validate Anthropic's already partitioned token accounting."""
    return TokenUsage(
        input_tokens=_token_count(_field(usage, "input_tokens")),
        output_tokens=_token_count(_field(usage, "output_tokens")),
        cache_creation_input_tokens=_token_count(
            _field(usage, "cache_creation_input_tokens", 0)
        ),
        cache_read_input_tokens=_token_count(
            _field(usage, "cache_read_input_tokens", 0)
        ),
    )


def _parse_responses_usage(usage: Any) -> TokenUsage:
    """Map Responses usage to the protocol-neutral token partition."""
    total_input = _token_count(_field(usage, "input_tokens"))
    output_tokens = _token_count(_field(usage, "output_tokens"))
    details = _field(usage, "input_tokens_details", _MISSING)
    if details is None:
        raise ValueError("invalid input token details")

    cache_write = _optional_detail_count(usage, details, "cache_write_tokens")
    cache_read = _optional_detail_count(usage, details, "cached_tokens")
    if cache_write + cache_read > total_input:
        raise ValueError("cache usage exceeds total input")

    total_tokens = _field(usage, "total_tokens", _MISSING)
    if (
        total_tokens is not _MISSING
        and _token_count(total_tokens) != total_input + output_tokens
    ):
        raise ValueError("total usage is inconsistent")

    return TokenUsage(
        input_tokens=total_input - cache_write - cache_read,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_write,
        cache_read_input_tokens=cache_read,
    )


def _optional_detail_count(usage: Any, details: Any, name: str) -> int:
    """Read one nullable cache count and reject conflicting locations."""
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
    """Require a non-negative integer token count without coercion."""
    if type(value) is not int or value < 0:
        raise ValueError("invalid token count")
    return value


def _field(value: Any, name: str, default: Any = _NO_DEFAULT) -> Any:
    """Read an SDK model or mapping field without normalizing its value."""
    if isinstance(value, dict):
        if name in value:
            return value[name]
    elif hasattr(value, name):
        return getattr(value, name)
    if default is not _NO_DEFAULT:
        return default
    raise AttributeError(name)


def _log_usage(
    scene_id: UUID,
    agent_id: AgentId,
    layer: Layer,
    model: str,
    usage: TokenUsage,
) -> None:
    """Log non-sensitive request identity and usage as structured JSON."""
    LOGGER.info(
        json.dumps(
            {
                "event": "layer_draft_generated",
                "scene_id": str(scene_id),
                "agent_id": agent_id,
                "layer": layer,
                "model": model,
                **usage.model_dump(),
            },
            sort_keys=True,
        )
    )
