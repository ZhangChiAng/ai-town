"""Provider-independent generation of one-step message drafts."""

import json
import logging
from collections.abc import Callable
from typing import Any, Literal, Protocol, TypedDict
from uuid import UUID

from app.config import ModelSettings
from app.models import (
    AGENT_IDS,
    Agent,
    AgentId,
    MessageDraftResponse,
    MessageDraftUsage,
    ModelReasoningBlock,
    Scene,
    parse_addressed_message,
)

LOGGER = logging.getLogger(__name__)
CACHE_CONTROL = {"type": "ephemeral", "ttl": "5m"}
RESPONSES_MAX_OUTPUT_TOKENS = 2048
RUNTIME_TURN_PROMPT = (
    "现在轮到你说话。请从另外两位 Agent（{recipient_ids}）中选择一位接收人，"
    "只输出一行 `To X: 消息正文`，其中 X 必须替换为所选接收人的 Agent ID。"
)
ModelProtocol = Literal["anthropic", "responses"]
_NO_DEFAULT = object()
_MISSING = object()


class _ContextBlock(TypedDict):
    """One provider-independent text block."""

    text: str
    cache_breakpoint: bool


class _ContextTurn(TypedDict):
    """One provider-independent alternating conversation turn."""

    role: Literal["user", "assistant"]
    content: list[_ContextBlock]


class MessagesResource(Protocol):
    """Minimal Anthropic Messages API surface used by the service."""

    def create(self, **kwargs: Any) -> Any:
        """Create one Anthropic message."""


class ResponsesResource(Protocol):
    """Minimal OpenAI Responses API surface used by the service."""

    def create(self, **kwargs: Any) -> Any:
        """Create one OpenAI-compatible response."""


class ModelClient(Protocol):
    """Shared lifecycle surface for either provider SDK client."""

    def close(self) -> None:
        """Close the client's network resources."""


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
        # Delay SDK imports so tests with fake clients stay provider-neutral.
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


class MessageDraftService:
    """Generate plain-text message drafts without mutating scene state."""

    def __init__(
        self,
        client: AnthropicClient | ResponsesClient,
        model: str,
    ) -> None:
        """Initialize the service with the model-selected SDK client."""
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

    def preview(self, scene: Scene, agent_id: AgentId) -> dict[str, Any]:
        """Build the exact next provider payload without calling the model."""
        return build_message_request(scene, agent_id, self._model)

    def generate(
        self,
        scene: Scene,
        agent_id: AgentId,
    ) -> MessageDraftResponse:
        """Generate one draft from the selected Agent's private context.

        Args:
            scene: The persisted scene snapshot to read.
            agent_id: Agent whose next private message should be drafted.

        Returns:
            A validated text draft and protocol-neutral usage metrics.

        Raises:
            DraftGenerationError: If the upstream request fails or returns an
                invalid response.
        """
        model_request = build_message_request(scene, agent_id, self._model)

        try:
            if self._protocol == "anthropic":
                response = self._client.messages.create(**model_request)
            else:
                response = self._client.responses.create(**model_request)
        except Exception as error:
            # Do not propagate provider response bodies or credentials.
            raise DraftGenerationError("Model request failed.") from error

        try:
            if self._protocol == "anthropic":
                draft = _parse_anthropic_response(
                    response, model_request, agent_id
                )
            else:
                draft = _parse_responses_response(
                    response, model_request, agent_id
                )
        except (AttributeError, TypeError, ValueError, KeyError) as error:
            raise DraftGenerationError(
                "Model returned an invalid message draft."
            ) from error

        _log_usage(scene.id, agent_id, self._model, draft.usage)
        return draft


def build_message_request(
    scene: Scene,
    agent_id: AgentId,
    model: str,
) -> dict[str, Any]:
    """Build the deterministic request selected by the model name."""
    if select_model_protocol(model) == "anthropic":
        return build_anthropic_request(scene, agent_id, model)
    return build_responses_request(scene, agent_id, model)


def build_anthropic_request(
    scene: Scene,
    agent_id: AgentId,
    model: str,
) -> dict[str, Any]:
    """Build a deterministic Anthropic request with explicit cache points."""
    system_prompt, turns = _build_model_context(scene, agent_id)
    messages: list[dict[str, Any]] = []
    for turn in turns:
        content: list[dict[str, Any]] = []
        for context_block in turn["content"]:
            block = {"type": "text", "text": context_block["text"]}
            if context_block["cache_breakpoint"]:
                block["cache_control"] = dict(CACHE_CONTROL)
            content.append(block)
        messages.append({"role": turn["role"], "content": content})

    return {
        "model": model,
        "max_tokens": 512,
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
    model: str,
) -> dict[str, Any]:
    """Build a deterministic, stateless OpenAI Responses request."""
    system_prompt, turns = _build_model_context(scene, agent_id)
    response_input = [
        {
            "role": turn["role"],
            "content": [
                {"type": "input_text", "text": block["text"]}
                for block in turn["content"]
            ],
        }
        for turn in turns
    ]
    return {
        "model": model,
        "instructions": system_prompt,
        "input": response_input,
        # Responses counts private reasoning against the same output budget.
        "max_output_tokens": RESPONSES_MAX_OUTPUT_TOKENS,
        "store": False,
    }


def _build_model_context(
    scene: Scene,
    agent_id: AgentId,
) -> tuple[str, list[_ContextTurn]]:
    """Build shared system text and alternating turns for either protocol."""
    agent = next(agent for agent in scene.agents if agent.id == agent_id)
    turns = _build_timeline_turns(agent)
    if turns:
        # Only the final persisted record is eligible for explicit caching.
        turns[-1]["content"][-1]["cache_breakpoint"] = True

    recipient_ids = "、".join(
        candidate for candidate in AGENT_IDS if candidate != agent_id
    )
    runtime_block: _ContextBlock = {
        "text": RUNTIME_TURN_PROMPT.format(recipient_ids=recipient_ids),
        "cache_breakpoint": False,
    }
    if turns and turns[-1]["role"] == "user":
        # Preserve alternation while keeping this as a distinct text block.
        turns[-1]["content"].append(runtime_block)
    else:
        turns.append({"role": "user", "content": [runtime_block]})
    return agent.system_prompt, turns


def _build_timeline_turns(agent: Agent) -> list[_ContextTurn]:
    """Map the private timeline to alternating provider-neutral turns."""
    turns: list[_ContextTurn] = []
    for record in agent.timeline:
        role: Literal["user", "assistant"] = (
            "user" if record.direction == "received" else "assistant"
        )
        block: _ContextBlock = {
            "text": record.content,
            "cache_breakpoint": False,
        }
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"].append(block)
        else:
            turns.append({"role": role, "content": [block]})
    return turns


def _parse_anthropic_response(
    response: Any,
    request_snapshot: dict[str, Any],
    sender_id: AgentId,
) -> MessageDraftResponse:
    """Extract one Anthropic text block while allowing thinking blocks."""
    content = _field(response, "content")
    if not isinstance(content, list) or not content:
        raise ValueError("expected response blocks")

    allowed_block_types = {"text", "thinking", "redacted_thinking"}
    if any(
        _field(block, "type", None) not in allowed_block_types
        for block in content
    ):
        raise ValueError("unexpected response block")

    # Thinking is observer-visible but remains separate from the editable text.
    reasoning = _parse_anthropic_reasoning(content)
    text_blocks = [
        block for block in content if _field(block, "type", None) == "text"
    ]
    if len(text_blocks) != 1:
        raise ValueError("expected exactly one text response")

    visible_content = _validate_visible_content(
        _field(text_blocks[0], "text"), sender_id
    )
    usage = _parse_anthropic_usage(_field(response, "usage"))
    return MessageDraftResponse(
        content=visible_content,
        reasoning=reasoning,
        usage=usage,
        request_snapshot=request_snapshot,
    )


def _parse_responses_response(
    response: Any,
    request_snapshot: dict[str, Any],
    sender_id: AgentId,
) -> MessageDraftResponse:
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

    visible_content = _validate_visible_content(
        _field(output_text, "text"), sender_id
    )
    reasoning = _parse_responses_reasoning(output)
    usage = _parse_responses_usage(_field(response, "usage"))
    return MessageDraftResponse(
        content=visible_content,
        reasoning=reasoning,
        usage=usage,
        request_snapshot=request_snapshot,
    )


def _parse_anthropic_reasoning(content: list[Any]) -> list[ModelReasoningBlock]:
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


def _parse_responses_reasoning(output: list[Any]) -> list[ModelReasoningBlock]:
    """Extract readable reasoning summaries and raw text in response order."""
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


def _validate_visible_content(content: Any, sender_id: AgentId) -> str:
    """Validate and return one addressed, user-visible output line."""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("invalid message draft")
    if "\n" in content or "\r" in content:
        raise ValueError("message draft must be one line")
    visible_content = content.strip()
    recipient_id, _body = parse_addressed_message(visible_content, sender_id)
    if recipient_id not in AGENT_IDS:
        raise ValueError("invalid message recipient")
    return visible_content


def _parse_anthropic_usage(usage: Any) -> MessageDraftUsage:
    """Validate Anthropic's already-partitioned token accounting."""
    return MessageDraftUsage(
        input_tokens=_token_count(_field(usage, "input_tokens")),
        output_tokens=_token_count(_field(usage, "output_tokens")),
        cache_creation_input_tokens=_token_count(
            _field(usage, "cache_creation_input_tokens")
        ),
        cache_read_input_tokens=_token_count(
            _field(usage, "cache_read_input_tokens")
        ),
    )


def _parse_responses_usage(usage: Any) -> MessageDraftUsage:
    """Map Responses usage to the public protocol-neutral partition."""
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

    return MessageDraftUsage(
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
        # Compatible endpoints may serialize unavailable counts as null.
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
    model: str,
    usage: MessageDraftUsage,
) -> None:
    """Log non-sensitive request identity and token usage as JSON."""
    LOGGER.info(
        json.dumps(
            {
                "event": "message_draft_generated",
                "scene_id": str(scene_id),
                "agent_id": agent_id,
                "model": model,
                **usage.model_dump(),
            },
            sort_keys=True,
        )
    )
