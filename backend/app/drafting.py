"""Anthropic-backed generation of one-step message drafts."""

import json
import logging
from collections.abc import Callable
from typing import Any, Protocol
from uuid import UUID

from app.config import ModelSettings
from app.models import (
    Agent,
    AgentId,
    MessageDraftResponse,
    MessageDraftUsage,
    Scene,
)

LOGGER = logging.getLogger(__name__)
COMPOSE_MESSAGE_TOOL = "compose_message"
CACHE_CONTROL = {"type": "ephemeral", "ttl": "5m"}


class MessagesResource(Protocol):
    """Minimal Messages API surface used by MessageDraftService."""

    def create(self, **kwargs: Any) -> Any:
        """Create one Anthropic message."""


class AnthropicClient(Protocol):
    """Minimal injectable Anthropic client surface."""

    messages: MessagesResource

    def close(self) -> None:
        """Close the client's network resources."""


class DraftGenerationError(RuntimeError):
    """Raised when an upstream response cannot produce a safe draft."""


def create_anthropic_client(
    settings: ModelSettings,
    client_factory: Callable[..., AnthropicClient] | None = None,
) -> AnthropicClient:
    """Create an Anthropic client with bounded requests and no retries."""
    if client_factory is None:
        # Delay the SDK import so fake-client tests do not initialize its
        # separate HTTP and async-library stack.
        from anthropic import Anthropic

        client_factory = Anthropic

    return client_factory(
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=60.0,
        max_retries=0,
    )


class MessageDraftService:
    """Generate structured message drafts without mutating scene state."""

    def __init__(
        self,
        client: AnthropicClient,
        model: str,
    ) -> None:
        """Initialize the service with an injectable Anthropic client."""
        self._client = client
        self._model = model

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
            A validated structured draft and the upstream usage metrics.

        Raises:
            DraftGenerationError: If the upstream request fails or returns an
                invalid tool result.
        """
        model_request = build_message_request(scene, agent_id, self._model)

        try:
            response = self._client.messages.create(**model_request)
        except Exception as error:
            # Do not propagate provider response bodies or credentials.
            raise DraftGenerationError("Model request failed.") from error

        try:
            draft = _parse_response(response, agent_id, model_request)
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
    """Build a deterministic, explicitly cached Anthropic request."""
    agent = next(agent for agent in scene.agents if agent.id == agent_id)
    recipients = [agent for agent in scene.agents if agent.id != agent_id]
    messages = _build_timeline_messages(agent)
    if messages:
        # The rolling breakpoint follows the final immutable timeline record.
        messages[-1]["content"][-1]["cache_control"] = dict(CACHE_CONTROL)

    return {
        "model": model,
        "max_tokens": 512,
        "system": [
            {
                "type": "text",
                "text": agent.system_prompt,
                "cache_control": dict(CACHE_CONTROL),
            }
        ],
        "tools": [_compose_message_tool(recipients)],
        "tool_choice": {
            "type": "tool",
            "name": COMPOSE_MESSAGE_TOOL,
            "disable_parallel_tool_use": True,
        },
        "messages": messages,
    }


def _compose_message_tool(recipients: list[Agent]) -> dict[str, Any]:
    """Return the strict tool schema limited to the other two agents."""
    recipient_ids = [agent.id for agent in recipients]
    return {
        "name": COMPOSE_MESSAGE_TOOL,
        "description": (
            "Submit the one private message this Agent chooses to send now. "
            "Choose one available recipient and provide non-empty message "
            "content written from the selected Agent's perspective."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "recipient_id": {
                    "type": "string",
                    "enum": recipient_ids,
                    "description": "ID of the Agent receiving the message.",
                },
                "content": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Non-empty private message body.",
                },
            },
            "required": ["recipient_id", "content"],
            "additionalProperties": False,
        },
    }


def _build_timeline_messages(agent: Agent) -> list[dict[str, Any]]:
    """Map the private timeline to alternating native Messages API turns."""
    messages: list[dict[str, Any]] = []
    for record in agent.timeline:
        if record.type == "inner_voice":
            role = "user"
            text = f"内心的声音：{record.content}"
        else:
            is_received = record.direction == "received"
            role = "user" if is_received else "assistant"
            prefix = "From" if is_received else "To"
            text = f"{prefix} {record.counterpart_id}: {record.content}"

        block = {"type": "text", "text": text}
        if messages and messages[-1]["role"] == role:
            # Some compatible gateways require strict role alternation even
            # though Anthropic itself combines adjacent same-role messages.
            messages[-1]["content"].append(block)
        else:
            messages.append({"role": role, "content": [block]})
    return messages


def _parse_response(
    response: Any,
    agent_id: AgentId,
    request_snapshot: dict[str, Any],
) -> MessageDraftResponse:
    """Require exactly one valid compose_message tool-use block."""
    content = response.content
    if not isinstance(content, list) or len(content) != 1:
        raise ValueError("expected exactly one response block")

    block = content[0]
    if (
        getattr(block, "type", None) != "tool_use"
        or getattr(block, "name", None) != COMPOSE_MESSAGE_TOOL
    ):
        raise ValueError("expected compose_message tool use")

    tool_input = block.input
    if not isinstance(tool_input, dict) or set(tool_input) != {
        "recipient_id",
        "content",
    }:
        raise ValueError("invalid tool input")

    recipient_id = tool_input["recipient_id"]
    content_value = tool_input["content"]
    if (
        recipient_id not in {"A", "B", "C"}
        or recipient_id == agent_id
        or not isinstance(content_value, str)
        or not content_value.strip()
    ):
        raise ValueError("invalid message draft")

    usage = response.usage
    return MessageDraftResponse(
        recipient_id=recipient_id,
        content=content_value.strip(),
        usage=MessageDraftUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens,
            cache_read_input_tokens=usage.cache_read_input_tokens,
        ),
        request_snapshot=request_snapshot,
    )


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
