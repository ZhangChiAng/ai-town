"""Anthropic-backed generation of one-step message drafts."""

import json
import logging
from collections.abc import Callable
from typing import Any, Protocol
from uuid import UUID

from app.config import ModelSettings
from app.models import (
    AGENT_IDS,
    Agent,
    AgentId,
    MessageDraftResponse,
    MessageDraftUsage,
    Scene,
    parse_addressed_message,
)

LOGGER = logging.getLogger(__name__)
CACHE_CONTROL = {"type": "ephemeral", "ttl": "5m"}
RUNTIME_TURN_PROMPT = (
    "现在轮到你说话。请从另外两位 Agent（{recipient_ids}）中选择一位接收人，"
    "只输出一行 `To X: 消息正文`，其中 X 必须替换为所选接收人的 Agent ID。"
)


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
    """Generate plain-text message drafts without mutating scene state."""

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
            A validated text draft and the upstream usage metrics.

        Raises:
            DraftGenerationError: If the upstream request fails or returns an
                invalid response.
        """
        model_request = build_message_request(scene, agent_id, self._model)

        try:
            response = self._client.messages.create(**model_request)
        except Exception as error:
            # Do not propagate provider response bodies or credentials.
            raise DraftGenerationError("Model request failed.") from error

        try:
            draft = _parse_response(response, model_request, agent_id)
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
    messages = _build_timeline_messages(agent)
    if messages:
        # The rolling breakpoint follows the final immutable timeline record.
        messages[-1]["content"][-1]["cache_control"] = dict(CACHE_CONTROL)
    recipient_ids = "、".join(
        candidate for candidate in AGENT_IDS if candidate != agent_id
    )
    runtime_block = {
        "type": "text",
        "text": RUNTIME_TURN_PROMPT.format(recipient_ids=recipient_ids),
    }
    if messages and messages[-1]["role"] == "user":
        # Keep strict role alternation while preserving the runtime instruction
        # as its own visible text block.
        messages[-1]["content"].append(runtime_block)
    else:
        messages.append(
            {
                "role": "user",
                "content": [runtime_block],
            }
        )

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
        "messages": messages,
    }


def _build_timeline_messages(agent: Agent) -> list[dict[str, Any]]:
    """Map the private timeline to alternating native Messages API turns."""
    messages: list[dict[str, Any]] = []
    for record in agent.timeline:
        role = "user" if record.direction == "received" else "assistant"

        block = {"type": "text", "text": record.content}
        if messages and messages[-1]["role"] == role:
            # Some compatible gateways require strict role alternation even
            # though Anthropic itself combines adjacent same-role messages.
            messages[-1]["content"].append(block)
        else:
            messages.append({"role": role, "content": [block]})
    return messages


def _parse_response(
    response: Any,
    request_snapshot: dict[str, Any],
    sender_id: AgentId,
) -> MessageDraftResponse:
    """Extract one text draft while allowing provider thinking blocks."""
    content = response.content
    if not isinstance(content, list) or not content:
        raise ValueError("expected response blocks")

    allowed_block_types = {"text", "thinking", "redacted_thinking"}
    if any(
        getattr(block, "type", None) not in allowed_block_types
        for block in content
    ):
        raise ValueError("unexpected response block")

    # Extended-thinking models may put private reasoning before the sole
    # user-visible text block; only that visible block becomes the draft.
    text_blocks = [
        block for block in content if getattr(block, "type", None) == "text"
    ]
    if len(text_blocks) != 1:
        raise ValueError("expected exactly one text response")

    content_value = text_blocks[0].text
    if not isinstance(content_value, str) or not content_value.strip():
        raise ValueError("invalid message draft")
    if "\n" in content_value or "\r" in content_value:
        raise ValueError("message draft must be one line")
    visible_content = content_value.strip()
    recipient_id, _body = parse_addressed_message(visible_content, sender_id)
    if recipient_id not in AGENT_IDS:
        raise ValueError("invalid message recipient")

    usage = response.usage
    return MessageDraftResponse(
        content=visible_content,
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
