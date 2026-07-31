"""Domain models and request/response schemas for scenes and agents."""

import re
from typing import Any, Literal, Self
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

SCHEMA_VERSION = 5
AGENT_IDS = ("A", "B", "C")
DEFAULT_SYSTEM_PROMPT_TEMPLATE = """\
【行为原则】
像真实的人一样交流。你不必坦白全部想法，可以试探、回避、推诿、隐瞒或撒谎，但你的表达应当符合你此刻的判断和目的。

【内在驱动】
你的欲望与恐惧会影响你如何理解别人说的话、注意哪些信息、相信什么，以及接下来选择说什么。它们不是需要直接复述的标签，也不要求你向别人解释自己的心理。面对含糊的信息时，按照这个人物的欲望、恐惧和既有记忆作出主观理解，而不是采用全知或完全客观的解释。

【人设】
{persona}

【欲望】
{desire}

【恐惧】
{fear}

【记忆】
{memory}

【输出要求】
按当前回合指定的地址格式，只输出一行消息。消息正文只包含人物此刻真正会说出口的话，不要输出心理分析、推理过程或括号包裹的动作。"""

AgentId = Literal["A", "B", "C"]
MessageDirection = Literal["sent", "received"]


class ApiModel(BaseModel):
    """Base model that rejects unknown fields in requests and responses."""

    model_config = ConfigDict(extra="forbid")


def _strip_non_blank_name(value: str) -> str:
    """Strip whitespace and require a non-empty string."""
    stripped = value.strip()
    if not stripped:
        raise ValueError("name must not be blank")
    return stripped


def _strip_non_blank_content(value: str) -> str:
    """Strip whitespace and require non-empty message content."""
    stripped = value.strip()
    if not stripped:
        raise ValueError("content must not be blank")
    return stripped


def _require_non_blank_system_prompt(value: str) -> str:
    """Require a non-blank prompt without changing its exact text."""
    if not value.strip():
        raise ValueError("system_prompt must not be blank")
    return value


def _strip_non_blank_model(value: str) -> str:
    """Strip whitespace and require a concrete model name."""
    stripped = value.strip()
    if not stripped:
        raise ValueError("model must not be blank")
    return stripped


def compose_system_prompt(
    persona: str,
    desire: str,
    fear: str,
    memory: str,
) -> str:
    """Compose the canonical editable system prompt from four source slots."""
    return DEFAULT_SYSTEM_PROMPT_TEMPLATE.format(
        persona=persona,
        desire=desire,
        fear=fear,
        memory=memory,
    )


class MessageTimelineRecord(ApiModel):
    """One confirmed message as seen from an individual agent."""

    type: Literal["message"] = "message"
    message_id: UUID
    direction: MessageDirection
    counterpart_id: AgentId
    content: str

    _validate_content = field_validator("content")(_strip_non_blank_content)


class Agent(ApiModel):
    """An agent in a scene with identity, personality traits, and a timeline."""

    id: AgentId
    name: str
    persona: str = ""
    desire: str = ""
    fear: str = ""
    memory: str = ""
    system_prompt: str
    timeline: list[MessageTimelineRecord] = Field(default_factory=list)

    _validate_name = field_validator("name")(_strip_non_blank_name)
    _validate_system_prompt = field_validator("system_prompt")(
        _require_non_blank_system_prompt
    )


class AgentUpdate(ApiModel):
    """Writable agent fields exposed through the scene-update endpoint."""

    id: AgentId
    name: str
    persona: str
    desire: str
    fear: str
    memory: str
    system_prompt: str

    _validate_name = field_validator("name")(_strip_non_blank_name)
    _validate_system_prompt = field_validator("system_prompt")(
        _require_non_blank_system_prompt
    )


class Scene(ApiModel):
    """A named scene containing exactly three agents (A, B, C)."""

    schema_version: Literal[5] = SCHEMA_VERSION
    id: UUID
    name: str
    model: str | None
    agents: list[Agent] = Field(min_length=3, max_length=3)

    _validate_name = field_validator("name")(_strip_non_blank_name)

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str | None) -> str | None:
        """Allow legacy unbound scenes or one concrete model name."""
        if value is None:
            return None
        return _strip_non_blank_model(value)

    @model_validator(mode="after")
    def validate_agent_ids(self) -> Self:
        """Ensure agent IDs are A, B, C in that exact order."""
        agent_ids = tuple(agent.id for agent in self.agents)
        if agent_ids != AGENT_IDS:
            raise ValueError("agents must contain A, B, and C in that order")
        return self


class SceneSummary(ApiModel):
    """Lightweight scene reference for list responses."""

    id: UUID
    name: str


class CreateSceneRequest(ApiModel):
    """Payload for creating a new scene."""

    name: str
    model: str

    _validate_name = field_validator("name")(_strip_non_blank_name)
    _validate_model = field_validator("model")(_strip_non_blank_model)


class BindSceneModelRequest(ApiModel):
    """One-time model binding payload for a legacy scene."""

    model: str

    _validate_model = field_validator("model")(_strip_non_blank_model)


class ModelOption(ApiModel):
    """One public model choice without endpoint credentials."""

    protocol: Literal["anthropic", "responses"]
    model: str


class ModelOptionsResponse(ApiModel):
    """Configured model choices in stable protocol order."""

    options: list[ModelOption]


class UpdateSceneRequest(ApiModel):
    """Payload for updating an existing scene's editable fields."""

    name: str
    agents: list[AgentUpdate] = Field(min_length=3, max_length=3)

    _validate_name = field_validator("name")(_strip_non_blank_name)

    @model_validator(mode="after")
    def validate_agent_ids(self) -> Self:
        """Ensure agent IDs are A, B, C in that exact order."""
        agent_ids = tuple(agent.id for agent in self.agents)
        if agent_ids != AGENT_IDS:
            raise ValueError("agents must contain A, B, and C in that order")
        return self


class CreateMessageRequest(ApiModel):
    """Payload for confirming a manually authored message."""

    sender_id: AgentId
    content: str

    @model_validator(mode="after")
    def validate_addressed_content(self) -> Self:
        """Require one valid, non-self recipient in the visible text."""
        parse_addressed_message(self.content, self.sender_id)
        return self


class MessageDraftUsage(ApiModel):
    """Protocol-neutral token partitions for one draft generation."""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_creation_input_tokens: int = Field(ge=0)
    cache_read_input_tokens: int = Field(ge=0)


class ModelReasoningBlock(ApiModel):
    """One readable model-provided reasoning block for observer display."""

    type: Literal["thinking", "summary_text", "reasoning_text"]
    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """Require readable content while preserving provider whitespace."""
        if not value.strip():
            raise ValueError("reasoning text must not be blank")
        return value


class MessageDraftResponse(ApiModel):
    """Editable model-generated message draft."""

    content: str
    reasoning: list[ModelReasoningBlock]
    usage: MessageDraftUsage
    request_snapshot: dict[str, Any]

    _validate_content = field_validator("content")(_strip_non_blank_content)


class ComposeSystemPromptRequest(ApiModel):
    """Four editable slots used to compose a prompt candidate."""

    persona: str
    desire: str
    fear: str
    memory: str


class ComposeSystemPromptResponse(ApiModel):
    """Canonical prompt candidate returned without persistence."""

    system_prompt: str


class ModelRequestPreviewResponse(ApiModel):
    """Exact saved-scene payload for the selected Agent's next request."""

    request: dict[str, Any]


class MessageNotFoundError(LookupError):
    """Raised when a scene does not contain the requested message."""


class MessageDeletionConflictError(RuntimeError):
    """Raised when a message cannot be safely removed from both timelines."""


class SceneModelBindingConflictError(RuntimeError):
    """Raised when attempting to replace an existing scene model binding."""


_ADDRESSED_MESSAGE_PATTERN = re.compile(
    r"\ATo\s+([A-C])\s*[:：]\s*(\S(?:[^\r\n]*\S)?)\s*\Z"
)
_RECEIVED_MESSAGE_PATTERN = re.compile(
    r"\AFrom\s+([A-C])\s*[:：]\s*(\S(?:[^\r\n]*\S)?)\s*\Z"
)


def parse_addressed_message(
    content: str,
    sender_id: AgentId | None = None,
) -> tuple[AgentId, str]:
    """Parse one visible ``To <AgentId>: <body>`` message.

    Extra spaces and a Chinese colon are accepted at the API boundary. The
    returned pieces are used to create canonical timeline text.
    """
    if "\n" in content or "\r" in content:
        raise ValueError("content must contain exactly one line")
    match = _ADDRESSED_MESSAGE_PATTERN.fullmatch(content)
    if match is None:
        raise ValueError(
            "content must be one line in the form 'To B: message body'"
        )
    recipient_id = match.group(1)
    if sender_id == recipient_id:
        raise ValueError("message recipient must be different from sender_id")
    return recipient_id, match.group(2)


def _parse_received_message(content: str) -> tuple[AgentId, str]:
    """Parse one authoritative ``From <AgentId>: <body>`` record."""
    match = _RECEIVED_MESSAGE_PATTERN.fullmatch(content)
    if match is None:
        raise ValueError("invalid received timeline content")
    return match.group(1), match.group(2)


def create_scene(name: str, model: str) -> Scene:
    """Create a new scene with default agents.

    Args:
        name: The display name for the scene.
        model: Concrete configured model bound to the scene.

    Returns:
        A Scene with a new UUID and one default agent per agent ID.
    """
    return Scene(
        id=uuid4(),
        name=name,
        model=model,
        agents=[
            Agent(
                id=agent_id,
                name=agent_id,
                system_prompt=compose_system_prompt("", "", "", ""),
            )
            for agent_id in AGENT_IDS
        ],
    )


def bind_scene_model(scene: Scene, model: str) -> Scene:
    """Bind a legacy scene to one model without allowing later replacement."""
    if scene.model is not None:
        raise SceneModelBindingConflictError(
            f"Scene '{scene.id}' already has a model binding."
        )
    return scene.model_copy(update={"model": _strip_non_blank_model(model)})


def add_message(scene: Scene, message: CreateMessageRequest) -> Scene:
    """Append matching timeline records for a confirmed message.

    Args:
        scene: The scene receiving the message.
        message: Validated sender, recipient, and message content.

    Returns:
        A new Scene with one record appended to each participant's timeline.
    """
    recipient_id, body = parse_addressed_message(
        message.content, message.sender_id
    )
    sent_content = f"To {recipient_id}: {body}"
    received_content = f"From {message.sender_id}: {body}"
    message_id = uuid4()
    agents: list[Agent] = []

    for agent in scene.agents:
        timeline = list(agent.timeline)
        if agent.id == message.sender_id:
            timeline.append(
                MessageTimelineRecord(
                    message_id=message_id,
                    direction="sent",
                    counterpart_id=recipient_id,
                    content=sent_content,
                )
            )
        elif agent.id == recipient_id:
            timeline.append(
                MessageTimelineRecord(
                    message_id=message_id,
                    direction="received",
                    counterpart_id=message.sender_id,
                    content=received_content,
                )
            )

        agents.append(agent.model_copy(update={"timeline": timeline}))

    return scene.model_copy(update={"agents": agents})


def delete_message(scene: Scene, message_id: UUID) -> Scene:
    """Remove a paired message at both participants' timeline tops.

    Args:
        scene: The scene containing the confirmed message.
        message_id: Shared ID of the two timeline records to remove.

    Returns:
        A new Scene without either participant's matching record.

    Raises:
        MessageNotFoundError: If no timeline contains the message ID.
        MessageDeletionConflictError: If the records are not a consistent
            sent/received pair or are not both at their timeline tops.
    """
    matches = [
        (agent, index, record)
        for agent in scene.agents
        for index, record in enumerate(agent.timeline)
        if record.message_id == message_id
    ]
    if not matches:
        raise MessageNotFoundError(
            f"Message '{message_id}' does not exist in this scene."
        )
    if len(matches) != 2:
        raise MessageDeletionConflictError(
            f"Message '{message_id}' does not have exactly two records."
        )

    sent_matches = [match for match in matches if match[2].direction == "sent"]
    received_matches = [
        match for match in matches if match[2].direction == "received"
    ]
    if len(sent_matches) != 1 or len(received_matches) != 1:
        raise MessageDeletionConflictError(
            f"Message '{message_id}' is not a sent/received pair."
        )

    sender, sender_index, sent_record = sent_matches[0]
    recipient, recipient_index, received_record = received_matches[0]
    try:
        sent_recipient, sent_body = parse_addressed_message(
            sent_record.content, sender.id
        )
        received_sender, received_body = _parse_received_message(
            received_record.content
        )
    except ValueError:
        sent_recipient = None
        sent_body = None
        received_sender = None
        received_body = None
    records_are_consistent = (
        sender.id == received_record.counterpart_id
        and recipient.id == sent_record.counterpart_id
        and sender.id != recipient.id
        and sent_recipient == recipient.id
        and received_sender == sender.id
        and received_body == sent_body
        and bool(received_body)
    )
    if not records_are_consistent:
        raise MessageDeletionConflictError(
            f"Message '{message_id}' has inconsistent participant records."
        )

    # Both records must be absolute timeline tops, including unrelated messages.
    records_are_timeline_tops = (
        sender_index == len(sender.timeline) - 1
        and recipient_index == len(recipient.timeline) - 1
    )
    if not records_are_timeline_tops:
        raise MessageDeletionConflictError(
            f"Message '{message_id}' is not last in both timelines."
        )

    participant_ids = {sender.id, recipient.id}
    agents = [
        agent.model_copy(update={"timeline": agent.timeline[:-1]})
        if agent.id in participant_ids
        else agent
        for agent in scene.agents
    ]
    return scene.model_copy(update={"agents": agents})


def update_scene(scene: Scene, update: UpdateSceneRequest) -> Scene:
    """Apply a scene update, preserving the existing timeline.

    Only the editable fields (name, agent names, persona, desire, fear,
    memory) are taken from the update payload. Schema version, scene ID,
    and agent timelines are carried forward from the original scene.

    Args:
        scene: The existing scene to update.
        update: The update payload with new values.

    Returns:
        A new Scene instance reflecting the merged state.
    """
    return Scene(
        schema_version=scene.schema_version,
        id=scene.id,
        name=update.name,
        model=scene.model,
        agents=[
            Agent(
                id=updated_agent.id,
                name=updated_agent.name,
                persona=updated_agent.persona,
                desire=updated_agent.desire,
                fear=updated_agent.fear,
                memory=updated_agent.memory,
                system_prompt=updated_agent.system_prompt,
                timeline=existing_agent.timeline,
            )
            for existing_agent, updated_agent in zip(
                scene.agents, update.agents, strict=True
            )
        ],
    )
