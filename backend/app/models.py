"""Domain models and request/response schemas for scenes and agents."""

from typing import Annotated, Any, Literal, Self
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

SCHEMA_VERSION = 3
AGENT_IDS = ("A", "B", "C")
DEFAULT_SYSTEM_PROMPT_TEMPLATE = """\
【规则】
像真人一样说话，你不必全盘托出，可以推诿和回避甚至撒谎。不要输出括号包裹的动作，只输出说的话。不要有换行，所有话一口气说完。记住，一个人最本质的东西是他的欲望和恐惧。

【人设】
{persona}

【欲望】
{desire}

【恐惧】
{fear}

【记忆】
{memory}"""

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


class InnerVoiceTimelineRecord(ApiModel):
    """One user-authored private prompt in an individual agent's timeline."""

    type: Literal["inner_voice"] = "inner_voice"
    inner_voice_id: UUID
    content: str

    _validate_content = field_validator("content")(_strip_non_blank_content)


TimelineRecord = Annotated[
    MessageTimelineRecord | InnerVoiceTimelineRecord,
    Field(discriminator="type"),
]


class Agent(ApiModel):
    """An agent in a scene with identity, personality traits, and a timeline."""

    id: AgentId
    name: str
    persona: str = ""
    desire: str = ""
    fear: str = ""
    memory: str = ""
    system_prompt: str
    timeline: list[TimelineRecord] = Field(default_factory=list)

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

    schema_version: Literal[3] = SCHEMA_VERSION
    id: UUID
    name: str
    agents: list[Agent] = Field(min_length=3, max_length=3)

    _validate_name = field_validator("name")(_strip_non_blank_name)

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

    _validate_name = field_validator("name")(_strip_non_blank_name)


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
    recipient_id: AgentId
    content: str

    _validate_content = field_validator("content")(_strip_non_blank_content)

    @model_validator(mode="after")
    def validate_distinct_participants(self) -> Self:
        """Require two different agents to participate in the message."""
        if self.sender_id == self.recipient_id:
            raise ValueError("sender_id and recipient_id must be different")
        return self


class CreateInnerVoiceRequest(ApiModel):
    """Payload for appending a private inner voice to one Agent."""

    content: str

    _validate_content = field_validator("content")(_strip_non_blank_content)


class MessageDraftUsage(ApiModel):
    """Anthropic token usage returned for one draft generation."""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_creation_input_tokens: int = Field(ge=0)
    cache_read_input_tokens: int = Field(ge=0)


class MessageDraftResponse(ApiModel):
    """Editable model-generated message draft."""

    recipient_id: AgentId | None
    content: str
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


class InnerVoiceNotFoundError(LookupError):
    """Raised when an Agent timeline does not contain an inner voice."""


class InnerVoiceDeletionConflictError(RuntimeError):
    """Raised when an inner voice is not the Agent timeline's final record."""


def create_scene(name: str) -> Scene:
    """Create a new scene with default agents.

    Args:
        name: The display name for the scene.

    Returns:
        A Scene with a new UUID and one default agent per agent ID.
    """
    return Scene(
        id=uuid4(),
        name=name,
        agents=[
            Agent(
                id=agent_id,
                name=agent_id,
                system_prompt=compose_system_prompt("", "", "", ""),
            )
            for agent_id in AGENT_IDS
        ],
    )


def add_message(scene: Scene, message: CreateMessageRequest) -> Scene:
    """Append matching timeline records for a confirmed message.

    Args:
        scene: The scene receiving the message.
        message: Validated sender, recipient, and message content.

    Returns:
        A new Scene with one record appended to each participant's timeline.
    """
    message_id = uuid4()
    agents: list[Agent] = []

    for agent in scene.agents:
        timeline = list(agent.timeline)
        if agent.id == message.sender_id:
            timeline.append(
                MessageTimelineRecord(
                    message_id=message_id,
                    direction="sent",
                    counterpart_id=message.recipient_id,
                    content=message.content,
                )
            )
        elif agent.id == message.recipient_id:
            timeline.append(
                MessageTimelineRecord(
                    message_id=message_id,
                    direction="received",
                    counterpart_id=message.sender_id,
                    content=message.content,
                )
            )

        agents.append(agent.model_copy(update={"timeline": timeline}))

    return scene.model_copy(update={"agents": agents})


def add_inner_voice(
    scene: Scene,
    agent_id: AgentId,
    request: CreateInnerVoiceRequest,
) -> Scene:
    """Append a private inner voice only to the target Agent's timeline."""
    agents = []
    for agent in scene.agents:
        if agent.id != agent_id:
            agents.append(agent)
            continue
        timeline = [
            *agent.timeline,
            InnerVoiceTimelineRecord(
                inner_voice_id=uuid4(),
                content=request.content,
            ),
        ]
        agents.append(agent.model_copy(update={"timeline": timeline}))
    return scene.model_copy(update={"agents": agents})


def delete_inner_voice(
    scene: Scene,
    agent_id: AgentId,
    inner_voice_id: UUID,
) -> Scene:
    """Delete an inner voice only when it is the target timeline's top."""
    agent = next(agent for agent in scene.agents if agent.id == agent_id)
    matching_indexes = [
        index
        for index, record in enumerate(agent.timeline)
        if (
            record.type == "inner_voice"
            and record.inner_voice_id == inner_voice_id
        )
    ]
    if not matching_indexes:
        raise InnerVoiceNotFoundError(
            f"Inner voice '{inner_voice_id}' does not exist for Agent "
            f"'{agent_id}'."
        )
    if matching_indexes[-1] != len(agent.timeline) - 1:
        raise InnerVoiceDeletionConflictError(
            f"Inner voice '{inner_voice_id}' is not last in Agent "
            f"'{agent_id}' timeline."
        )

    agents = [
        current.model_copy(update={"timeline": current.timeline[:-1]})
        if current.id == agent_id
        else current
        for current in scene.agents
    ]
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
        if record.type == "message" and record.message_id == message_id
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
    records_are_consistent = (
        sender.id == received_record.counterpart_id
        and recipient.id == sent_record.counterpart_id
        and sender.id != recipient.id
        and sent_record.content == received_record.content
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
