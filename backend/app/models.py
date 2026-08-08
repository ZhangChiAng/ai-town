"""Domain models and state transitions for two-layer AI Town scenes."""

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

SCENE_SCHEMA_NAMESPACE = "ai-town.scene"
SCENE_SCHEMA_MAJOR = 1
CURRENT_SCENE_SCHEMA = f"{SCENE_SCHEMA_NAMESPACE}/1.0"
AGENT_IDS = ("A", "B", "C")

_SCENE_SCHEMA_PATTERN = re.compile(
    rf"\A{re.escape(SCENE_SCHEMA_NAMESPACE)}/"
    r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\Z"
)

AgentId = Literal["A", "B", "C"]
Layer = Literal["inner", "outer"]
EventKind = Literal["manual", "agent_message"]


class ApiModel(BaseModel):
    """Base API model that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid")


def _require_non_blank(value: str) -> str:
    """Require non-empty text while preserving user-authored whitespace."""
    if not value.strip():
        raise ValueError("text must not be blank")
    return value


def _strip_non_blank_name(value: str) -> str:
    """Strip surrounding whitespace and require a non-empty name."""
    stripped = value.strip()
    if not stripped:
        raise ValueError("name must not be blank")
    return stripped


def _strip_non_blank_model(value: str) -> str:
    """Strip surrounding whitespace and require a concrete model name."""
    stripped = value.strip()
    if not stripped:
        raise ValueError("model must not be blank")
    return stripped


def parse_scene_schema(value: object) -> tuple[int, int]:
    """Parse a strict namespaced scene schema identifier."""
    if not isinstance(value, str):
        raise ValueError("scene schema must be a string")
    match = _SCENE_SCHEMA_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError("scene schema has an invalid namespace or version")
    return int(match.group(1)), int(match.group(2))


_SEMANTIC_MESSAGE_PREFIX = re.compile(r"\A对\s*(.+?)\s*说\s*[:：]\s*")


def parse_semantic_message(
    content: str,
    interactions: dict[AgentId, Interaction],
) -> tuple[AgentId, str, str]:
    """Parse one configured ``对{称呼}说：{正文}`` message.

    Surrounding whitespace and either colon form are accepted for semantic
    speech, then the configured address and body are returned canonically.
    ``STOP`` is intentionally handled separately because it has no route.
    """
    match = _SEMANTIC_MESSAGE_PREFIX.match(content.strip())
    if match is None:
        raise ValueError(
            "content must be '对{已配置称呼}说：{非空正文}' or exact 'STOP'"
        )
    address = match.group(1).strip()
    recipients = [
        target_id
        for target_id, interaction in interactions.items()
        if address in interaction.addresses
    ]
    if len(recipients) != 1:
        raise ValueError("message address is not uniquely configured")
    body = content.strip()[match.end() :].strip()
    if not body:
        raise ValueError("message body must not be blank")
    return recipients[0], address, body


def parse_canonical_semantic_message(content: str) -> tuple[str, str]:
    """Parse canonical saved semantic speech without current configuration."""
    match = _SEMANTIC_MESSAGE_PREFIX.match(content)
    if match is None:
        raise ValueError("outer output must be canonical semantic speech")
    address = match.group(1).strip()
    body = content[match.end() :].strip()
    if not address or not body or content != f"对{address}说：{body}":
        raise ValueError("outer output must be canonical semantic speech")
    return address, body


class TokenUsage(ApiModel):
    """Provider token usage for one generation."""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_creation_input_tokens: int = Field(ge=0)
    cache_read_input_tokens: int = Field(ge=0)


class ModelReasoningBlock(ApiModel):
    """One readable provider reasoning block projected before confirmation."""

    type: Literal["thinking", "summary_text", "reasoning_text"]
    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """Require readable content while preserving provider whitespace."""
        if not value.strip():
            raise ValueError("reasoning text must not be blank")
        return value


class ExternalEvent(ApiModel):
    """One event queued for or already consumed by a single Agent."""

    id: UUID
    sequence: int = Field(ge=1)
    kind: EventKind
    content: str
    source_agent_id: AgentId | None
    source_call_id: UUID | None

    _validate_content = field_validator("content")(_require_non_blank)

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        """Keep manual and Agent-produced event provenance unambiguous."""
        has_agent = self.source_agent_id is not None
        has_call = self.source_call_id is not None
        if self.kind == "manual" and (has_agent or has_call):
            raise ValueError("manual events must not have Agent provenance")
        if self.kind == "agent_message" and not (has_agent and has_call):
            raise ValueError("Agent events require Agent provenance")
        return self


class InnerTurn(ApiModel):
    """One confirmed inner-layer model call.

    A single inner turn consumes every event that was queued for the Agent
    at the start of the round, so it stores the full consumed batch and the
    ids of those events in FIFO order. ``event_ids`` aligns 1:1 with
    ``consumed_events``.
    """

    call_id: UUID
    event_ids: list[UUID]
    sequence: int = Field(ge=1)
    input: str
    output: str
    consumed_events: list[ExternalEvent]
    reasoning: list[ModelReasoningBlock]

    _validate_input = field_validator("input")(_require_non_blank)
    _validate_output = field_validator("output")(_require_non_blank)

    @model_validator(mode="after")
    def validate_event_references(self) -> Self:
        """Require each turn to embed its full consumed event batch."""
        if not self.event_ids or not self.consumed_events:
            raise ValueError("inner turn must consume at least one event")
        if len(self.event_ids) != len(self.consumed_events):
            raise ValueError(
                "inner turn event_ids and consumed_events must align"
            )
        for event_id, event in zip(
            self.event_ids, self.consumed_events, strict=True
        ):
            if event_id != event.id:
                raise ValueError(
                    "inner turn event_ids must match consumed_events"
                )
        return self


class OuterTurn(ApiModel):
    """One confirmed outer-layer call and its optional routed event.

    ``event_ids`` mirrors the event batch consumed by the matching inner
    turn. Semantic speech has both routing fields; ``STOP`` has neither.
    """

    call_id: UUID
    event_ids: list[UUID]
    sequence: int = Field(ge=1)
    input: str
    output: str
    recipient_id: AgentId | None
    generated_event_id: UUID | None
    reasoning: list[ModelReasoningBlock]

    _validate_input = field_validator("input")(_require_non_blank)
    _validate_output = field_validator("output")(_require_non_blank)

    @model_validator(mode="after")
    def validate_route(self) -> Self:
        """Keep STOP and routed speech metadata mutually consistent."""
        if self.output == "STOP":
            if (
                self.recipient_id is not None
                or self.generated_event_id is not None
            ):
                raise ValueError("STOP turns must not have routing metadata")
            return self
        if self.recipient_id is None or self.generated_event_id is None:
            raise ValueError("semantic speech requires routing metadata")
        parse_canonical_semantic_message(self.output)
        return self


class InnerContext(ApiModel):
    """Complete confirmed inner history without a persisted prompt."""

    turns: list[InnerTurn]


class OuterContext(ApiModel):
    """Complete confirmed outer history without a persisted prompt."""

    turns: list[OuterTurn]


class PromptProfile(ApiModel):
    """User-authored variables inserted into backend prompt templates."""

    pronoun: str
    hidden_beliefs: str
    inner_memories: str
    outer_memories: str


class Interaction(ApiModel):
    """One sender's relationship view of an interactive person."""

    description: str
    addresses: dict[str, str]

    @field_validator("addresses", mode="before")
    @classmethod
    def normalize_addresses(cls, value: Any) -> Any:
        """Trim route keys and occasions while retaining their input order."""
        if not isinstance(value, dict):
            return value
        normalized: dict[Any, Any] = {}
        for raw_address, raw_occasion in value.items():
            address = (
                raw_address.strip()
                if isinstance(raw_address, str)
                else raw_address
            )
            occasion = (
                raw_occasion.strip()
                if isinstance(raw_occasion, str)
                else raw_occasion
            )
            if not address or not occasion:
                raise ValueError(
                    "interaction addresses and occasions must not be blank"
                )
            if address in normalized:
                raise ValueError("interaction addresses must be unique")
            normalized[address] = occasion
        return normalized


class Agent(ApiModel):
    """One Agent with isolated inner/outer contexts and a FIFO event queue."""

    id: AgentId
    name: str
    prompt_profile: PromptProfile
    interactions: dict[AgentId, Interaction]
    inner_context: InnerContext
    outer_context: OuterContext
    pending_events: list[ExternalEvent]

    _validate_name = field_validator("name")(_strip_non_blank_name)

    @model_validator(mode="after")
    def validate_round_alignment(self) -> Self:
        """Allow only complete rounds or one confirmed inner half-round."""
        if self.id in self.interactions:
            raise ValueError("an Agent cannot interact with itself")
        seen_addresses: set[str] = set()
        for interaction in self.interactions.values():
            for address in interaction.addresses:
                if address in seen_addresses:
                    raise ValueError(
                        "interaction addresses must be unique per sender"
                    )
                seen_addresses.add(address)
        inner_count = len(self.inner_context.turns)
        outer_count = len(self.outer_context.turns)
        if inner_count not in (outer_count, outer_count + 1):
            raise ValueError("inner and outer turns are not phase-aligned")
        for inner_turn, outer_turn in zip(
            self.inner_context.turns,
            self.outer_context.turns,
            strict=False,
        ):
            if inner_turn.event_ids != outer_turn.event_ids:
                raise ValueError("inner and outer turn events must align")
        if any(
            left.sequence >= right.sequence
            for left, right in zip(
                self.pending_events,
                self.pending_events[1:],
                strict=False,
            )
        ):
            raise ValueError("pending_events must retain FIFO sequence order")
        return self


class ConfirmedCallReference(ApiModel):
    """Minimal scene-level reference used by the global rollback stack."""

    call_id: UUID
    agent_id: AgentId
    layer: Layer


class Scene(ApiModel):
    """A current-major scene containing exactly three two-layer Agents."""

    schema_id: str = Field(
        validation_alias="schema",
        serialization_alias="schema",
    )
    id: UUID
    name: str
    model: str
    agents: list[Agent] = Field(min_length=3, max_length=3)
    rollback_stack: list[ConfirmedCallReference]
    next_sequence: int = Field(ge=1)

    _validate_name = field_validator("name")(_strip_non_blank_name)
    _validate_model = field_validator("model")(_strip_non_blank_model)

    @field_validator("schema_id")
    @classmethod
    def validate_schema(cls, value: str) -> str:
        """Accept every well-formed schema in the current major version."""
        major, _minor = parse_scene_schema(value)
        if major != SCENE_SCHEMA_MAJOR:
            raise ValueError(f"scene schema major must be {SCENE_SCHEMA_MAJOR}")
        return value

    @model_validator(mode="after")
    def validate_scene_integrity(self) -> Self:
        """Validate IDs, call order, event ownership, and routing references."""
        if tuple(agent.id for agent in self.agents) != AGENT_IDS:
            raise ValueError("agents must contain A, B, and C in order")
        names = [agent.name for agent in self.agents]
        if len(set(names)) != len(names):
            raise ValueError("Agent names must be unique within a scene")

        calls: list[tuple[int, ConfirmedCallReference]] = []
        events: dict[UUID, tuple[AgentId, ExternalEvent]] = {}
        generated_events: list[tuple[AgentId, OuterTurn]] = []
        used_call_ids: set[UUID] = set()
        used_call_sequences: set[int] = set()
        max_sequence = 0

        for agent in self.agents:
            for event in agent.pending_events:
                _record_event(events, agent.id, event)
                max_sequence = max(max_sequence, event.sequence)
            for turn in agent.inner_context.turns:
                for consumed in turn.consumed_events:
                    _record_event(events, agent.id, consumed)
                reference = ConfirmedCallReference(
                    call_id=turn.call_id,
                    agent_id=agent.id,
                    layer="inner",
                )
                _record_call(
                    calls,
                    used_call_ids,
                    used_call_sequences,
                    turn.sequence,
                    reference,
                )
                for consumed in turn.consumed_events:
                    max_sequence = max(
                        max_sequence,
                        turn.sequence,
                        consumed.sequence,
                    )
            for turn in agent.outer_context.turns:
                if turn.output == "STOP":
                    if (
                        turn.recipient_id is not None
                        or turn.generated_event_id is not None
                    ):
                        raise ValueError("STOP turn routing must be empty")
                else:
                    _address, _body = parse_canonical_semantic_message(
                        turn.output
                    )
                    if turn.recipient_id == agent.id:
                        raise ValueError(
                            "outer message recipient must differ from sender"
                        )
                reference = ConfirmedCallReference(
                    call_id=turn.call_id,
                    agent_id=agent.id,
                    layer="outer",
                )
                _record_call(
                    calls,
                    used_call_ids,
                    used_call_sequences,
                    turn.sequence,
                    reference,
                )
                if turn.generated_event_id is not None:
                    generated_events.append((agent.id, turn))
                if not turn.event_ids:
                    raise ValueError("outer turn must reference inner events")
                max_sequence = max(max_sequence, turn.sequence)

        calls.sort(key=lambda item: item[0])
        if [reference for _sequence, reference in calls] != self.rollback_stack:
            raise ValueError(
                "rollback_stack must reference every call in order"
            )

        for sender_id, turn in generated_events:
            assert turn.generated_event_id is not None
            assert turn.recipient_id is not None
            event_location = events.get(turn.generated_event_id)
            if event_location is None:
                raise ValueError("routed event is missing")
            recipient_id, event = event_location
            _address, body = parse_canonical_semantic_message(turn.output)
            if (
                recipient_id != turn.recipient_id
                or event.kind != "agent_message"
                or event.source_agent_id != sender_id
                or event.source_call_id != turn.call_id
                or event.sequence != turn.sequence
                or event.content != body
            ):
                raise ValueError("routed event does not match its outer turn")

        if self.next_sequence <= max_sequence:
            raise ValueError(
                "next_sequence must exceed saved display sequences"
            )
        return self


def _record_event(
    events: dict[UUID, tuple[AgentId, ExternalEvent]],
    agent_id: AgentId,
    event: ExternalEvent,
) -> None:
    """Add one actual event occurrence while rejecting duplicate IDs."""
    if event.id in events:
        raise ValueError("each event must occur exactly once")
    events[event.id] = (agent_id, event)


def _record_call(
    calls: list[tuple[int, ConfirmedCallReference]],
    used_ids: set[UUID],
    used_sequences: set[int],
    sequence: int,
    reference: ConfirmedCallReference,
) -> None:
    """Add one confirmed call while enforcing global identity and order."""
    if reference.call_id in used_ids:
        raise ValueError("call_id must be unique")
    if sequence in used_sequences:
        raise ValueError("confirmed calls must have unique sequences")
    used_ids.add(reference.call_id)
    used_sequences.add(sequence)
    calls.append((sequence, reference))


class SceneSummary(ApiModel):
    """Lightweight scene reference for list responses."""

    id: UUID
    name: str


class CreateSceneRequest(ApiModel):
    """Payload for creating a scene."""

    name: str
    model: str

    _validate_name = field_validator("name")(_strip_non_blank_name)
    _validate_model = field_validator("model")(_strip_non_blank_model)


class ModelOption(ApiModel):
    """One public model choice without protocol or endpoint details."""

    model: str


class ModelOptionsResponse(ApiModel):
    """Configured model choices in stable TOML order."""

    options: list[ModelOption]


class AgentUpdate(ApiModel):
    """Writable Agent identity, prompt variables, and interactions."""

    id: AgentId
    name: str
    prompt_profile: PromptProfile
    interactions: dict[AgentId, Interaction]

    _validate_name = field_validator("name")(_strip_non_blank_name)

    @model_validator(mode="after")
    def validate_interactions(self) -> Self:
        """Reuse Agent validation for self-target and address uniqueness."""
        Agent(
            id=self.id,
            name=self.name,
            prompt_profile=self.prompt_profile,
            interactions=self.interactions,
            inner_context=InnerContext(turns=[]),
            outer_context=OuterContext(turns=[]),
            pending_events=[],
        )
        return self


class UpdateSceneRequest(ApiModel):
    """Payload for editing scene and Agent prompt text."""

    name: str
    agents: list[AgentUpdate] = Field(min_length=3, max_length=3)

    _validate_name = field_validator("name")(_strip_non_blank_name)

    @model_validator(mode="after")
    def validate_agent_ids(self) -> Self:
        """Require stable IDs and unique trimmed names."""
        if tuple(agent.id for agent in self.agents) != AGENT_IDS:
            raise ValueError("agents must contain A, B, and C in order")
        names = [agent.name for agent in self.agents]
        if len(set(names)) != len(names):
            raise ValueError("Agent names must be unique within a scene")
        return self


class EventContentRequest(ApiModel):
    """Payload for creating or editing a manual event."""

    content: str

    _validate_content = field_validator("content")(_require_non_blank)


class LayerDraftResponse(ApiModel):
    """Browser-only result and captured JSON body from one successful call."""

    layer: Layer
    call_id: UUID
    event_ids: list[UUID]
    content: str
    reasoning: list[ModelReasoningBlock]
    usage: TokenUsage
    request_snapshot: dict[str, Any]
    state_token: str = Field(min_length=64, max_length=64)

    _validate_content = field_validator("content")(_require_non_blank)


class ConfirmLayerRequest(ApiModel):
    """Browser-held generation identity and editable confirmed output.

    ``event_ids`` is the ordered batch this confirmation must match against
    the rebuilt ``build_inner_input`` / ``build_outer_input`` state.
    """

    call_id: UUID
    event_ids: list[UUID]
    content: str
    state_token: str = Field(min_length=64, max_length=64)
    reasoning: list[ModelReasoningBlock] = Field(default_factory=list)

    _validate_content = field_validator("content")(_require_non_blank)


class ModelRequestContextItem(ApiModel):
    """One ordered, readable item in a model request preview."""

    role: Literal["system", "user", "assistant"]
    text: str


class ModelRequestPreviewResponse(ApiModel):
    """Protocol-neutral context for one selected persona layer."""

    layer: Layer
    event_ids: list[UUID]
    context: list[ModelRequestContextItem]


class SceneConflictError(RuntimeError):
    """Raised when an operation conflicts with the current scene state."""


class InvalidLayerOutputError(ValueError):
    """Raised when edited output violates the selected layer contract."""


class EventNotFoundError(LookupError):
    """Raised when a scene has no event with the requested ID."""


def create_scene(name: str, model: str) -> Scene:
    """Create an empty scene at the user-confirmed current schema."""
    return Scene(
        schema=CURRENT_SCENE_SCHEMA,
        id=uuid4(),
        name=name,
        model=model,
        agents=[
            Agent(
                id=agent_id,
                name=agent_id,
                prompt_profile=PromptProfile(
                    pronoun="",
                    hidden_beliefs="",
                    inner_memories="",
                    outer_memories="",
                ),
                interactions={},
                inner_context=InnerContext(turns=[]),
                outer_context=OuterContext(turns=[]),
                pending_events=[],
            )
            for agent_id in AGENT_IDS
        ],
        rollback_stack=[],
        next_sequence=1,
    )


def update_scene(scene: Scene, update: UpdateSceneRequest) -> Scene:
    """Apply prompt variables and interactions while preserving history."""
    agents = [
        Agent(
            id=updated.id,
            name=updated.name,
            prompt_profile=updated.prompt_profile,
            interactions=updated.interactions,
            inner_context=InnerContext(
                turns=current.inner_context.turns,
            ),
            outer_context=OuterContext(
                turns=current.outer_context.turns,
            ),
            pending_events=current.pending_events,
        )
        for current, updated in zip(
            scene.agents,
            update.agents,
            strict=True,
        )
    ]
    return _replace_scene(scene, name=update.name, agents=agents)


def add_manual_event(
    scene: Scene,
    agent_id: AgentId,
    content: str,
) -> Scene:
    """Append one user-authored event to an Agent's FIFO queue."""
    event = ExternalEvent(
        id=uuid4(),
        sequence=scene.next_sequence,
        kind="manual",
        content=content,
        source_agent_id=None,
        source_call_id=None,
    )
    agents = [
        agent.model_copy(
            update={"pending_events": [*agent.pending_events, event]}
        )
        if agent.id == agent_id
        else agent
        for agent in scene.agents
    ]
    return _replace_scene(
        scene,
        agents=agents,
        next_sequence=scene.next_sequence + 1,
    )


def edit_manual_event(
    scene: Scene,
    agent_id: AgentId,
    event_id: UUID,
    content: str,
) -> Scene:
    """Edit one still-pending manual event without changing FIFO position."""
    owner, status, event = _find_event(scene, event_id)
    if owner != agent_id:
        raise EventNotFoundError(f"Event '{event_id}' does not exist.")
    if status != "pending":
        raise SceneConflictError("Only queued manual events can be edited.")
    if event.kind != "manual":
        raise SceneConflictError("Agent-produced events cannot be edited.")

    agents = []
    for agent in scene.agents:
        pending_events = [
            queued.model_copy(update={"content": content})
            if queued.id == event_id
            else queued
            for queued in agent.pending_events
        ]
        agents.append(
            agent.model_copy(update={"pending_events": pending_events})
        )
    return _replace_scene(scene, agents=agents)


def delete_manual_event(
    scene: Scene,
    agent_id: AgentId,
    event_id: UUID,
) -> Scene:
    """Delete one still-pending manual event."""
    owner, status, event = _find_event(scene, event_id)
    if owner != agent_id:
        raise EventNotFoundError(f"Event '{event_id}' does not exist.")
    if status != "pending":
        raise SceneConflictError("Only queued manual events can be deleted.")
    if event.kind != "manual":
        raise SceneConflictError("Agent-produced events cannot be deleted.")

    agents = [
        agent.model_copy(
            update={
                "pending_events": [
                    queued
                    for queued in agent.pending_events
                    if queued.id != event_id
                ]
            }
        )
        if agent.id == agent_id
        else agent
        for agent in scene.agents
    ]
    return _replace_scene(scene, agents=agents)


def _join_event_contents(events: list[ExternalEvent]) -> str:
    """Render a FIFO event batch's contents joined by blank lines."""
    return "\n\n".join(event.content for event in events)


def build_inner_input(
    scene: Scene,
    agent_id: AgentId,
) -> tuple[list[UUID], str]:
    """Build the exact next inner user text from the current pending batch.

    Every event currently in the queue is consumed in this round. Events are
    presented in FIFO order under a single ``外部事件：`` block. The optional
    prior outer result is expressed with its saved semantic address, or as
    an explicit no-speech sentence for ``STOP``.
    """
    agent = get_agent(scene, agent_id)
    if len(agent.inner_context.turns) != len(agent.outer_context.turns):
        raise SceneConflictError("The Agent is waiting for outer confirmation.")
    if not agent.pending_events:
        raise SceneConflictError("The Agent has no pending event.")

    events = list(agent.pending_events)
    event_ids = [event.id for event in events]
    sections: list[str] = []
    if agent.outer_context.turns:
        previous_outer = agent.outer_context.turns[-1]
        if previous_outer.output == "STOP":
            sections.append("上一轮：你没有说话。")
        else:
            address, body = parse_canonical_semantic_message(
                previous_outer.output
            )
            sections.append(f"上一轮：\n你对{address}说：\n{body}")
    sections.append("外部事件：\n" + _join_event_contents(events))
    return event_ids, "\n\n".join(sections)


def build_outer_input(
    scene: Scene, agent_id: AgentId
) -> tuple[list[UUID], str]:
    """Build the exact outer user text for a confirmed inner half-round."""
    agent = get_agent(scene, agent_id)
    if len(agent.inner_context.turns) != len(agent.outer_context.turns) + 1:
        raise SceneConflictError(
            "The Agent has no confirmed inner turn awaiting outer output."
        )
    inner_turn = agent.inner_context.turns[-1]
    text = (
        f"外部事件：\n{_join_event_contents(inner_turn.consumed_events)}\n\n"
        f"你内心有一个声音：\n{inner_turn.output}"
    )
    return inner_turn.event_ids, text


def confirm_inner_turn(
    scene: Scene,
    agent_id: AgentId,
    confirmation: ConfirmLayerRequest,
    actual_input: str,
) -> Scene:
    """Consume the entire pending batch and append one confirmed inner turn."""
    event_ids, expected_input = build_inner_input(scene, agent_id)
    if confirmation.event_ids != event_ids or actual_input != expected_input:
        raise SceneConflictError("The inner draft is stale.")
    _require_new_call_id(scene, confirmation.call_id)

    agent = get_agent(scene, agent_id)
    events = list(agent.pending_events)
    turn = InnerTurn(
        call_id=confirmation.call_id,
        event_ids=[event.id for event in events],
        sequence=scene.next_sequence,
        input=actual_input,
        output=confirmation.content,
        consumed_events=events,
        reasoning=confirmation.reasoning,
    )
    agents = [
        current.model_copy(
            update={
                "inner_context": current.inner_context.model_copy(
                    update={"turns": [*current.inner_context.turns, turn]}
                ),
                "pending_events": [],
            }
        )
        if current.id == agent_id
        else current
        for current in scene.agents
    ]
    return _replace_scene(
        scene,
        agents=agents,
        rollback_stack=[
            *scene.rollback_stack,
            ConfirmedCallReference(
                call_id=confirmation.call_id,
                agent_id=agent_id,
                layer="inner",
            ),
        ],
        next_sequence=scene.next_sequence + 1,
    )


def confirm_outer_turn(
    scene: Scene,
    agent_id: AgentId,
    confirmation: ConfirmLayerRequest,
    actual_input: str,
) -> Scene:
    """Append speech or STOP and route only semantic speech atomically."""
    event_ids, expected_input = build_outer_input(scene, agent_id)
    if confirmation.event_ids != event_ids or actual_input != expected_input:
        raise SceneConflictError("The outer draft is stale.")
    _require_new_call_id(scene, confirmation.call_id)

    agent = get_agent(scene, agent_id)
    recipient_id: AgentId | None = None
    generated_event: ExternalEvent | None = None
    if confirmation.content == "STOP":
        canonical_output = "STOP"
    else:
        recipient_id, address, body = parse_semantic_message(
            confirmation.content,
            agent.interactions,
        )
        canonical_output = f"对{address}说：{body}"
        generated_event = ExternalEvent(
            id=uuid4(),
            sequence=scene.next_sequence,
            kind="agent_message",
            content=body,
            source_agent_id=agent_id,
            source_call_id=confirmation.call_id,
        )
    turn = OuterTurn(
        call_id=confirmation.call_id,
        event_ids=event_ids,
        sequence=scene.next_sequence,
        input=actual_input,
        output=canonical_output,
        recipient_id=recipient_id,
        generated_event_id=(
            generated_event.id if generated_event is not None else None
        ),
        reasoning=confirmation.reasoning,
    )

    agents = []
    for current in scene.agents:
        outer_context = current.outer_context
        pending_events = current.pending_events
        if current.id == agent_id:
            outer_context = current.outer_context.model_copy(
                update={"turns": [*current.outer_context.turns, turn]}
            )
        if generated_event is not None and current.id == recipient_id:
            pending_events = [*current.pending_events, generated_event]
        agents.append(
            current.model_copy(
                update={
                    "outer_context": outer_context,
                    "pending_events": pending_events,
                }
            )
        )

    return _replace_scene(
        scene,
        agents=agents,
        rollback_stack=[
            *scene.rollback_stack,
            ConfirmedCallReference(
                call_id=confirmation.call_id,
                agent_id=agent_id,
                layer="outer",
            ),
        ],
        next_sequence=scene.next_sequence + 1,
    )


def rollback_latest_call(scene: Scene) -> Scene:
    """Undo exactly the global rollback-stack top confirmed model call."""
    if not scene.rollback_stack:
        raise SceneConflictError("There is no confirmed model call to undo.")

    reference = scene.rollback_stack[-1]
    if reference.layer == "inner":
        return _rollback_inner(scene, reference)
    return _rollback_outer(scene, reference)


def _rollback_inner(
    scene: Scene,
    reference: ConfirmedCallReference,
) -> Scene:
    """Remove the latest inner turn and restore its consumed batch at queue head."""  # noqa: E501
    agent = get_agent(scene, reference.agent_id)
    if (
        not agent.inner_context.turns
        or agent.inner_context.turns[-1].call_id != reference.call_id
        or len(agent.inner_context.turns) != len(agent.outer_context.turns) + 1
    ):
        raise SceneConflictError("The rollback stack is inconsistent.")
    turn = agent.inner_context.turns[-1]

    # Restore the consumed batch to the queue head, preserving its original
    # FIFO order ahead of any events that arrived later.
    restored = list(turn.consumed_events)
    agents = [
        current.model_copy(
            update={
                "inner_context": current.inner_context.model_copy(
                    update={"turns": current.inner_context.turns[:-1]}
                ),
                "pending_events": [
                    *restored,
                    *current.pending_events,
                ],
            }
        )
        if current.id == reference.agent_id
        else current
        for current in scene.agents
    ]
    return _replace_scene(
        scene,
        agents=agents,
        rollback_stack=scene.rollback_stack[:-1],
    )


def _rollback_outer(
    scene: Scene,
    reference: ConfirmedCallReference,
) -> Scene:
    """Remove the latest outer turn and any still-queued routed event."""
    sender = get_agent(scene, reference.agent_id)
    if (
        not sender.outer_context.turns
        or sender.outer_context.turns[-1].call_id != reference.call_id
        or len(sender.inner_context.turns) != len(sender.outer_context.turns)
    ):
        raise SceneConflictError("The rollback stack is inconsistent.")
    turn = sender.outer_context.turns[-1]
    if turn.recipient_id is not None:
        recipient = get_agent(scene, turn.recipient_id)
        if not any(
            event.id == turn.generated_event_id
            for event in recipient.pending_events
        ):
            raise SceneConflictError(
                "The routed event is no longer available for rollback."
            )

    agents = []
    for current in scene.agents:
        outer_context = current.outer_context
        pending_events = current.pending_events
        if current.id == reference.agent_id:
            outer_context = current.outer_context.model_copy(
                update={"turns": current.outer_context.turns[:-1]}
            )
        if turn.recipient_id is not None and current.id == turn.recipient_id:
            pending_events = [
                event
                for event in current.pending_events
                if event.id != turn.generated_event_id
            ]
        agents.append(
            current.model_copy(
                update={
                    "outer_context": outer_context,
                    "pending_events": pending_events,
                }
            )
        )

    return _replace_scene(
        scene,
        agents=agents,
        rollback_stack=scene.rollback_stack[:-1],
    )


def get_agent(scene: Scene, agent_id: AgentId) -> Agent:
    """Return one Agent by its fixed ID."""
    return next(agent for agent in scene.agents if agent.id == agent_id)


def require_agent_ready(agent: Agent) -> None:
    """Require prompt variables and complete routable relationships."""
    profile = agent.prompt_profile
    variables = (
        profile.pronoun,
        profile.hidden_beliefs,
        profile.inner_memories,
        profile.outer_memories,
    )
    if not all(value.strip() for value in variables):
        raise SceneConflictError(
            "All four prompt variables must be filled before model requests."
        )
    configured = [
        interaction
        for interaction in agent.interactions.values()
        if interaction.addresses
    ]
    if not configured:
        raise SceneConflictError(
            "At least one interaction address is required before model "
            "requests."
        )
    if any(not interaction.description.strip() for interaction in configured):
        raise SceneConflictError(
            "Every person with interaction addresses must have a "
            "description before model requests."
        )


def _find_event(
    scene: Scene,
    event_id: UUID,
) -> tuple[AgentId, Literal["pending", "consumed"], ExternalEvent]:
    """Locate the single authoritative occurrence of an event."""
    for agent in scene.agents:
        for event in agent.pending_events:
            if event.id == event_id:
                return agent.id, "pending", event
        for turn in agent.inner_context.turns:
            for consumed in turn.consumed_events:
                if consumed.id == event_id:
                    return agent.id, "consumed", consumed
    raise EventNotFoundError(f"Event '{event_id}' does not exist.")


def _require_new_call_id(scene: Scene, call_id: UUID) -> None:
    """Reject replay of a call identity already persisted in the scene."""
    if any(reference.call_id == call_id for reference in scene.rollback_stack):
        raise SceneConflictError("The model call was already confirmed.")


def _replace_scene(scene: Scene, **updates: Any) -> Scene:
    """Return a fully revalidated scene after a state transition."""
    values = {
        "schema": scene.schema_id,
        "id": scene.id,
        "name": scene.name,
        "model": scene.model,
        "agents": scene.agents,
        "rollback_stack": scene.rollback_stack,
        "next_sequence": scene.next_sequence,
    }
    values.update(updates)
    return Scene.model_validate(values)
