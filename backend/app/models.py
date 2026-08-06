"""Domain models and state transitions for two-layer AI Town scenes."""

import re
from copy import deepcopy
from typing import Any, Literal, Self
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

SCHEMA_VERSION = 7
AGENT_IDS = ("A", "B", "C")
LEGACY_SCHEMA_VERSION = 6

AgentId = Literal["A", "B", "C"]
Layer = Literal["inner", "outer"]
EventKind = Literal["manual", "agent_message"]


class ApiModel(BaseModel):
    """Base API model that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid")


def _strip_non_blank(value: str) -> str:
    """Strip surrounding whitespace and require non-empty text."""
    stripped = value.strip()
    if not stripped:
        raise ValueError("text must not be blank")
    return stripped


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


_ADDRESSED_MESSAGE_PATTERN = re.compile(
    r"\ATo\s+([A-C])\s*[:：]\s*(\S(?:[^\r\n]*\S)?)\s*\Z"
)


def parse_addressed_message(
    content: str,
    sender_id: AgentId,
) -> tuple[AgentId, str]:
    """Parse one visible ``To <AgentId>: <body>`` message."""
    if "\n" in content or "\r" in content:
        raise ValueError("content must contain exactly one line")
    match = _ADDRESSED_MESSAGE_PATTERN.fullmatch(content)
    if match is None:
        raise ValueError(
            "content must be one line in the form 'To B: message body'"
        )
    recipient_id = match.group(1)
    if recipient_id == sender_id:
        raise ValueError("message recipient must differ from sender")
    return recipient_id, match.group(2)


class TokenUsage(ApiModel):
    """Provider token usage for one generation."""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_creation_input_tokens: int = Field(ge=0)
    cache_read_input_tokens: int = Field(ge=0)


class ModelReasoningBlock(ApiModel):
    """One readable provider reasoning block shown only in the browser."""

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
    source_agent_id: AgentId | None = None
    source_call_id: UUID | None = None

    _validate_content = field_validator("content")(_strip_non_blank)

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

    _validate_input = field_validator("input")(_strip_non_blank)
    _validate_output = field_validator("output")(_strip_non_blank)

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
    """One confirmed outer-layer model call and its routed event reference.

    ``event_ids`` mirrors the event batch consumed by the matching inner
    turn. ``generated_event_id`` is the routed message that this outer turn
    produces for its recipient's queue.
    """

    call_id: UUID
    event_ids: list[UUID]
    sequence: int = Field(ge=1)
    input: str
    output: str
    recipient_id: AgentId
    generated_event_id: UUID

    _validate_input = field_validator("input")(_strip_non_blank)
    _validate_output = field_validator("output")(_strip_non_blank)


class InnerContext(ApiModel):
    """Independent system prompt and complete confirmed inner history."""

    system_prompt: str
    turns: list[InnerTurn] = Field(default_factory=list)


class OuterContext(ApiModel):
    """Independent system prompt and complete confirmed outer history."""

    system_prompt: str
    turns: list[OuterTurn] = Field(default_factory=list)


class Agent(ApiModel):
    """One Agent with isolated inner/outer contexts and a FIFO event queue."""

    id: AgentId
    name: str
    inner_context: InnerContext
    outer_context: OuterContext
    pending_events: list[ExternalEvent] = Field(default_factory=list)

    _validate_name = field_validator("name")(_strip_non_blank_name)

    @model_validator(mode="after")
    def validate_round_alignment(self) -> Self:
        """Allow only complete rounds or one confirmed inner half-round."""
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
    """A schema-v7 scene containing exactly three two-layer Agents.

    Schema v6 raw files are migrated to v7 by ``migrate_v6_to_v7`` in the
    storage layer before being validated as a ``Scene``. Earlier schemas
    (v1-v5) are rejected without migration.
    """

    schema_version: Literal[7] = SCHEMA_VERSION
    id: UUID
    name: str
    model: str | None
    agents: list[Agent] = Field(min_length=3, max_length=3)
    rollback_stack: list[ConfirmedCallReference] = Field(default_factory=list)
    next_sequence: int = Field(default=1, ge=1)

    _validate_name = field_validator("name")(_strip_non_blank_name)

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str | None) -> str | None:
        """Allow an explicit one-time unbound state or one model name."""
        if value is None:
            return None
        return _strip_non_blank_model(value)

    @model_validator(mode="after")
    def validate_scene_integrity(self) -> Self:
        """Validate IDs, call order, event ownership, and routing references."""
        if tuple(agent.id for agent in self.agents) != AGENT_IDS:
            raise ValueError("agents must contain A, B, and C in order")

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
                recipient_id, body = parse_addressed_message(
                    turn.output,
                    agent.id,
                )
                if (
                    recipient_id != turn.recipient_id
                    or turn.output != f"To {recipient_id}: {body}"
                ):
                    raise ValueError("outer output must be canonical")
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
            event_location = events.get(turn.generated_event_id)
            if event_location is None:
                raise ValueError("routed event is missing")
            recipient_id, event = event_location
            _recipient, body = parse_addressed_message(turn.output, sender_id)
            if (
                recipient_id != turn.recipient_id
                or event.kind != "agent_message"
                or event.source_agent_id != sender_id
                or event.source_call_id != turn.call_id
                or event.sequence != turn.sequence
                or event.content != f"From {sender_id}: {body}"
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


class BindSceneModelRequest(ApiModel):
    """One-time model binding payload for an unbound schema-v6 scene."""

    model: str

    _validate_model = field_validator("model")(_strip_non_blank_model)


class ModelOption(ApiModel):
    """One public model choice without protocol or endpoint details."""

    model: str


class ModelOptionsResponse(ApiModel):
    """Configured model choices in stable TOML order."""

    options: list[ModelOption]


class ContextUpdate(ApiModel):
    """Writable portion of one independent persona context."""

    system_prompt: str


class AgentUpdate(ApiModel):
    """Writable Agent identity and complete inner/outer prompt texts."""

    id: AgentId
    name: str
    inner_context: ContextUpdate
    outer_context: ContextUpdate

    _validate_name = field_validator("name")(_strip_non_blank_name)


class UpdateSceneRequest(ApiModel):
    """Payload for editing scene and Agent prompt text."""

    name: str
    agents: list[AgentUpdate] = Field(min_length=3, max_length=3)

    _validate_name = field_validator("name")(_strip_non_blank_name)

    @model_validator(mode="after")
    def validate_agent_ids(self) -> Self:
        """Require editable Agents in stable A/B/C order."""
        if tuple(agent.id for agent in self.agents) != AGENT_IDS:
            raise ValueError("agents must contain A, B, and C in order")
        return self


class EventContentRequest(ApiModel):
    """Payload for creating or editing a manual event."""

    content: str

    _validate_content = field_validator("content")(_strip_non_blank)


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

    _validate_content = field_validator("content")(_strip_non_blank)


class ConfirmLayerRequest(ApiModel):
    """Browser-held generation identity and editable confirmed output.

    ``event_ids`` is the ordered batch this confirmation must match against
    the rebuilt ``build_inner_input`` / ``build_outer_input`` state.
    """

    call_id: UUID
    event_ids: list[UUID]
    content: str
    state_token: str = Field(min_length=64, max_length=64)

    _validate_content = field_validator("content")(_strip_non_blank)


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


class SceneModelBindingConflictError(RuntimeError):
    """Raised when replacing an existing scene model binding."""


def _wrap_v6_inner_turn(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert one legacy single-event inner turn to the v7 batch form."""
    if "consumed_events" in raw:
        return raw
    consumed = raw.pop("consumed_event", None)
    event_id = raw.pop("event_id", None)
    if consumed is None or event_id is None:
        raise ValueError("legacy inner turn must have consumed_event")
    raw["event_ids"] = [event_id]
    raw["consumed_events"] = [consumed]
    return raw


def _wrap_v6_outer_turn(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert one legacy single-event outer turn to the v7 batch form."""
    if "event_ids" in raw:
        return raw
    event_id = raw.pop("event_id", None)
    if event_id is None:
        raise ValueError("legacy outer turn must have event_id")
    raw["event_ids"] = [event_id]
    return raw


def migrate_v6_to_v7(raw: dict[str, Any]) -> dict[str, Any]:
    """Migrate one parsed schema-v6 raw scene to the schema-v7 batch form.

    The migration is shallow and structural: it wraps each inner turn's
    legacy ``consumed_event`` and ``event_id`` into single-element lists
    and each outer turn's ``event_id`` into ``event_ids``. It does not
    re-validate the scene; the caller (storage) runs ``Scene`` validation
    on the migrated value. ``schema_version`` is set to 7.
    """
    migrated = deepcopy(raw)
    for agent in migrated.get("agents", []):
        inner_context = agent.get("inner_context", {})
        for turn in inner_context.get("turns", []):
            _wrap_v6_inner_turn(turn)
        outer_context = agent.get("outer_context", {})
        for turn in outer_context.get("turns", []):
            _wrap_v6_outer_turn(turn)
    migrated["schema_version"] = SCHEMA_VERSION
    return migrated


def create_scene(name: str, model: str) -> Scene:
    """Create an empty schema-v7 scene bound to one configured model."""
    return Scene(
        id=uuid4(),
        name=name,
        model=model,
        agents=[
            Agent(
                id=agent_id,
                name=agent_id,
                inner_context=InnerContext(system_prompt=""),
                outer_context=OuterContext(system_prompt=""),
            )
            for agent_id in AGENT_IDS
        ],
    )


def bind_scene_model(scene: Scene, model: str) -> Scene:
    """Bind an unbound scene once without allowing later replacement."""
    if scene.model is not None:
        raise SceneModelBindingConflictError(
            f"Scene '{scene.id}' already has a model binding."
        )
    return _replace_scene(scene, model=_strip_non_blank_model(model))


def update_scene(scene: Scene, update: UpdateSceneRequest) -> Scene:
    """Apply editable fields while preserving every event and confirmed turn."""
    agents = [
        Agent(
            id=updated.id,
            name=updated.name,
            inner_context=InnerContext(
                system_prompt=updated.inner_context.system_prompt,
                turns=current.inner_context.turns,
            ),
            outer_context=OuterContext(
                system_prompt=updated.outer_context.system_prompt,
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
    presented in FIFO order under a single ``外部事件：`` block, separated
    by blank lines. The optional prior outer-speech preamble is retained
    verbatim and remains above the event block.
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
        recipient = get_agent(scene, previous_outer.recipient_id)
        _parsed_recipient_id, body = parse_addressed_message(
            previous_outer.output,
            agent_id,
        )
        # Present the prior speech as routed dialogue, without its protocol tag.
        sections.append(
            "外层人格上一轮对 "
            f"Agent {previous_outer.recipient_id}（{recipient.name}）说：\n"
            f"{body}"
        )
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
    """Append one outer turn and atomically route its generated event."""
    event_ids, expected_input = build_outer_input(scene, agent_id)
    if confirmation.event_ids != event_ids or actual_input != expected_input:
        raise SceneConflictError("The outer draft is stale.")
    _require_new_call_id(scene, confirmation.call_id)

    recipient_id, body = parse_addressed_message(
        confirmation.content,
        agent_id,
    )
    canonical_output = f"To {recipient_id}: {body}"
    generated_event = ExternalEvent(
        id=uuid4(),
        sequence=scene.next_sequence,
        kind="agent_message",
        content=f"From {agent_id}: {body}",
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
        generated_event_id=generated_event.id,
    )

    agents = []
    for current in scene.agents:
        outer_context = current.outer_context
        pending_events = current.pending_events
        if current.id == agent_id:
            outer_context = current.outer_context.model_copy(
                update={"turns": [*current.outer_context.turns, turn]}
            )
        if current.id == recipient_id:
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
    """Remove the latest outer turn and its still-queued routed event."""
    sender = get_agent(scene, reference.agent_id)
    if (
        not sender.outer_context.turns
        or sender.outer_context.turns[-1].call_id != reference.call_id
        or len(sender.inner_context.turns) != len(sender.outer_context.turns)
    ):
        raise SceneConflictError("The rollback stack is inconsistent.")
    turn = sender.outer_context.turns[-1]
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
        if current.id == turn.recipient_id:
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
        "schema_version": scene.schema_version,
        "id": scene.id,
        "name": scene.name,
        "model": scene.model,
        "agents": scene.agents,
        "rollback_stack": scene.rollback_stack,
        "next_sequence": scene.next_sequence,
    }
    values.update(updates)
    return Scene.model_validate(values)
