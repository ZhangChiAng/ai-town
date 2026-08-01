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

SCHEMA_VERSION = 6
AGENT_IDS = ("A", "B", "C")

DEFAULT_INNER_SYSTEM_PROMPT = """\
你是一个人的内层人格，是这个人内在更直接、感性和原始的一面。
你知道自己是内层人格，只对同一个人的外层人格说话。

相对稳定的欲望与恐惧塑造你如何解释经验。
你会特别留意触动这些欲望与恐惧的信息，并从自身感受出发形成联想、
猜测、评价和冲动。你的判断可以片面、情绪化、自相矛盾，
也可以随着新经验动摇；不要假装自己拥有客观全知的视角。

像脑中自然冒出的声音一样表达。你可以提醒、催促、抱怨、怀疑、
诱惑或质问外层人格，自然地向它施加压力，
但不必把欲望与恐惧复述成标签。

你只能表达内在判断、感受和冲动。你不能在现实中行动，
不能直接向其他人发送消息，也不能替外层人格决定最终行动。
外层人格可以同意、反驳、压制或重新理解你。

输出非空的自然文本，可以使用多行。
不要使用 `To X:` 或 `From X:` 地址格式。"""

DEFAULT_OUTER_SYSTEM_PROMPT_TEMPLATE = """\
你是一个生活在现实关系中的人，也是 Agent {agent_id} 的外层人格。
请在这份完整提示词中直接写明你的现实身份、经历、关系和
相对稳定的行为方式。

你负责理解现实局面、与他人交流并承担行动后果。
每轮你会听见内层人格自然冒出的声音；
它是带有欲望、恐惧和主观偏见的内在意见，不是事实或命令。
你可以顺从它，也可以怀疑、反驳、压制、修饰或重新理解它。

你只能看到请求中明确给出的外部事件、内层声音以及
你自己的外层历史。你无法读取内层人格的隐藏提示、
其他 Agent 的上下文、观察者信息或未发送的内容，
不要假装知道这些信息。

你的自我理解来自有限经验、他人反馈和对自身行为的观察，
因此也可能不完整或出错。像真实的人一样作出选择；
可以坦率、试探、回避、嘴硬、推诿、隐瞒、撒谎或保持沉默，
但表达应符合你此刻的判断并由你承担后果。

你本人的固定地址是 Agent {agent_id}。
每次只向 {recipient_ids} 中的一位发送消息。
输出必须恰好是一行 `To X: 正文`，
其中 X 是接收者的 Agent ID；不能发给自己。
正文只写真正对外说出的话，不要输出心理分析、推理过程、
层级标签或括号包裹的动作。"""

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


def _require_non_blank_prompt(value: str) -> str:
    """Require a prompt without changing its exact saved text."""
    if not value.strip():
        raise ValueError("system_prompt must not be blank")
    return value


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


def default_outer_system_prompt(agent_id: AgentId) -> str:
    """Return the complete default outer prompt for one Agent ID."""
    recipient_ids = "、".join(
        candidate for candidate in AGENT_IDS if candidate != agent_id
    )
    return DEFAULT_OUTER_SYSTEM_PROMPT_TEMPLATE.format(
        agent_id=agent_id,
        recipient_ids=recipient_ids,
    )


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
    """One confirmed inner-layer model call."""

    call_id: UUID
    event_id: UUID
    sequence: int = Field(ge=1)
    input: str
    output: str
    consumed_event: ExternalEvent

    _validate_input = field_validator("input")(_strip_non_blank)
    _validate_output = field_validator("output")(_strip_non_blank)

    @model_validator(mode="after")
    def validate_event_reference(self) -> Self:
        """Require the turn to reference its embedded consumed event."""
        if self.event_id != self.consumed_event.id:
            raise ValueError("inner turn event_id must match consumed_event")
        return self


class OuterTurn(ApiModel):
    """One confirmed outer-layer model call and its routed event reference."""

    call_id: UUID
    event_id: UUID
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

    _validate_system_prompt = field_validator("system_prompt")(
        _require_non_blank_prompt
    )


class OuterContext(ApiModel):
    """Independent system prompt and complete confirmed outer history."""

    system_prompt: str
    turns: list[OuterTurn] = Field(default_factory=list)

    _validate_system_prompt = field_validator("system_prompt")(
        _require_non_blank_prompt
    )


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
            if inner_turn.event_id != outer_turn.event_id:
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
    """A schema-v6 scene containing exactly three two-layer Agents."""

    schema_version: Literal[6] = SCHEMA_VERSION
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
                _record_event(events, agent.id, turn.consumed_event)
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
                max_sequence = max(
                    max_sequence,
                    turn.sequence,
                    turn.consumed_event.sequence,
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

    _validate_system_prompt = field_validator("system_prompt")(
        _require_non_blank_prompt
    )


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
    event_id: UUID
    content: str
    reasoning: list[ModelReasoningBlock]
    usage: TokenUsage
    request_snapshot: dict[str, Any]
    state_token: str = Field(min_length=64, max_length=64)

    _validate_content = field_validator("content")(_strip_non_blank)


class ConfirmLayerRequest(ApiModel):
    """Browser-held generation identity and editable confirmed output."""

    call_id: UUID
    event_id: UUID
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
    event_id: UUID
    context: list[ModelRequestContextItem]


class SceneConflictError(RuntimeError):
    """Raised when an operation conflicts with the current scene state."""


class InvalidLayerOutputError(ValueError):
    """Raised when edited output violates the selected layer contract."""


class EventNotFoundError(LookupError):
    """Raised when a scene has no event with the requested ID."""


class SceneModelBindingConflictError(RuntimeError):
    """Raised when replacing an existing scene model binding."""


def create_scene(name: str, model: str) -> Scene:
    """Create an empty schema-v6 scene bound to one configured model."""
    return Scene(
        id=uuid4(),
        name=name,
        model=model,
        agents=[
            Agent(
                id=agent_id,
                name=agent_id,
                inner_context=InnerContext(
                    system_prompt=DEFAULT_INNER_SYSTEM_PROMPT
                ),
                outer_context=OuterContext(
                    system_prompt=default_outer_system_prompt(agent_id)
                ),
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


def build_inner_input(scene: Scene, agent_id: AgentId) -> tuple[UUID, str]:
    """Build the exact next inner user text from the current FIFO head."""
    agent = get_agent(scene, agent_id)
    if len(agent.inner_context.turns) != len(agent.outer_context.turns):
        raise SceneConflictError("The Agent is waiting for outer confirmation.")
    if not agent.pending_events:
        raise SceneConflictError("The Agent has no pending event.")

    event = agent.pending_events[0]
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
    sections.append("外部事件：\n" + event.content)
    return event.id, "\n\n".join(sections)


def build_outer_input(scene: Scene, agent_id: AgentId) -> tuple[UUID, str]:
    """Build the exact outer user text for a confirmed inner half-round."""
    agent = get_agent(scene, agent_id)
    if len(agent.inner_context.turns) != len(agent.outer_context.turns) + 1:
        raise SceneConflictError(
            "The Agent has no confirmed inner turn awaiting outer output."
        )
    inner_turn = agent.inner_context.turns[-1]
    text = (
        f"外部事件：\n{inner_turn.consumed_event.content}\n\n"
        f"你内心有一个声音：\n{inner_turn.output}"
    )
    return inner_turn.event_id, text


def confirm_inner_turn(
    scene: Scene,
    agent_id: AgentId,
    confirmation: ConfirmLayerRequest,
    actual_input: str,
) -> Scene:
    """Consume the FIFO head and append one confirmed inner turn."""
    event_id, expected_input = build_inner_input(scene, agent_id)
    if confirmation.event_id != event_id or actual_input != expected_input:
        raise SceneConflictError("The inner draft is stale.")
    _require_new_call_id(scene, confirmation.call_id)

    agent = get_agent(scene, agent_id)
    event = agent.pending_events[0]
    turn = InnerTurn(
        call_id=confirmation.call_id,
        event_id=event.id,
        sequence=scene.next_sequence,
        input=actual_input,
        output=confirmation.content,
        consumed_event=event,
    )
    agents = [
        current.model_copy(
            update={
                "inner_context": current.inner_context.model_copy(
                    update={"turns": [*current.inner_context.turns, turn]}
                ),
                "pending_events": current.pending_events[1:],
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
    event_id, expected_input = build_outer_input(scene, agent_id)
    if confirmation.event_id != event_id or actual_input != expected_input:
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
        event_id=event_id,
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
    """Remove the latest inner turn and restore its event at queue head."""
    agent = get_agent(scene, reference.agent_id)
    if (
        not agent.inner_context.turns
        or agent.inner_context.turns[-1].call_id != reference.call_id
        or len(agent.inner_context.turns) != len(agent.outer_context.turns) + 1
    ):
        raise SceneConflictError("The rollback stack is inconsistent.")
    turn = agent.inner_context.turns[-1]

    agents = [
        current.model_copy(
            update={
                "inner_context": current.inner_context.model_copy(
                    update={"turns": current.inner_context.turns[:-1]}
                ),
                "pending_events": [
                    turn.consumed_event,
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
            if turn.consumed_event.id == event_id:
                return agent.id, "consumed", turn.consumed_event
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
