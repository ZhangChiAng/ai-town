"""Protocol-neutral preview, generation, and confirmation workflow."""

import hashlib
import hmac
import json
import logging
from dataclasses import dataclass
from typing import Literal
from uuid import UUID, uuid4

from app.model_backends.contracts import (
    ModelBackend,
    ModelConversation,
    ModelGeneration,
    ModelTurn,
)
from app.models import (
    AgentId,
    ConfirmLayerRequest,
    ExternalEvent,
    InvalidLayerOutputError,
    Layer,
    LayerDraftResponse,
    ModelReasoningBlock,
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
ContextRole = Literal["system", "user", "assistant"]


class DraftGenerationError(RuntimeError):
    """Raised when an upstream call cannot produce a safe browser draft."""


@dataclass(frozen=True, slots=True)
class DraftContextItem:
    """One readable item in the exact context sent to a backend."""

    role: ContextRole
    text: str


@dataclass(frozen=True, slots=True)
class DraftPreview:
    """Protocol-neutral readable context for one model request."""

    layer: Layer
    event_ids: tuple[UUID, ...]
    context: tuple[DraftContextItem, ...]


@dataclass(frozen=True, slots=True)
class _DraftCallContext:
    """Protocol-neutral business context for one preview or generation."""

    event_ids: tuple[UUID, ...]
    events: tuple[ExternalEvent, ...]
    conversation: ModelConversation


class DraftWorkflow:
    """Drive model-facing operations through one protocol-neutral backend."""

    def __init__(self, backend: ModelBackend) -> None:
        """Bind the workflow to one immutable configured backend."""
        self._backend = backend
        self._model = backend.model

    @property
    def model(self) -> str:
        """Return the exact model served by this workflow."""
        return self._model

    def preview(
        self,
        scene: Scene,
        agent_id: AgentId,
        layer: Layer,
    ) -> DraftPreview:
        """Build the next readable context without touching the backend."""
        draft = self._build_call_context(scene, agent_id, layer)
        return DraftPreview(
            layer=layer,
            event_ids=draft.event_ids,
            context=build_readable_context(draft.conversation),
        )

    async def generate(
        self,
        scene: Scene,
        agent_id: AgentId,
        layer: Layer,
    ) -> LayerDraftResponse:
        """Build context, call once, and return an unpersisted browser draft."""
        draft = self._build_call_context(scene, agent_id, layer)
        generation: ModelGeneration | None = None
        backend_failed = False
        try:
            generation = await self._backend.generate(draft.conversation)
        except Exception as exc:
            # Provider bodies and credentials never cross this boundary; log
            # only the safe business context plus exception type. The backend
            # layer is responsible for recording its own safe details; here we
            # avoid echoing upstream exception strings or tracebacks, which may
            # carry URLs, request bodies, or credentials.
            LOGGER.warning(
                "model generation failed: scene=%s agent=%s layer=%s "
                "event=%s model=%s exc_type=%s",
                scene.id,
                agent_id,
                layer,
                draft.event_ids,
                self._model,
                type(exc).__name__,
            )
            backend_failed = True

        if backend_failed or generation is None:
            # Raise outside the handler so no provider exception stays linked.
            raise DraftGenerationError("Model request failed.") from None

        invalid_outer = False
        if layer == "outer":
            try:
                parse_addressed_message(generation.content, agent_id)
            except ValueError:
                invalid_outer = True
        if invalid_outer:
            # Visible output validation is distinct from backend failures; the
            # raw LLM content is the only way to diagnose format problems.
            LOGGER.warning(
                "invalid outer draft: scene=%s agent=%s event=%s model=%s "
                "content=%r",
                scene.id,
                agent_id,
                draft.event_ids,
                self._model,
                generation.content,
            )
            raise DraftGenerationError(
                "Model returned an invalid outer draft."
            ) from None

        usage = TokenUsage(
            input_tokens=generation.usage.input_tokens,
            output_tokens=generation.usage.output_tokens,
            cache_creation_input_tokens=(
                generation.usage.cache_creation_input_tokens
            ),
            cache_read_input_tokens=generation.usage.cache_read_input_tokens,
        )
        result = LayerDraftResponse(
            layer=layer,
            call_id=uuid4(),
            event_ids=list(draft.event_ids),
            content=generation.content,
            reasoning=[
                ModelReasoningBlock(type=block.type, text=block.text)
                for block in generation.reasoning
            ],
            usage=usage,
            request_snapshot=generation.request_snapshot,
            state_token=_state_token(
                scene.model,
                agent_id,
                layer,
                draft.events,
                draft.conversation,
            ),
        )
        _log_usage(scene.id, agent_id, layer, self.model, usage)
        return result

    def _build_call_context(
        self,
        scene: Scene,
        agent_id: AgentId,
        layer: Layer,
    ) -> _DraftCallContext:
        """Build one protocol-neutral conversation exactly once."""
        _require_matching_model(scene, self.model)
        event_ids, events, conversation = build_model_conversation(
            scene,
            agent_id,
            layer,
        )
        return _DraftCallContext(
            event_ids=event_ids,
            events=events,
            conversation=conversation,
        )


def build_model_conversation(
    scene: Scene,
    agent_id: AgentId,
    layer: Layer,
) -> tuple[tuple[UUID, ...], tuple[ExternalEvent, ...], ModelConversation]:
    """Build only the selected persona layer's complete visible context."""
    agent = get_agent(scene, agent_id)
    if layer == "inner":
        event_ids, current_input = build_inner_input(scene, agent_id)
        events = tuple(agent.pending_events)
        system_prompt = agent.inner_context.system_prompt
        saved_turns = agent.inner_context.turns
    else:
        event_ids, current_input = build_outer_input(scene, agent_id)
        inner_turn = agent.inner_context.turns[-1]
        events = tuple(inner_turn.consumed_events)
        system_prompt = agent.outer_context.system_prompt
        saved_turns: list[OuterTurn] = agent.outer_context.turns

    if not system_prompt.strip():
        # Blank prompts are saveable scene state but never reach a model.
        raise SceneConflictError(
            f"The {layer} system prompt must be filled before model requests."
        )

    conversation = ModelConversation(
        system_prompt=system_prompt,
        turns=tuple(
            ModelTurn(input=turn.input, output=turn.output)
            for turn in saved_turns
        ),
        current_input=current_input,
    )
    return tuple(event_ids), events, conversation


def build_readable_context(
    conversation: ModelConversation,
) -> tuple[DraftContextItem, ...]:
    """Return system, complete alternating history, and current user input."""
    context = [DraftContextItem(role="system", text=conversation.system_prompt)]
    for turn in conversation.turns:
        context.extend(
            (
                DraftContextItem(role="user", text=turn.input),
                DraftContextItem(role="assistant", text=turn.output),
            )
        )
    context.append(
        DraftContextItem(role="user", text=conversation.current_input)
    )
    return tuple(context)


def draft_state_token(
    scene: Scene,
    agent_id: AgentId,
    layer: Layer,
) -> str:
    """Hash protocol-neutral business context for stale-draft detection."""
    if scene.model is None:
        raise SceneConflictError(
            "Scene must be bound to a model before draft confirmation."
        )
    _event_ids, events, conversation = build_model_conversation(
        scene,
        agent_id,
        layer,
    )
    return _state_token(
        scene.model,
        agent_id,
        layer,
        events,
        conversation,
    )


def confirm_draft(
    scene: Scene,
    agent_id: AgentId,
    layer: Layer,
    confirmation: ConfirmLayerRequest,
) -> Scene:
    """Confirm against current business state without resolving a backend."""
    event_ids, _events, conversation = build_model_conversation(
        scene,
        agent_id,
        layer,
    )
    if list(confirmation.event_ids) != list(event_ids):
        raise SceneConflictError("The draft event is no longer current.")
    current_token = _state_token(
        scene.model,
        agent_id,
        layer,
        _events,
        conversation,
    )
    if not hmac.compare_digest(confirmation.state_token, current_token):
        raise SceneConflictError(
            "The scene changed after this draft was generated."
        )

    if layer == "inner":
        return confirm_inner_turn(
            scene,
            agent_id,
            confirmation,
            conversation.current_input,
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
        conversation.current_input,
    )


def _require_matching_model(scene: Scene, backend_model: str) -> None:
    """Prevent an available backend from serving a differently bound scene."""
    if scene.model is None:
        raise SceneConflictError(
            "Scene must be bound to a model before model requests."
        )
    if scene.model != backend_model:
        raise SceneConflictError(
            "The scene is bound to a different configured model."
        )


def _state_token(
    model: str | None,
    agent_id: AgentId,
    layer: Layer,
    events: tuple[ExternalEvent, ...],
    conversation: ModelConversation,
) -> str:
    """Hash only model identity and protocol-neutral conversation state."""
    if model is None:
        raise SceneConflictError(
            "Scene must be bound to a model before draft confirmation."
        )
    state = {
        "model": model,
        "agent_id": agent_id,
        "layer": layer,
        "events": [event.model_dump(mode="json") for event in events],
        "system_prompt": conversation.system_prompt,
        "turns": [
            {"input": turn.input, "output": turn.output}
            for turn in conversation.turns
        ],
        "current_input": conversation.current_input,
    }
    serialized = json.dumps(
        state,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


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
