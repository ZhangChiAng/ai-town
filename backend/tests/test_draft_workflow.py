"""Tests for the protocol-neutral two-layer draft workflow."""

import asyncio
from copy import deepcopy
from typing import Any

import pytest

from app.draft_workflow import (
    DraftGenerationError,
    DraftWorkflow,
    confirm_draft,
    draft_state_token,
)
from app.models import (
    AgentId,
    ConfirmLayerRequest,
    InvalidLayerOutputError,
    Layer,
    LayerDraftResponse,
    ModelReasoningBlock,
    Scene,
    SceneConflictError,
    UpdateSceneRequest,
    add_manual_event,
    edit_manual_event,
    update_scene,
)
from app.prompts import build_system_prompt
from tests.helpers import (
    FakeBackend,
    confirmation,
    neutral_payload,
)
from tests.helpers import (
    create_prompted_scene as prompted_scene,
)

MODEL = "fake/model-Case"


def _generate(
    workflow: DraftWorkflow,
    scene: Scene,
    agent_id: AgentId,
    layer: Layer,
) -> LayerDraftResponse:
    """Run one async workflow generation without an async pytest plugin."""
    return asyncio.run(workflow.generate(scene, agent_id, layer))


def _versioned_workflow(
    output: str,
    version: int,
) -> DraftWorkflow:
    """Build a workflow whose backend envelope carries a version tag."""
    return DraftWorkflow(
        FakeBackend(
            [output],
            model=MODEL,
            payload_builder=lambda c, m: neutral_payload(c, m, version=version),
        ),
    )


def test_preview_exposes_neutral_context_without_backend_call() -> None:
    """Readable preview builds only neutral context and never calls backend."""
    scene = add_manual_event(
        prompted_scene("Preview", MODEL),
        "A",
        "a private event",
    )
    backend = FakeBackend([], model=MODEL)

    preview = DraftWorkflow(backend).preview(scene, "A", "inner")

    assert backend.conversations == []
    assert backend.generate_calls == []
    assert [(item.role, item.text) for item in preview.context] == [
        ("system", build_system_prompt(scene.agents[0], "inner")),
        ("user", "外部事件：\na private event"),
    ]
    assert preview.layer == "inner"
    assert list(preview.event_ids) == [
        event.id for event in scene.agents[0].pending_events
    ]


def test_generation_calls_once_and_maps_snapshot_and_observer_metadata() -> (
    None
):
    """One browser generation maps one backend call and its wire snapshot."""
    scene = add_manual_event(prompted_scene("Generate", MODEL), "C", "event")
    backend = FakeBackend(["inner answer"], model=MODEL)

    draft = _generate(DraftWorkflow(backend), scene, "C", "inner")

    assert len(backend.conversations) == 1
    assert len(backend.generate_calls) == 1
    assert draft.request_snapshot == neutral_payload(
        backend.generate_calls[0],
        MODEL,
    )
    assert draft.content == "inner answer"
    assert [block.model_dump() for block in draft.reasoning] == [
        {"type": "summary_text", "text": "observer-only fake reasoning"}
    ]
    assert draft.usage.model_dump() == {
        "input_tokens": 10,
        "output_tokens": 4,
        "cache_creation_input_tokens": 2,
        "cache_read_input_tokens": 3,
    }


def test_confirmation_has_no_backend_dependency_or_model_call() -> None:
    """A valid browser draft confirms after its backend is unavailable."""
    scene = add_manual_event(prompted_scene("Confirm", MODEL), "A", "event")
    backend = FakeBackend(["inner answer"], model=MODEL)
    draft = _generate(DraftWorkflow(backend), scene, "A", "inner")
    calls_before = len(backend.generate_calls)

    confirmed = confirm_draft(
        scene,
        "A",
        "inner",
        confirmation(draft, "edited inner answer"),
    )

    assert len(backend.generate_calls) == calls_before
    assert confirmed.agents[0].inner_context.turns[-1].output == (
        "edited inner answer"
    )


def test_backend_payload_changes_do_not_change_state_token() -> None:
    """Transport-only adapter changes cannot make a browser draft stale."""
    scene = add_manual_event(prompted_scene("Stable", MODEL), "B", "event")
    first = _generate(_versioned_workflow("answer", 1), scene, "B", "inner")
    second = _generate(_versioned_workflow("answer", 2), scene, "B", "inner")

    assert first.request_snapshot != second.request_snapshot
    assert first.state_token == second.state_token
    assert first.state_token == draft_state_token(scene, "B", "inner")


def test_event_prompt_history_or_model_changes_make_token_stale() -> None:
    """Every frozen business-context input participates in the state hash."""
    scene = add_manual_event(prompted_scene("Stale", MODEL), "A", "event one")
    draft = _generate(
        DraftWorkflow(FakeBackend(["inner one"], model=MODEL)),
        scene,
        "A",
        "inner",
    )

    changed_event = edit_manual_event(
        scene,
        "A",
        scene.agents[0].pending_events[0].id,
        "event two",
    )
    with pytest.raises(SceneConflictError, match="changed"):
        confirm_draft(
            changed_event,
            "A",
            "inner",
            confirmation(draft),
        )

    changed_prompt = _replace_inner_prompt(scene, "different prompt")
    assert draft_state_token(changed_prompt, "A", "inner") != (
        draft.state_token
    )
    changed_model = scene.model_copy(update={"model": "fake/other"})
    assert draft_state_token(changed_model, "A", "inner") != (draft.state_token)

    confirmed_inner = confirm_draft(
        scene,
        "A",
        "inner",
        confirmation(draft),
    )
    outer_draft = _generate(
        DraftWorkflow(FakeBackend(["对B说：first outer"], model=MODEL)),
        confirmed_inner,
        "A",
        "outer",
    )
    completed = confirm_draft(
        confirmed_inner,
        "A",
        "outer",
        confirmation(outer_draft),
    )
    next_scene = add_manual_event(completed, "A", "next event")
    original_token = draft_state_token(next_scene, "A", "inner")
    changed_history = _replace_previous_inner_output(next_scene, "changed")
    assert draft_state_token(changed_history, "A", "inner") != original_token


def test_readable_context_contains_complete_alternating_layer_history() -> None:
    """Preview derives every readable item from the neutral conversation."""
    scene = add_manual_event(prompted_scene("History", MODEL), "A", "first")
    inner = _generate(
        DraftWorkflow(FakeBackend(["inner one"], model=MODEL)),
        scene,
        "A",
        "inner",
    )
    scene = confirm_draft(scene, "A", "inner", confirmation(inner))
    outer = _generate(
        DraftWorkflow(FakeBackend(["对B说：outer one"], model=MODEL)),
        scene,
        "A",
        "outer",
    )
    scene = confirm_draft(scene, "A", "outer", confirmation(outer))
    scene = add_manual_event(scene, "A", "second")

    preview = DraftWorkflow(FakeBackend([], model=MODEL)).preview(
        scene,
        "A",
        "inner",
    )

    assert [(item.role, item.text) for item in preview.context[1:]] == [
        ("user", "外部事件：\nfirst"),
        ("assistant", "inner one"),
        (
            "user",
            "上一轮：\n你对B说：\nouter one\n\n外部事件：\nsecond",
        ),
    ]


def test_interaction_change_invalidates_later_inner_draft() -> None:
    """Any sender interaction change makes its browser draft stale."""
    scene = add_manual_event(
        prompted_scene("Rename stale", MODEL),
        "A",
        "first",
    )
    inner = _generate(
        DraftWorkflow(FakeBackend(["inner one"], model=MODEL)),
        scene,
        "A",
        "inner",
    )
    scene = confirm_draft(scene, "A", "inner", confirmation(inner))
    outer = _generate(
        DraftWorkflow(FakeBackend(["对B说：outer one"], model=MODEL)),
        scene,
        "A",
        "outer",
    )
    scene = confirm_draft(scene, "A", "outer", confirmation(outer))
    scene = add_manual_event(scene, "A", "second")
    draft = _generate(
        DraftWorkflow(FakeBackend(["inner two"], model=MODEL)),
        scene,
        "A",
        "inner",
    )
    update = UpdateSceneRequest.model_validate(
        {
            "name": scene.name,
            "agents": [
                {
                    "id": agent.id,
                    "name": agent.name,
                    "prompt_profile": agent.prompt_profile.model_dump(),
                    "interactions": (
                        {"B": {"儿子": "一般场合"}}
                        if agent.id == "A"
                        else agent.interactions
                    ),
                }
                for agent in scene.agents
            ],
        }
    )
    changed = update_scene(scene, update)

    assert draft_state_token(changed, "A", "inner") != draft.state_token
    with pytest.raises(SceneConflictError, match="changed"):
        confirm_draft(
            changed,
            "A",
            "inner",
            confirmation(draft),
        )


def test_generation_errors_are_sanitized_and_never_retried() -> None:
    """Both transport and invalid-output failures remain one-shot and safe."""
    scene = add_manual_event(prompted_scene("Errors", MODEL), "A", "event")
    transport = FakeBackend([RuntimeError("secret provider body")], model=MODEL)

    with pytest.raises(DraftGenerationError) as caught:
        _generate(DraftWorkflow(transport), scene, "A", "inner")

    assert str(caught.value) == "Model request failed."
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert len(transport.generate_calls) == 1
    invalid_outer = FakeBackend(["not addressed"], model=MODEL)
    inner = _generate(
        DraftWorkflow(FakeBackend(["inner"], model=MODEL)),
        scene,
        "A",
        "inner",
    )
    half_round = confirm_draft(scene, "A", "inner", confirmation(inner))

    with pytest.raises(DraftGenerationError, match="invalid outer draft"):
        _generate(DraftWorkflow(invalid_outer), half_round, "A", "outer")

    assert len(invalid_outer.generate_calls) == 1


def test_outer_confirmation_rejects_multiline_blank_body() -> None:
    """Edited outer output still needs non-blank text after the address."""
    scene = add_manual_event(prompted_scene("Blank body", MODEL), "A", "event")
    inner = _generate(
        DraftWorkflow(FakeBackend(["inner"], model=MODEL)),
        scene,
        "A",
        "inner",
    )
    scene = confirm_draft(scene, "A", "inner", confirmation(inner))
    outer = _generate(
        DraftWorkflow(FakeBackend(["对B说：first"], model=MODEL)),
        scene,
        "A",
        "outer",
    )

    with pytest.raises(InvalidLayerOutputError, match="semantic speech"):
        confirm_draft(scene, "A", "outer", confirmation(outer, "对B说：\n\n"))


def test_workflow_rejects_backend_for_a_different_scene_model() -> None:
    """Registry mistakes cannot route a scene through another model."""
    scene = add_manual_event(prompted_scene("Mismatch", MODEL), "A", "event")
    workflow = DraftWorkflow(
        FakeBackend(["answer"], model="fake/other"),
    )

    with pytest.raises(SceneConflictError, match="different"):
        _generate(workflow, scene, "A", "inner")


def _confirmation_with_reasoning(
    draft: LayerDraftResponse,
    reasoning: list[ModelReasoningBlock],
) -> ConfirmLayerRequest:
    """Build one confirmation DTO carrying the given reasoning blocks."""
    return ConfirmLayerRequest(
        call_id=draft.call_id,
        event_ids=list(draft.event_ids),
        content=draft.content,
        state_token=draft.state_token,
        reasoning=reasoning,
    )


def test_confirmed_turns_persist_reasoning_and_leave_it_out_of_requests() -> (
    None
):
    """Inner and outer confirmation writes reasoning onto saved turns only."""
    scene = add_manual_event(
        prompted_scene("Reasoning persist", MODEL),
        "A",
        "event",
    )
    inner_backend = FakeBackend(["inner"], model=MODEL)
    inner = _generate(DraftWorkflow(inner_backend), scene, "A", "inner")
    reasoning = [
        ModelReasoningBlock(type="thinking", text="inner thinking text"),
    ]
    scene = confirm_draft(
        scene,
        "A",
        "inner",
        _confirmation_with_reasoning(inner, reasoning),
    )
    assert scene.agents[0].inner_context.turns[-1].reasoning == reasoning

    outer_backend = FakeBackend(["对B说：outer"], model=MODEL)
    outer = _generate(DraftWorkflow(outer_backend), scene, "A", "outer")
    # The next request replays only input/output turns, never reasoning.
    assert "inner thinking text" not in repr(outer_backend.generate_calls[-1])
    scenario = confirm_draft(
        scene,
        "A",
        "outer",
        _confirmation_with_reasoning(outer, []),
    )
    assert scenario.agents[0].outer_context.turns[-1].reasoning == []
    assert scenario.agents[0].outer_context.turns[-1].output == "对B说：outer"


def test_empty_reasoning_still_confirms_both_layers() -> None:
    """A provider that returned no thinking confirms with an empty list."""
    scene = add_manual_event(
        prompted_scene("Empty reasoning", MODEL),
        "A",
        "event",
    )
    inner = _generate(
        DraftWorkflow(FakeBackend(["inner"], model=MODEL)),
        scene,
        "A",
        "inner",
    )
    scene = confirm_draft(
        scene,
        "A",
        "inner",
        _confirmation_with_reasoning(inner, []),
    )
    assert scene.agents[0].inner_context.turns[-1].reasoning == []
    outer = _generate(
        DraftWorkflow(FakeBackend(["对B说：outer"], model=MODEL)),
        scene,
        "A",
        "outer",
    )
    completed = confirm_draft(
        scene,
        "A",
        "outer",
        _confirmation_with_reasoning(outer, []),
    )
    assert completed.agents[0].outer_context.turns[-1].reasoning == []


def _replace_inner_prompt(scene: Any, prompt: str) -> Any:
    """Return a valid scene with only Agent A's inner memory changed."""
    agent = scene.agents[0]
    changed = agent.model_copy(
        update={
            "prompt_profile": agent.prompt_profile.model_copy(
                update={"inner_memories": prompt}
            ),
        }
    )
    return scene.model_copy(update={"agents": [changed, *scene.agents[1:]]})


def _replace_previous_inner_output(scene: Any, output: str) -> Any:
    """Return a valid scene with one confirmed inner history output changed."""
    copied = deepcopy(scene)
    agent = copied.agents[0]
    turn = agent.inner_context.turns[0].model_copy(update={"output": output})
    inner_context = agent.inner_context.model_copy(update={"turns": [turn]})
    changed = agent.model_copy(update={"inner_context": inner_context})
    return copied.model_copy(update={"agents": [changed, *copied.agents[1:]]})
