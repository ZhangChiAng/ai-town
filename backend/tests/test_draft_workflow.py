"""Tests for the protocol-neutral two-layer draft workflow."""

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
    SceneConflictError,
    add_manual_event,
    create_scene,
    edit_manual_event,
)
from tests.helpers import (
    FakeBackend,
    confirmation,
    neutral_payload,
)

MODEL = "fake/model-Case"


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


def test_preview_exposes_neutral_context_and_exact_fake_payload() -> None:
    """Readable preview never interprets backend-specific request fields."""
    scene = add_manual_event(
        create_scene("Preview", MODEL),
        "A",
        "a private event",
    )
    backend = FakeBackend([], model=MODEL)

    preview = DraftWorkflow(backend).preview(scene, "A", "inner")

    assert len(backend.conversations) == 1
    assert backend.generate_calls == []
    assert [(item.role, item.text) for item in preview.context] == [
        ("system", scene.agents[0].inner_context.system_prompt),
        ("user", "外部事件：\na private event"),
    ]
    assert preview.request == {
        "engine": MODEL,
        "version": 1,
        "dialogue": {
            "rulebook": scene.agents[0].inner_context.system_prompt,
            "past": [],
            "next": "外部事件：\na private event",
        },
    }


def test_generation_prepares_once_calls_once_and_maps_observer_metadata() -> (
    None
):
    """One browser generation is one preparation and one backend call."""
    scene = add_manual_event(create_scene("Generate", MODEL), "C", "event")
    backend = FakeBackend(["inner answer"], model=MODEL)

    draft = DraftWorkflow(backend).generate(scene, "C", "inner")

    assert len(backend.conversations) == 1
    assert len(backend.generate_calls) == 1
    assert backend.generate_calls[0].payload == draft.request_snapshot
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
    scene = add_manual_event(create_scene("Confirm", MODEL), "A", "event")
    backend = FakeBackend(["inner answer"], model=MODEL)
    draft = DraftWorkflow(backend).generate(scene, "A", "inner")
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
    scene = add_manual_event(create_scene("Stable", MODEL), "B", "event")
    first = _versioned_workflow("answer", 1).generate(scene, "B", "inner")
    second = _versioned_workflow("answer", 2).generate(scene, "B", "inner")

    assert first.request_snapshot != second.request_snapshot
    assert first.state_token == second.state_token
    assert first.state_token == draft_state_token(scene, "B", "inner")


def test_event_prompt_history_or_model_changes_make_token_stale() -> None:
    """Every frozen business-context input participates in the state hash."""
    scene = add_manual_event(create_scene("Stale", MODEL), "A", "event one")
    draft = DraftWorkflow(
        FakeBackend(["inner one"], model=MODEL),
    ).generate(scene, "A", "inner")

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
    outer_draft = DraftWorkflow(
        FakeBackend(["To B: first outer"], model=MODEL),
    ).generate(confirmed_inner, "A", "outer")
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
    scene = add_manual_event(create_scene("History", MODEL), "A", "first")
    inner = DraftWorkflow(
        FakeBackend(["inner one"], model=MODEL),
    ).generate(scene, "A", "inner")
    scene = confirm_draft(scene, "A", "inner", confirmation(inner))
    outer = DraftWorkflow(
        FakeBackend(["To B: outer one"], model=MODEL),
    ).generate(scene, "A", "outer")
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
            "外层人格：\nTo B: outer one\n\n外部事件：\nsecond",
        ),
    ]


def test_generation_errors_are_sanitized_and_never_retried() -> None:
    """Both transport and invalid-output failures remain one-shot and safe."""
    scene = add_manual_event(create_scene("Errors", MODEL), "A", "event")
    transport = FakeBackend([RuntimeError("secret provider body")], model=MODEL)

    with pytest.raises(DraftGenerationError) as caught:
        DraftWorkflow(transport).generate(scene, "A", "inner")

    assert str(caught.value) == "Model request failed."
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert len(transport.generate_calls) == 1
    invalid_outer = FakeBackend(["not addressed"], model=MODEL)
    inner = DraftWorkflow(
        FakeBackend(["inner"], model=MODEL),
    ).generate(scene, "A", "inner")
    half_round = confirm_draft(scene, "A", "inner", confirmation(inner))

    with pytest.raises(DraftGenerationError, match="invalid outer draft"):
        DraftWorkflow(invalid_outer).generate(half_round, "A", "outer")

    assert len(invalid_outer.generate_calls) == 1


def test_workflow_rejects_backend_for_a_different_scene_model() -> None:
    """Registry mistakes cannot route a scene through another model."""
    scene = add_manual_event(create_scene("Mismatch", MODEL), "A", "event")
    workflow = DraftWorkflow(
        FakeBackend(["answer"], model="fake/other"),
    )

    with pytest.raises(SceneConflictError, match="different"):
        workflow.generate(scene, "A", "inner")


def _replace_inner_prompt(scene: Any, prompt: str) -> Any:
    """Return a valid scene with only Agent A's inner prompt changed."""
    agent = scene.agents[0]
    changed = agent.model_copy(
        update={
            "inner_context": agent.inner_context.model_copy(
                update={"system_prompt": prompt}
            )
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
