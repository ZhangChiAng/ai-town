"""Product-boundary tests for isolated inner and outer model context."""

import json

import pytest

from app.draft_workflow import (
    DraftGenerationError,
    DraftWorkflow,
    confirm_draft,
)
from app.models import (
    SceneConflictError,
    add_manual_event,
    create_scene,
)
from tests.helpers import FakeBackend, confirmation

MODEL = "fake/protocol-neutral"


def test_first_inner_and_outer_inputs_are_exact() -> None:
    """The first round uses the two fixed user-text formats verbatim."""
    scene = add_manual_event(
        create_scene("格式", MODEL),
        "A",
        "门外传来两声敲门。",
    )
    backend = FakeBackend(
        ["先别开门。\n问清楚是谁。", "To B: 你在门外吗？"],
        model=MODEL,
    )
    workflow = DraftWorkflow(backend)

    inner = workflow.generate(scene, "A", "inner")
    assert backend.conversations[0].current_input == (
        "外部事件：\n门外传来两声敲门。"
    )
    scene = confirm_draft(scene, "A", "inner", confirmation(inner))

    outer = workflow.generate(scene, "A", "outer")
    assert backend.conversations[1].current_input == (
        "外部事件：\n门外传来两声敲门。\n\n"
        "你内心有一个声音：\n先别开门。\n问清楚是谁。"
    )
    assert outer.event_id == inner.event_id


def test_later_inner_input_includes_only_previous_outer_output_and_event() -> (
    None
):
    """Cross-layer flow is limited to the fixed prior-output prefix."""
    scene = add_manual_event(create_scene("后续", MODEL), "A", "第一件事")
    backend = FakeBackend(["内层一", "To B: 外层一"], model=MODEL)
    workflow = DraftWorkflow(backend)
    inner = workflow.generate(scene, "A", "inner")
    scene = confirm_draft(scene, "A", "inner", confirmation(inner))
    outer = workflow.generate(scene, "A", "outer")
    scene = confirm_draft(scene, "A", "outer", confirmation(outer))
    scene = add_manual_event(scene, "A", "第二件事")

    preview = DraftWorkflow(FakeBackend([], model=MODEL)).preview(
        scene,
        "A",
        "inner",
    )

    assert preview.context[-1].text == (
        "外层人格：\nTo B: 外层一\n\n外部事件：\n第二件事"
    )


def test_two_layers_send_only_their_own_complete_history() -> None:
    """No other layer, Agent, queue, or rollback metadata enters context."""
    scene = create_scene("隔离", MODEL)
    scene = scene.model_copy(
        update={
            "agents": [
                agent.model_copy(
                    update={
                        "inner_context": agent.inner_context.model_copy(
                            update={"system_prompt": f"INNER {agent.id}"}
                        ),
                        "outer_context": agent.outer_context.model_copy(
                            update={"system_prompt": f"OUTER {agent.id}"}
                        ),
                    }
                )
                for agent in scene.agents
            ]
        }
    )
    scene = add_manual_event(scene, "A", "A first event")
    backend = FakeBackend(
        ["A inner secret", "To B: A public text"], model=MODEL
    )
    workflow = DraftWorkflow(backend)
    inner = workflow.generate(scene, "A", "inner")
    scene = confirm_draft(scene, "A", "inner", confirmation(inner))
    outer = workflow.generate(scene, "A", "outer")
    scene = confirm_draft(scene, "A", "outer", confirmation(outer))
    scene = add_manual_event(scene, "A", "A second event")
    scene = add_manual_event(scene, "B", "B private pending event")

    preview = workflow.preview(scene, "A", "inner")
    serialized = json.dumps(
        [{"role": item.role, "text": item.text} for item in preview.context],
        ensure_ascii=False,
    )

    assert preview.context[0].text == "INNER A"
    assert [
        (turn.input, turn.output) for turn in backend.conversations[-1].turns
    ] == [("外部事件：\nA first event", "A inner secret")]
    assert "OUTER A" not in serialized
    assert "INNER B" not in serialized
    assert "B private pending event" not in serialized
    assert str(scene.rollback_stack[0].call_id) not in serialized


def test_each_layer_keeps_every_confirmed_turn_without_truncation() -> None:
    """All confirmed layer turns remain verbatim in their original order."""
    scene = create_scene("完整历史", MODEL)
    outputs: list[str] = []
    for index in range(4):
        outputs.extend((f"INNER-{index}", f"To B: OUTER-{index}"))
    workflow = DraftWorkflow(FakeBackend(outputs, model=MODEL))

    for index in range(4):
        scene = add_manual_event(scene, "A", f"EVENT-{index}")
        inner = workflow.generate(scene, "A", "inner")
        scene = confirm_draft(scene, "A", "inner", confirmation(inner))
        outer = workflow.generate(scene, "A", "outer")
        scene = confirm_draft(scene, "A", "outer", confirmation(outer))

    scene = add_manual_event(scene, "A", "EVENT-next")
    backend = FakeBackend([], model=MODEL)
    DraftWorkflow(backend).preview(scene, "A", "inner")

    assert len(backend.conversations[-1].turns) == 4
    for index, turn in enumerate(backend.conversations[-1].turns):
        assert turn.input.endswith(f"EVENT-{index}")
        assert turn.output == f"INNER-{index}"


def test_reasoning_is_observer_only_and_never_persisted_or_reused() -> None:
    """Temporary reasoning disappears at the confirmation boundary."""
    scene = add_manual_event(create_scene("Reasoning", MODEL), "C", "event")
    backend = FakeBackend(["inner answer"], model=MODEL)
    draft = DraftWorkflow(backend).generate(scene, "C", "inner")
    assert draft.reasoning[0].text == "observer-only fake reasoning"

    scene = confirm_draft(scene, "C", "inner", confirmation(draft))
    next_backend = FakeBackend([], model=MODEL)
    DraftWorkflow(next_backend).preview(scene, "C", "outer")
    serialized_scene = json.dumps(scene.model_dump(mode="json"))
    serialized_context = repr(next_backend.conversations[-1])

    assert "observer-only fake reasoning" not in serialized_scene
    assert "observer-only fake reasoning" not in serialized_context


def test_invalid_outer_generation_fails_after_exactly_one_call() -> None:
    """A malformed visible outer result is rejected without retry."""
    scene = add_manual_event(create_scene("坏外层", MODEL), "A", "event")
    inner_backend = FakeBackend(["inner"], model=MODEL)
    inner = DraftWorkflow(inner_backend).generate(scene, "A", "inner")
    scene = confirm_draft(scene, "A", "inner", confirmation(inner))
    invalid_backend = FakeBackend(["not addressed"], model=MODEL)

    with pytest.raises(DraftGenerationError, match="invalid outer draft"):
        DraftWorkflow(invalid_backend).generate(scene, "A", "outer")

    assert len(invalid_backend.generate_calls) == 1


def test_outer_preview_is_rejected_before_inner_confirmation() -> None:
    """The workflow enforces the manual inner-before-outer phase."""
    scene = add_manual_event(create_scene("阶段", MODEL), "A", "event")

    with pytest.raises(SceneConflictError, match="no confirmed inner"):
        DraftWorkflow(FakeBackend([], model=MODEL)).preview(scene, "A", "outer")
