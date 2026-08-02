"""Product-boundary tests for isolated inner and outer model context."""

import asyncio
import json

import pytest

from app.draft_workflow import (
    DraftGenerationError,
    DraftWorkflow,
    confirm_draft,
)
from app.models import (
    AgentId,
    Layer,
    LayerDraftResponse,
    Scene,
    SceneConflictError,
    UpdateSceneRequest,
    add_manual_event,
    create_scene,
    update_scene,
)
from tests.helpers import FakeBackend, confirmation

MODEL = "fake/protocol-neutral"


def _generate(
    workflow: DraftWorkflow,
    scene: Scene,
    agent_id: AgentId,
    layer: Layer,
) -> LayerDraftResponse:
    """Run one async workflow generation without an async pytest plugin."""
    return asyncio.run(workflow.generate(scene, agent_id, layer))


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

    inner = _generate(workflow, scene, "A", "inner")
    assert backend.conversations[0].current_input == (
        "外部事件：\n门外传来两声敲门。"
    )
    scene = confirm_draft(scene, "A", "inner", confirmation(inner))

    outer = _generate(workflow, scene, "A", "outer")
    assert backend.conversations[1].current_input == (
        "外部事件：\n门外传来两声敲门。\n\n"
        "你内心有一个声音：\n先别开门。\n问清楚是谁。"
    )
    assert outer.event_ids == inner.event_ids


@pytest.mark.parametrize("recipient_id", ["B", "C"])
def test_later_inner_input_names_recipient_and_uses_only_message_body(
    recipient_id: str,
) -> None:
    """Later inner input renders routed speech without its address prefix."""
    scene = add_manual_event(create_scene("后续", MODEL), "A", "第一件事")
    backend = FakeBackend(
        ["内层一", f"To {recipient_id}: 外层一"],
        model=MODEL,
    )
    workflow = DraftWorkflow(backend)
    inner = _generate(workflow, scene, "A", "inner")
    scene = confirm_draft(scene, "A", "inner", confirmation(inner))
    outer = _generate(workflow, scene, "A", "outer")
    scene = confirm_draft(scene, "A", "outer", confirmation(outer))
    recipient = next(
        agent for agent in scene.agents if agent.id == recipient_id
    )
    assert scene.agents[0].outer_context.turns[-1].output == (
        f"To {recipient_id}: 外层一"
    )
    assert recipient.pending_events[-1].content == "From A: 外层一"
    scene = add_manual_event(scene, "A", "第二件事")

    preview = DraftWorkflow(FakeBackend([], model=MODEL)).preview(
        scene,
        "A",
        "inner",
    )

    assert preview.context[-1].text == (
        f"外层人格上一轮对 Agent {recipient_id}（{recipient_id}）说：\n"
        "外层一\n\n外部事件：\n第二件事"
    )


def test_later_inner_input_uses_recipient_current_name() -> None:
    """The speech direction reflects a recipient rename after the outer turn."""
    scene = add_manual_event(create_scene("改名", MODEL), "A", "第一件事")
    workflow = DraftWorkflow(
        FakeBackend(["内层一", "To B: 外层一"], model=MODEL)
    )
    inner = _generate(workflow, scene, "A", "inner")
    scene = confirm_draft(scene, "A", "inner", confirmation(inner))
    outer = _generate(workflow, scene, "A", "outer")
    scene = confirm_draft(scene, "A", "outer", confirmation(outer))
    update = UpdateSceneRequest.model_validate(
        {
            "name": scene.name,
            "agents": [
                {
                    "id": agent.id,
                    "name": "儿子" if agent.id == "B" else agent.name,
                    "inner_context": {
                        "system_prompt": agent.inner_context.system_prompt
                    },
                    "outer_context": {
                        "system_prompt": agent.outer_context.system_prompt
                    },
                }
                for agent in scene.agents
            ],
        }
    )
    scene = add_manual_event(update_scene(scene, update), "A", "第二件事")

    preview = DraftWorkflow(FakeBackend([], model=MODEL)).preview(
        scene,
        "A",
        "inner",
    )

    assert preview.context[-1].text == (
        "外层人格上一轮对 Agent B（儿子）说：\n外层一\n\n外部事件：\n第二件事"
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
    inner = _generate(workflow, scene, "A", "inner")
    scene = confirm_draft(scene, "A", "inner", confirmation(inner))
    outer = _generate(workflow, scene, "A", "outer")
    scene = confirm_draft(scene, "A", "outer", confirmation(outer))
    scene = add_manual_event(scene, "A", "A second event")
    scene = add_manual_event(scene, "B", "B private pending event")

    preview = workflow.preview(scene, "A", "inner")
    serialized = json.dumps(
        [{"role": item.role, "text": item.text} for item in preview.context],
        ensure_ascii=False,
    )

    assert preview.context[0].text == "INNER A"
    assert [(item.role, item.text) for item in preview.context[1:3]] == [
        ("user", "外部事件：\nA first event"),
        ("assistant", "A inner secret"),
    ]
    assert len(backend.generate_calls) == 2
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
        inner = _generate(workflow, scene, "A", "inner")
        scene = confirm_draft(scene, "A", "inner", confirmation(inner))
        outer = _generate(workflow, scene, "A", "outer")
        scene = confirm_draft(scene, "A", "outer", confirmation(outer))

    scene = add_manual_event(scene, "A", "EVENT-next")
    backend = FakeBackend([], model=MODEL)
    preview = DraftWorkflow(backend).preview(scene, "A", "inner")

    assert backend.generate_calls == []
    assert len(preview.context) == 10
    for index in range(4):
        user_item = preview.context[1 + index * 2]
        assistant_item = preview.context[2 + index * 2]
        assert user_item.role == "user"
        assert user_item.text.endswith(f"EVENT-{index}")
        assert assistant_item.role == "assistant"
        assert assistant_item.text == f"INNER-{index}"


def test_reasoning_is_observer_only_and_never_persisted_or_reused() -> None:
    """Temporary reasoning disappears at the confirmation boundary."""
    scene = add_manual_event(create_scene("Reasoning", MODEL), "C", "event")
    backend = FakeBackend(["inner answer"], model=MODEL)
    draft = _generate(DraftWorkflow(backend), scene, "C", "inner")
    assert draft.reasoning[0].text == "observer-only fake reasoning"

    scene = confirm_draft(scene, "C", "inner", confirmation(draft))
    next_backend = FakeBackend([], model=MODEL)
    serialized_scene = json.dumps(scene.model_dump(mode="json"))
    serialized_context = repr(
        DraftWorkflow(next_backend).preview(scene, "C", "outer").context
    )

    assert "observer-only fake reasoning" not in serialized_scene
    assert "observer-only fake reasoning" not in serialized_context
    assert next_backend.generate_calls == []


def test_invalid_outer_generation_fails_after_exactly_one_call() -> None:
    """A malformed visible outer result is rejected without retry."""
    scene = add_manual_event(create_scene("坏外层", MODEL), "A", "event")
    inner_backend = FakeBackend(["inner"], model=MODEL)
    inner = _generate(DraftWorkflow(inner_backend), scene, "A", "inner")
    scene = confirm_draft(scene, "A", "inner", confirmation(inner))
    invalid_backend = FakeBackend(["not addressed"], model=MODEL)

    with pytest.raises(DraftGenerationError, match="invalid outer draft"):
        _generate(DraftWorkflow(invalid_backend), scene, "A", "outer")

    assert len(invalid_backend.generate_calls) == 1


def test_outer_preview_is_rejected_before_inner_confirmation() -> None:
    """The workflow enforces the manual inner-before-outer phase."""
    scene = add_manual_event(create_scene("阶段", MODEL), "A", "event")

    with pytest.raises(SceneConflictError, match="no confirmed inner"):
        DraftWorkflow(FakeBackend([], model=MODEL)).preview(scene, "A", "outer")
