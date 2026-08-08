"""Snapshot and domain tests for v9 prompt assembly and semantic routing."""

# Exact product-owned template snapshot lines intentionally exceed the limit.
# ruff: noqa: E501

import asyncio

import pytest
from pydantic import ValidationError

from app.draft_workflow import (
    DraftGenerationError,
    DraftWorkflow,
    confirm_draft,
)
from app.models import (
    Agent,
    InnerContext,
    OuterContext,
    PromptProfile,
    SceneConflictError,
    add_manual_event,
    get_agent,
    rollback_latest_call,
)
from app.prompts import build_system_prompt
from tests.helpers import FakeBackend, confirmation, create_prompted_scene

MODEL = "fake/v9"


def _generate(
    workflow: DraftWorkflow,
    scene: object,
    agent_id: str,
    layer: str,
) -> object:
    """Run one asynchronous draft generation in a synchronous test."""
    return asyncio.run(workflow.generate(scene, agent_id, layer))


def test_lin_xiumei_system_prompts_match_fixed_templates_exactly() -> None:
    """Only the five profile values and interaction block are substituted."""
    agent = Agent(
        id="A",
        name="林秀梅",
        prompt_profile=PromptProfile(
            pronoun="她",
            hidden_beliefs="血缘关系是爱的纽带。",
            inner_memories="儿子小时候很听话。",
            outer_memories="她在农村长大。",
        ),
        interactions={
            "B": {
                "儿子": "一般场合",
                "小名": "亲昵场合",
            }
        },
        inner_context=InnerContext(),
        outer_context=OuterContext(),
    )

    assert (
        build_system_prompt(agent, "inner")
        == """你是“林秀梅”——不，更准确地说，你是她的潜意识，是她自己几乎察觉不到的、却无时无刻不在问“这事对我是好是坏、赖不赖我自己”的部分。

【你知道但绝对不能说出口的东西】
血缘关系是爱的纽带。

【你依稀记得】
儿子小时候很听话。你的记忆中明确写出的内容，才是你现在能想起来的；没有对应内容就是想不起来，禁止编造。

【你唯一要做的事】
思考“这事对我是好是坏、赖不赖我自己”，然后对自己的外层人格说话。

【你如何说话】
你的话始终是说给外层人格听的，其中的“你”只能指外层人格，严禁直接对其他人说话；当外层人格不按你说的做时，你会划清界限般说“你”和“我”；其它大部分时候你应该说“我们”或“咱们”来拉拢外层人格。使用普通人脑中会冒出的简单口语，应该具体、直接。你得不到满足时，应该更情绪化。每轮只输出一句短话，说完即止。

【不该说话的时候不说话是一种美德】
如果你没什么新的内容讲，就老老实实说`唉`。"""
    )
    assert (
        build_system_prompt(agent, "outer")
        == """你是“林秀梅”。

【你记得】
她在农村长大。你的记忆中明确写出的内容，才是你现在能想起来的；没有对应内容就是想不起来，禁止编造。

【内心的声音】
你经常听到来自内心的声音与你直接对话，它说“我们”或“咱们”时，指的就是你和它，毕竟你们属于同一个人。它是在帮你理解现状和判断形势。

【当下可互动角色】
B：
- 儿子：一般场合
- 小名：亲昵场合

【你怎么说话】
像现实世界中的人一样说话，考虑场合。输出`对{称呼}说：{正文}`。

【不该说话的时候不说话是一种美德】
当你发现错失了说话的良机，或者你觉得没有必要继续说话的时候，输出`STOP`。"""
    )


def test_interactions_render_by_agent_id_then_entry_order() -> None:
    """Target order is fixed even when the JSON mapping arrived out of order."""
    scene = create_prompted_scene("order", MODEL)
    agent = scene.agents[0].model_copy(
        update={
            "interactions": {
                "C": {"同事": "工作场合"},
                "B": {"小名": "亲昵场合", "儿子": "一般场合"},
            }
        }
    )

    prompt = build_system_prompt(agent, "outer")

    assert (
        "B：\n- 小名：亲昵场合\n- 儿子：一般场合\nC：\n- 同事：工作场合"
        in prompt
    )


@pytest.mark.parametrize(
    "interactions",
    [
        {"A": {"自己": "任何场合"}},
        {
            "B": {"家人": "一般场合"},
            "C": {"家人": "正式场合"},
        },
    ],
)
def test_interactions_reject_self_targets_and_duplicate_addresses(
    interactions: dict[str, dict[str, str]],
) -> None:
    """A sender cannot target itself or make reverse lookup ambiguous."""
    with pytest.raises(ValidationError):
        Agent(
            id="A",
            name="A",
            prompt_profile=PromptProfile(
                pronoun="",
                hidden_beliefs="",
                inner_memories="",
                outer_memories="",
            ),
            interactions=interactions,
            inner_context=InnerContext(),
            outer_context=OuterContext(),
        )


def test_only_configured_agent_can_generate_and_route_by_multiple_labels() -> (
    None
):
    """B/C may stay blank while A routes two labels to the same target."""
    scene = create_prompted_scene("single", MODEL)
    agent_a = scene.agents[0].model_copy(
        update={
            "interactions": {
                "B": {
                    "儿子": "一般场合",
                    "小名": "亲昵场合",
                }
            }
        }
    )
    blank_agents = [
        agent.model_copy(
            update={
                "prompt_profile": PromptProfile(
                    pronoun="",
                    hidden_beliefs="",
                    inner_memories="",
                    outer_memories="",
                ),
                "interactions": {},
            }
        )
        for agent in scene.agents[1:]
    ]
    scene = scene.model_copy(update={"agents": [agent_a, *blank_agents]})
    scene = add_manual_event(scene, "A", "手工回复原样：儿子对你说：回来吧")
    workflow = DraftWorkflow(
        FakeBackend(["内层判断", "对小名说：  回来吃饭。  "], model=MODEL)
    )

    inner = _generate(workflow, scene, "A", "inner")
    scene = confirm_draft(scene, "A", "inner", confirmation(inner))
    outer = _generate(workflow, scene, "A", "outer")
    scene = confirm_draft(scene, "A", "outer", confirmation(outer))

    turn = get_agent(scene, "A").outer_context.turns[-1]
    assert turn.output == "对小名说：回来吃饭。"
    assert turn.recipient_id == "B"
    assert get_agent(scene, "B").pending_events[-1].content == "回来吃饭。"
    with pytest.raises(SceneConflictError, match="four prompt variables"):
        DraftWorkflow(FakeBackend([], model=MODEL)).preview(scene, "B", "inner")


def test_stop_has_no_route_builds_no_speech_history_and_rolls_back() -> None:
    """STOP persists only an outer turn and rollback touches no other queue."""
    scene = add_manual_event(create_prompted_scene("stop", MODEL), "A", "事件")
    workflow = DraftWorkflow(FakeBackend(["内层判断", "STOP"], model=MODEL))
    inner = _generate(workflow, scene, "A", "inner")
    scene = confirm_draft(scene, "A", "inner", confirmation(inner))
    outer = _generate(workflow, scene, "A", "outer")
    scene = confirm_draft(scene, "A", "outer", confirmation(outer))

    turn = get_agent(scene, "A").outer_context.turns[-1]
    assert turn.output == "STOP"
    assert turn.recipient_id is None
    assert turn.generated_event_id is None
    assert all(not agent.pending_events for agent in scene.agents)

    next_scene = add_manual_event(scene, "A", "下一事件")
    preview = DraftWorkflow(FakeBackend([], model=MODEL)).preview(
        next_scene, "A", "inner"
    )
    assert preview.context[-1].text == (
        "上一轮：你没有说话。\n\n外部事件：\n下一事件"
    )

    rolled_back = rollback_latest_call(scene)
    assert len(get_agent(rolled_back, "A").inner_context.turns) == 1
    assert get_agent(rolled_back, "A").outer_context.turns == []
    assert all(not agent.pending_events for agent in rolled_back.agents)


@pytest.mark.parametrize(
    "invalid_output",
    [
        " STOP ",
        "To B: 旧格式",
        "From B: 旧格式",
        "对未知称呼说：正文",
        "对A说：发给自己",
        "对B说：   ",
    ],
)
def test_outer_generation_rejects_every_non_v9_output(
    invalid_output: str,
) -> None:
    """Only configured semantic speech or exact STOP can become a draft."""
    scene = add_manual_event(
        create_prompted_scene("invalid", MODEL), "A", "事件"
    )
    inner_workflow = DraftWorkflow(FakeBackend(["内层判断"], model=MODEL))
    inner = _generate(inner_workflow, scene, "A", "inner")
    scene = confirm_draft(scene, "A", "inner", confirmation(inner))

    with pytest.raises(DraftGenerationError, match="invalid outer draft"):
        _generate(
            DraftWorkflow(FakeBackend([invalid_output], model=MODEL)),
            scene,
            "A",
            "outer",
        )
