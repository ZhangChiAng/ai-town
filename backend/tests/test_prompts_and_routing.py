"""Snapshot and domain tests for prompt assembly and semantic routing."""

# Exact product-owned prompt snapshots intentionally exceed the limit.
# ruff: noqa: E501

import asyncio

import pytest
from jinja2 import UndefinedError
from pydantic import ValidationError

from app.draft_workflow import (
    DraftGenerationError,
    DraftWorkflow,
    confirm_draft,
)
from app.models import (
    Agent,
    InnerContext,
    Interaction,
    OuterContext,
    PromptProfile,
    SceneConflictError,
    add_manual_event,
    get_agent,
    rollback_latest_call,
)
from app.prompts import _render_template, build_system_prompt
from tests.helpers import FakeBackend, confirmation, create_prompted_scene

MODEL = "fake/model"


def _generate(
    workflow: DraftWorkflow,
    scene: object,
    agent_id: str,
    layer: str,
) -> object:
    """Run one asynchronous draft generation in a synchronous test."""
    return asyncio.run(workflow.generate(scene, agent_id, layer))


def test_lin_xiumei_system_prompts_match_fixed_templates_exactly() -> None:
    """The outer prompt shows names and relationship details, never IDs."""
    scene = create_prompted_scene("母子", MODEL)
    mother = scene.agents[0].model_copy(
        update={
            "name": "林秀梅",
            "prompt_profile": PromptProfile(
                pronoun="她",
                hidden_beliefs="血缘关系是爱的纽带。",
                inner_memories="儿子小时候很听话。",
                outer_memories="她在农村长大。",
            ),
            "interactions": {
                "B": Interaction(
                    description="你的儿子，最近工作不顺。",
                    addresses={
                        "儿子": "一般场合",
                        "国栋": "对他感到生气时",
                    },
                )
            },
        }
    )
    son = scene.agents[1].model_copy(update={"name": "李国栋"})
    scene = scene.model_copy(update={"agents": [mother, son, scene.agents[2]]})

    assert (
        build_system_prompt(scene, "A", "inner")
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
        build_system_prompt(scene, "A", "outer")
        == """你是“林秀梅”。

【你记得】
她在农村长大。你的记忆中明确写出的内容，才是你现在能想起来的；没有对应内容就是想不起来，禁止编造。

【内心的声音】
你经常听到来自内心的声音与你直接对话，它说“我们”或“咱们”时，指的就是你和它，毕竟你们属于同一个人。它是在帮你理解现状和判断形势。

【当下可互动角色】
以下每组信息分别说明一位可互动人物的姓名、你对其的简单认识，以及你可以使用的全部称呼和对应场合。

人物姓名：李国栋
人物简介：
你的儿子，最近工作不顺。

可用称呼及对应场合：
- 称呼：儿子
  使用场合：一般场合
- 称呼：国栋
  使用场合：对他感到生气时

【你怎么说话】
像现实世界中的人一样说话，考虑场合。从上方列出的可用称呼中选择符合当前场合的称呼，然后输出`对{称呼}说：{正文}`。

【不该说话的时候不说话是一种美德】
当你发现错失了说话的良机，或者你觉得没有必要继续说话的时候，输出`STOP`。"""
    )


def test_interactions_render_by_agent_id_then_entry_order() -> None:
    """Target order is fixed even when the JSON mapping arrived out of order."""
    scene = create_prompted_scene("order", MODEL)
    agent = scene.agents[0].model_copy(
        update={
            "interactions": {
                "C": Interaction(
                    description="同事关系。",
                    addresses={"同事": "工作场合"},
                ),
                "B": Interaction(
                    description="母子关系。",
                    addresses={"小名": "亲昵场合", "儿子": "一般场合"},
                ),
            }
        }
    )
    scene = scene.model_copy(update={"agents": [agent, *scene.agents[1:]]})

    prompt = build_system_prompt(scene, "A", "outer")

    assert (
        "人物姓名：B\n人物简介：\n母子关系。\n\n"
        "可用称呼及对应场合：\n- 称呼：小名\n  使用场合：亲昵场合\n"
        "- 称呼：儿子\n  使用场合：一般场合\n\n"
        "人物姓名：C\n人物简介：\n同事关系。" in prompt
    )


def test_jinja_is_strict_and_never_reexecutes_user_text() -> None:
    """Missing owned variables fail while user-authored braces stay literal."""
    with pytest.raises(UndefinedError):
        _render_template("inner_system_prompt.j2", name="缺少其他变量")

    scene = create_prompted_scene("jinja", MODEL)
    agent = scene.agents[0].model_copy(
        update={
            "prompt_profile": scene.agents[0].prompt_profile.model_copy(
                update={"outer_memories": "记得 {{ untouched }}。"}
            ),
            "interactions": {
                "B": Interaction(
                    description="你的 {{ relationship }}。",
                    addresses={"{{ nickname }}": "{{ occasion }}"},
                )
            },
        }
    )
    scene = scene.model_copy(update={"agents": [agent, *scene.agents[1:]]})

    prompt = build_system_prompt(scene, "A", "outer")

    assert "记得 {{ untouched }}。" in prompt
    assert "你的 {{ relationship }}。" in prompt
    assert "称呼：{{ nickname }}" in prompt
    assert "使用场合：{{ occasion }}" in prompt


@pytest.mark.parametrize(
    "interactions",
    [
        {
            "A": {
                "description": "自己。",
                "addresses": {"自己": "任何场合"},
            }
        },
        {
            "B": {
                "description": "家人。",
                "addresses": {"家人": "一般场合"},
            },
            "C": {
                "description": "也是家人。",
                "addresses": {"家人": "正式场合"},
            },
        },
    ],
)
def test_interactions_reject_self_targets_and_duplicate_addresses(
    interactions: dict[str, object],
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
            inner_context=InnerContext(turns=[]),
            outer_context=OuterContext(turns=[]),
            pending_events=[],
        )


def test_only_configured_agent_can_generate_and_route_by_multiple_labels() -> (
    None
):
    """B/C may stay blank while A routes two labels to the same target."""
    scene = create_prompted_scene("single", MODEL)
    agent_a = scene.agents[0].model_copy(
        update={
            "interactions": {
                "B": Interaction(
                    description="你的儿子。",
                    addresses={
                        "儿子": "一般场合",
                        "小名": "亲昵场合",
                    },
                )
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


def test_blank_relationship_description_blocks_every_model_entry() -> None:
    """Blank descriptions save, but preview, generation, and confirmation stop."""
    scene = add_manual_event(
        create_prompted_scene("简介未完成", MODEL),
        "A",
        "事件",
    )
    valid_workflow = DraftWorkflow(FakeBackend(["原草稿"], model=MODEL))
    draft = _generate(valid_workflow, scene, "A", "inner")
    agent = scene.agents[0].model_copy(
        update={
            "interactions": {
                "B": Interaction(
                    description="",
                    addresses={"儿子": "一般场合"},
                )
            }
        }
    )
    incomplete = scene.model_copy(update={"agents": [agent, *scene.agents[1:]]})
    blocked_backend = FakeBackend(["不应调用"], model=MODEL)
    blocked_workflow = DraftWorkflow(blocked_backend)

    with pytest.raises(SceneConflictError, match="description"):
        blocked_workflow.preview(incomplete, "A", "inner")
    with pytest.raises(SceneConflictError, match="description"):
        _generate(blocked_workflow, incomplete, "A", "inner")
    with pytest.raises(SceneConflictError, match="description"):
        confirm_draft(incomplete, "A", "inner", confirmation(draft))
    assert blocked_backend.generate_calls == []

    repaired_agent = agent.model_copy(
        update={
            "interactions": {
                "B": Interaction(
                    description="你的儿子。",
                    addresses={"儿子": "一般场合"},
                )
            }
        }
    )
    repaired = incomplete.model_copy(
        update={"agents": [repaired_agent, *incomplete.agents[1:]]}
    )
    assert blocked_workflow.preview(repaired, "A", "inner").context


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
def test_outer_generation_rejects_every_invalid_output(
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
