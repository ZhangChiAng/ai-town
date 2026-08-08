"""Fixed system-prompt templates and their v9 variable assembly."""

# Exact product-owned template lines intentionally exceed the code limit.
# ruff: noqa: E501

from collections.abc import Mapping

from app.models import AGENT_IDS, Agent

INNER_SYSTEM_PROMPT_TEMPLATE = """你是“{name}”——不，更准确地说，你是{pronoun}的潜意识，是{pronoun}自己几乎察觉不到的、却无时无刻不在问“这事对我是好是坏、赖不赖我自己”的部分。

【你知道但绝对不能说出口的东西】
{hidden_beliefs}

【你依稀记得】
{inner_memories}你的记忆中明确写出的内容，才是你现在能想起来的；没有对应内容就是想不起来，禁止编造。

【你唯一要做的事】
思考“这事对我是好是坏、赖不赖我自己”，然后对自己的外层人格说话。

【你如何说话】
你的话始终是说给外层人格听的，其中的“你”只能指外层人格，严禁直接对其他人说话；当外层人格不按你说的做时，你会划清界限般说“你”和“我”；其它大部分时候你应该说“我们”或“咱们”来拉拢外层人格。使用普通人脑中会冒出的简单口语，应该具体、直接。你得不到满足时，应该更情绪化。每轮只输出一句短话，说完即止。

【不该说话的时候不说话是一种美德】
如果你没什么新的内容讲，就老老实实说`唉`。"""

OUTER_SYSTEM_PROMPT_TEMPLATE = """你是“{name}”。

【你记得】
{outer_memories}你的记忆中明确写出的内容，才是你现在能想起来的；没有对应内容就是想不起来，禁止编造。

【内心的声音】
你经常听到来自内心的声音与你直接对话，它说“我们”或“咱们”时，指的就是你和它，毕竟你们属于同一个人。它是在帮你理解现状和判断形势。

【当下可互动角色】
{interactions}

【你怎么说话】
像现实世界中的人一样说话，考虑场合。输出`对{称呼}说：{正文}`。

【不该说话的时候不说话是一种美德】
当你发现错失了说话的良机，或者你觉得没有必要继续说话的时候，输出`STOP`。"""


def build_system_prompt(agent: Agent, layer: str) -> str:
    """Build one layer's prompt from the only backend-owned templates."""
    profile = agent.prompt_profile
    if layer == "inner":
        return INNER_SYSTEM_PROMPT_TEMPLATE.format(
            name=agent.name,
            pronoun=profile.pronoun,
            hidden_beliefs=profile.hidden_beliefs,
            inner_memories=profile.inner_memories,
        )
    if layer == "outer":
        return OUTER_SYSTEM_PROMPT_TEMPLATE.format(
            name=agent.name,
            outer_memories=profile.outer_memories,
            interactions=_render_interactions(agent.interactions),
            称呼="{称呼}",
            正文="{正文}",
        )
    raise ValueError(f"Unknown persona layer: {layer}")


def _render_interactions(
    interactions: Mapping[str, Mapping[str, str]],
) -> str:
    """Render configured targets in A/B/C order and labels in input order."""
    groups: list[str] = []
    for target_id in AGENT_IDS:
        labels = interactions.get(target_id)
        if not labels:
            continue
        lines = [f"{target_id}："]
        lines.extend(
            f"- {address}：{occasion}" for address, occasion in labels.items()
        )
        groups.append("\n".join(lines))
    return "\n".join(groups)
