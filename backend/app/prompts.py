"""Fixed Jinja system-prompt templates and their scene assembly."""

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.models import AGENT_IDS, AgentId, Layer, Scene, get_agent

_TEMPLATE_DIRECTORY = Path(__file__).with_name("prompt_templates")
_JINJA_ENVIRONMENT = Environment(
    loader=FileSystemLoader(_TEMPLATE_DIRECTORY),
    autoescape=False,
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


def build_system_prompt(
    scene: Scene,
    agent_id: AgentId,
    layer: Layer,
) -> str:
    """Build one layer's prompt from the backend-owned Jinja templates."""
    agent = get_agent(scene, agent_id)
    profile = agent.prompt_profile
    if layer == "inner":
        return _render_template(
            "inner_system_prompt.j2",
            name=agent.name,
            pronoun=profile.pronoun,
            hidden_beliefs=profile.hidden_beliefs,
            inner_memories=profile.inner_memories,
        )

    interactive_characters: list[dict[str, Any]] = []
    for target_id in AGENT_IDS:
        interaction = agent.interactions.get(target_id)
        if interaction is None or not interaction.addresses:
            continue
        target = get_agent(scene, target_id)
        # Internal IDs establish stable order but never enter template data.
        interactive_characters.append(
            {
                "name": target.name,
                "description": interaction.description,
                "addresses": [
                    {"address": address, "occasion": occasion}
                    for address, occasion in interaction.addresses.items()
                ],
            }
        )
    return _render_template(
        "outer_system_prompt.j2",
        name=agent.name,
        outer_memories=profile.outer_memories,
        interactive_characters=interactive_characters,
    )


def _render_template(template_name: str, **context: Any) -> str:
    """Render a fixed template exactly once with strict variables."""
    return _JINJA_ENVIRONMENT.get_template(template_name).render(**context)
