"""Domain models and request/response schemas for scenes and agents."""

from typing import Literal, Self
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

SCHEMA_VERSION = 1
AGENT_IDS = ("A", "B", "C")

AgentId = Literal["A", "B", "C"]


class ApiModel(BaseModel):
    """Base model that rejects unknown fields in requests and responses."""

    model_config = ConfigDict(extra="forbid")


def _strip_non_blank_name(value: str) -> str:
    """Strip whitespace and require a non-empty string."""
    stripped = value.strip()
    if not stripped:
        raise ValueError("name must not be blank")
    return stripped


class Agent(ApiModel):
    """An agent in a scene with identity, personality traits, and a timeline."""

    id: AgentId
    name: str
    persona: str = ""
    desire: str = ""
    fear: str = ""
    memory: str = ""
    timeline: list[object] = Field(default_factory=list, max_length=0)

    _validate_name = field_validator("name")(_strip_non_blank_name)


class AgentUpdate(ApiModel):
    """Writable agent fields exposed through the scene-update endpoint."""

    id: AgentId
    name: str
    persona: str
    desire: str
    fear: str
    memory: str

    _validate_name = field_validator("name")(_strip_non_blank_name)


class Scene(ApiModel):
    """A named scene containing exactly three agents (A, B, C)."""

    schema_version: Literal[1] = SCHEMA_VERSION
    id: UUID
    name: str
    agents: list[Agent] = Field(min_length=3, max_length=3)

    _validate_name = field_validator("name")(_strip_non_blank_name)

    @model_validator(mode="after")
    def validate_agent_ids(self) -> Self:
        """Ensure agent IDs are A, B, C in that exact order."""
        agent_ids = tuple(agent.id for agent in self.agents)
        if agent_ids != AGENT_IDS:
            raise ValueError("agents must contain A, B, and C in that order")
        return self


class SceneSummary(ApiModel):
    """Lightweight scene reference for list responses."""

    id: UUID
    name: str


class CreateSceneRequest(ApiModel):
    """Payload for creating a new scene."""

    name: str

    _validate_name = field_validator("name")(_strip_non_blank_name)


class UpdateSceneRequest(ApiModel):
    """Payload for updating an existing scene's editable fields."""

    name: str
    agents: list[AgentUpdate] = Field(min_length=3, max_length=3)

    _validate_name = field_validator("name")(_strip_non_blank_name)

    @model_validator(mode="after")
    def validate_agent_ids(self) -> Self:
        """Ensure agent IDs are A, B, C in that exact order."""
        agent_ids = tuple(agent.id for agent in self.agents)
        if agent_ids != AGENT_IDS:
            raise ValueError("agents must contain A, B, and C in that order")
        return self


def create_scene(name: str) -> Scene:
    """Create a new scene with default agents.

    Args:
        name: The display name for the scene.

    Returns:
        A Scene with a new UUID and one default agent per agent ID.
    """
    return Scene(
        id=uuid4(),
        name=name,
        agents=[Agent(id=agent_id, name=agent_id) for agent_id in AGENT_IDS],
    )


def update_scene(scene: Scene, update: UpdateSceneRequest) -> Scene:
    """Apply a scene update, preserving the existing timeline.

    Only the editable fields (name, agent names, persona, desire, fear,
    memory) are taken from the update payload. Schema version, scene ID,
    and agent timelines are carried forward from the original scene.

    Args:
        scene: The existing scene to update.
        update: The update payload with new values.

    Returns:
        A new Scene instance reflecting the merged state.
    """
    return Scene(
        schema_version=scene.schema_version,
        id=scene.id,
        name=update.name,
        agents=[
            Agent(
                id=updated_agent.id,
                name=updated_agent.name,
                persona=updated_agent.persona,
                desire=updated_agent.desire,
                fear=updated_agent.fear,
                memory=updated_agent.memory,
                timeline=existing_agent.timeline,
            )
            for existing_agent, updated_agent in zip(
                scene.agents, update.agents, strict=True
            )
        ],
    )
