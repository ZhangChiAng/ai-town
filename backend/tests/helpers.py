"""Shared test doubles and HTTP helpers for the backend test suite."""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from app.main import create_app
from app.model_backends import (
    JsonObject,
    ModelBackend,
    ModelConversation,
    ModelGeneration,
    ModelReasoning,
    ModelUsage,
)
from app.models import ConfirmLayerRequest, create_scene
from app.storage import SceneStorage
from tests.client import TestClient

# Observer-only defaults reused across every fake generation so tests do not
# invent ad-hoc reasoning text that could leak into persisted scene state.
DEFAULT_REASONING = ModelReasoning(
    type="summary_text",
    text="observer-only fake reasoning",
)
DEFAULT_USAGE = ModelUsage(
    input_tokens=10,
    output_tokens=4,
    cache_creation_input_tokens=2,
    cache_read_input_tokens=3,
)


def neutral_payload(
    conversation: ModelConversation,
    model: str,
    *,
    version: int = 1,
) -> JsonObject:
    """Build a deliberately provider-unlike JSON-safe request envelope.

    The shape is intentionally unrelated to any real adapter wire format so
    preview and snapshot assertions stay protocol-neutral.
    """
    return {
        "engine": model,
        "version": version,
        "dialogue": {
            "rulebook": conversation.system_prompt,
            "past": [[turn.input, turn.output] for turn in conversation.turns],
            "next": conversation.current_input,
        },
    }


class FakeBackend:
    """Protocol-neutral backend double with observable call tracking.

    ``outputs`` may hold plain strings (wrapped with the shared observer
    defaults), full ``ModelGeneration`` values, or ``Exception`` instances
    raised on generation. ``payload_builder`` lets a test supply its own
    envelope shape while reusing the lifecycle and tracking plumbing.
    """

    def __init__(
        self,
        outputs: list[str | ModelGeneration | Exception],
        *,
        model: str,
        payload_builder: Callable[[ModelConversation, str], JsonObject]
        | None = None,
        events: list[str] | None = None,
    ) -> None:
        """Bind one exact model, queued results, and an optional close sink."""
        self._model = model
        self._outputs = list(outputs)
        self._payload_builder = payload_builder or neutral_payload
        self._events = events
        self.generate_calls: list[ModelConversation] = []
        self.conversations = self.generate_calls
        self.close_calls = 0
        self._closed = False

    @property
    def model(self) -> str:
        """Return the exact case-sensitive model identity."""
        return self._model

    async def generate(
        self,
        conversation: ModelConversation,
    ) -> ModelGeneration:
        """Consume one queued result as a single upstream-equivalent call."""
        self.generate_calls.append(conversation)
        result = self._outputs.pop(0)
        if isinstance(result, Exception):
            raise result
        if isinstance(result, ModelGeneration):
            return result
        return ModelGeneration(
            content=result,
            reasoning=(DEFAULT_REASONING,),
            usage=DEFAULT_USAGE,
            request_snapshot=self._payload_builder(
                conversation,
                self._model,
            ),
        )

    async def aclose(self) -> None:
        """Record one idempotent release and optionally announce it."""
        if self._closed:
            return
        self._closed = True
        self.close_calls += 1
        if self._events is not None:
            self._events.append(f"close:{self._model}")


def make_client(
    scene_directory: Path,
    backends: Mapping[str, ModelBackend] | ModelBackend | None = None,
    *,
    texts: list[str | ModelGeneration | Exception] | None = None,
    model: str | None = None,
) -> TestClient:
    """Create a ``TestClient`` over isolated storage and one fake backend.

    Pass ``backends`` to inject a pre-built backend or mapping. Pass
    ``texts`` (with ``model``) to spin up a default ``FakeBackend``.
    """
    if backends is None:
        if model is None:
            raise TypeError("model is required when backends is None")
        backends = FakeBackend(texts or [], model=model)
    return TestClient(create_app(SceneStorage(scene_directory), backends))


def require_json(
    response: Any,
    status_code: int = 200,
) -> dict[str, Any]:
    """Assert one response status and return its JSON object."""
    assert response.status_code == status_code, response.text
    body = response.json()
    assert isinstance(body, dict)
    return body


def post_scene(
    client: TestClient,
    *,
    model: str,
    name: str = "HTTP integration",
    expected_status: int = 201,
) -> dict[str, Any]:
    """Create one bound scene through the public API."""
    return require_json(
        client.post("/api/scenes", json={"name": name, "model": model}),
        expected_status,
    )


def fill_prompts(
    client: TestClient,
    scene: dict[str, Any],
    *,
    text: str | Callable[[str, str], str] | None = None,
) -> dict[str, Any]:
    """Write complete inner/outer prompts for every Agent via the PUT API.

    ``text`` is either one literal string or a factory receiving the Agent
    ID and layer; the default factory keeps every layer observably distinct.
    """
    factory = (
        (text if isinstance(text, Callable) else (lambda agent_id, layer: text))
        if text is not None
        else (lambda agent_id, layer: f"{layer.upper()} {agent_id}")
    )
    update = {
        "name": scene["name"],
        "agents": [
            {
                "id": agent["id"],
                "name": agent["name"],
                "inner_context": {
                    "system_prompt": factory(agent["id"], "inner")
                },
                "outer_context": {
                    "system_prompt": factory(agent["id"], "outer")
                },
            }
            for agent in scene["agents"]
        ],
    }
    return require_json(client.put(f"/api/scenes/{scene['id']}", json=update))


def post_prompted_scene(
    client: TestClient,
    *,
    model: str,
    name: str = "HTTP integration",
    text: str | Callable[[str, str], str] | None = None,
) -> dict[str, Any]:
    """Create one bound scene whose Agents already hold complete prompts."""
    return fill_prompts(
        client,
        post_scene(client, model=model, name=name),
        text=text,
    )


def create_prompted_scene(name: str, model: str) -> Any:
    """Build an unpersisted scene with complete prompts for every Agent."""
    return _replace_all_prompts(create_scene(name, model))


def _replace_all_prompts(scene: Any) -> Any:
    """Return a scene with distinct INNER/OUTER prompts on every Agent."""
    return scene.model_copy(
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


def post_event(
    client: TestClient,
    scene_id: str,
    agent_id: str,
    content: str,
    *,
    expected_status: int = 201,
) -> dict[str, Any]:
    """Queue one manual event through the public API."""
    return require_json(
        client.post(
            f"/api/scenes/{scene_id}/agents/{agent_id}/events",
            json={"content": content},
        ),
        expected_status,
    )


def generate_draft(
    client: TestClient,
    scene_id: str,
    agent_id: str,
    layer: str,
    *,
    expected_status: int = 200,
) -> dict[str, Any]:
    """Generate one browser-held draft through the public API."""
    return require_json(
        client.post(f"/api/scenes/{scene_id}/agents/{agent_id}/{layer}-drafts"),
        expected_status,
    )


def confirm_draft(
    client: TestClient,
    scene_id: str,
    agent_id: str,
    layer: str,
    draft: dict[str, Any],
    *,
    content: str | None = None,
    reasoning: list[dict[str, str]] | None = None,
) -> Any:
    """Submit one browser draft; returns the raw response for status checks."""
    body = {
        "call_id": draft["call_id"],
        "event_ids": draft["event_ids"],
        "content": draft["content"] if content is None else content,
        "state_token": draft["state_token"],
    }
    if reasoning is not None:
        body["reasoning"] = reasoning
    return client.post(
        f"/api/scenes/{scene_id}/agents/{agent_id}/{layer}-confirmations",
        json=body,
    )


def confirmation(
    draft: Any,
    content: str | None = None,
) -> ConfirmLayerRequest:
    """Convert one workflow draft object to the confirmation DTO."""
    return ConfirmLayerRequest(
        call_id=draft.call_id,
        event_ids=list(draft.event_ids),
        content=draft.content if content is None else content,
        state_token=draft.state_token,
    )
