"""HTTP integration tests for protocol-neutral model backend workflows."""

import json
from pathlib import Path
from typing import Any
from uuid import UUID

from app.main import create_app
from app.model_backends import (
    JsonObject,
    ModelConversation,
    ModelGeneration,
    ModelReasoning,
    ModelTurn,
    ModelUsage,
    PreparedModelRequest,
)
from app.models import Scene
from app.storage import SceneStorage
from tests.client import TestClient

FAKE_MODEL = "third-protocol/Case-Sensitive"
OTHER_MODEL = "other/available"


class FakeThirdProtocolBackend:
    """Test-only backend with a provider-unlike, versioned payload."""

    def __init__(
        self,
        outputs: list[str],
        *,
        model: str = FAKE_MODEL,
        payload_version: int = 1,
    ) -> None:
        """Queue visible results and select one unrelated wire mapping."""
        self._model = model
        self._outputs = outputs
        self._payload_version = payload_version
        self.conversations: list[ModelConversation] = []
        self.prepared_requests: list[PreparedModelRequest] = []
        self.generate_calls: list[PreparedModelRequest] = []
        self.close_calls = 0

    @property
    def model(self) -> str:
        """Return the exact case-sensitive model identity."""
        return self._model

    def prepare(
        self,
        conversation: ModelConversation,
    ) -> PreparedModelRequest:
        """Map only neutral conversation fields into a fake ritual."""
        self.conversations.append(conversation)
        prepared = PreparedModelRequest(
            payload=_fake_payload(
                conversation,
                self.model,
                self._payload_version,
            )
        )
        self.prepared_requests.append(prepared)
        return prepared

    def generate(
        self,
        prepared: PreparedModelRequest,
    ) -> ModelGeneration:
        """Record one upstream-equivalent call and return one queued result."""
        self.generate_calls.append(prepared)
        if not self._outputs:
            raise RuntimeError("test fake has no queued output")
        return ModelGeneration(
            content=self._outputs.pop(0),
            reasoning=(
                ModelReasoning(
                    type="summary_text",
                    text="observer-only fake reasoning",
                ),
            ),
            usage=ModelUsage(
                input_tokens=13,
                output_tokens=5,
                cache_creation_input_tokens=3,
                cache_read_input_tokens=2,
            ),
        )

    def close(self) -> None:
        """Record lifecycle compatibility without owning a real client."""
        self.close_calls += 1


def _fake_payload(
    conversation: ModelConversation,
    model: str,
    version: int,
) -> JsonObject:
    """Build one of two structurally different fake protocol payloads."""
    if version == 1:
        return {
            "machine": model,
            "ritual": {
                "axiom": conversation.system_prompt,
                "echoes": [
                    {"stimulus": turn.input, "reply": turn.output}
                    for turn in conversation.turns
                ],
                "pending": conversation.current_input,
            },
        }
    return {
        "machine": model,
        "capsule": {
            "charter": conversation.system_prompt,
            "records": [
                {"received": turn.input, "returned": turn.output}
                for turn in conversation.turns
            ],
            "candidate": conversation.current_input,
        },
    }


def _require_json(response: Any, status_code: int = 200) -> dict[str, Any]:
    """Assert one response status and return its JSON object."""
    assert response.status_code == status_code, response.text
    body = response.json()
    assert isinstance(body, dict)
    return body


def _post_scene(
    client: TestClient,
    *,
    model: str = FAKE_MODEL,
    name: str = "HTTP integration",
) -> dict[str, Any]:
    """Create one bound scene through the public API."""
    return _require_json(
        client.post("/api/scenes", json={"name": name, "model": model}),
        201,
    )


def _post_event(
    client: TestClient,
    scene_id: str,
    agent_id: str,
    content: str,
) -> dict[str, Any]:
    """Queue one manual event through the public API."""
    return _require_json(
        client.post(
            f"/api/scenes/{scene_id}/agents/{agent_id}/events",
            json={"content": content},
        ),
        201,
    )


def _generate(
    client: TestClient,
    scene_id: str,
    agent_id: str,
    layer: str,
) -> dict[str, Any]:
    """Generate one browser-held draft through the public API."""
    return _require_json(
        client.post(f"/api/scenes/{scene_id}/agents/{agent_id}/{layer}-drafts")
    )


def _confirm(
    client: TestClient,
    scene_id: str,
    agent_id: str,
    layer: str,
    draft: dict[str, Any],
    *,
    content: str | None = None,
) -> Any:
    """Submit one browser draft without consulting a backend."""
    return client.post(
        f"/api/scenes/{scene_id}/agents/{agent_id}/{layer}-confirmations",
        json={
            "call_id": draft["call_id"],
            "event_id": draft["event_id"],
            "content": draft["content"] if content is None else content,
            "state_token": draft["state_token"],
        },
    )


def _scene_update(scene: dict[str, Any]) -> dict[str, Any]:
    """Extract the editable scene fields accepted by the PUT endpoint."""
    return {
        "name": scene["name"],
        "agents": [
            {
                "id": agent["id"],
                "name": agent["name"],
                "inner_context": {
                    "system_prompt": agent["inner_context"]["system_prompt"]
                },
                "outer_context": {
                    "system_prompt": agent["outer_context"]["system_prompt"]
                },
            }
            for agent in scene["agents"]
        ],
    }


def _json_keys(value: Any) -> set[str]:
    """Return every object key in a nested JSON-safe value."""
    if isinstance(value, dict):
        return set(value) | set().union(
            *(_json_keys(item) for item in value.values())
        )
    if isinstance(value, list):
        return set().union(*(_json_keys(item) for item in value))
    return set()


def _new_pending_inner_draft(
    tmp_path: Path,
    *,
    output: str = "pending inner answer",
) -> tuple[
    SceneStorage,
    TestClient,
    FakeThirdProtocolBackend,
    dict[str, Any],
    dict[str, Any],
]:
    """Create a persisted scene with one unconfirmed inner draft."""
    storage = SceneStorage(tmp_path / "scenes")
    backend = FakeThirdProtocolBackend([output])
    client = TestClient(create_app(storage, {FAKE_MODEL: backend}))
    scene = _post_scene(client)
    scene = _post_event(client, scene["id"], "A", "pending event")
    draft = _generate(client, scene["id"], "A", "inner")
    return storage, client, backend, scene, draft


def test_fake_third_protocol_completes_http_round_with_isolated_context(
    tmp_path: Path,
) -> None:
    """A provider-unlike backend previews and completes both manual stages."""
    storage = SceneStorage(tmp_path / "scenes")
    backend = FakeThirdProtocolBackend(
        ["INNER MODEL OUTPUT", "To B: PUBLIC OUTER OUTPUT"]
    )
    other_backend = FakeThirdProtocolBackend([], model=OTHER_MODEL)
    client = TestClient(
        create_app(
            storage,
            {OTHER_MODEL: other_backend, FAKE_MODEL: backend},
        )
    )

    options = _require_json(client.get("/api/model-options"))
    assert options == {
        "options": [{"model": OTHER_MODEL}, {"model": FAKE_MODEL}]
    }

    scene = _post_scene(client, name="OBSERVER SCENE NAME")
    prompt_values = {
        "A": ("A INNER SYSTEM", "A OUTER SYSTEM"),
        "B": ("B INNER PRIVATE SYSTEM", "B OUTER PRIVATE SYSTEM"),
        "C": ("C INNER PRIVATE SYSTEM", "C OUTER PRIVATE SYSTEM"),
    }
    update = _scene_update(scene)
    for agent in update["agents"]:
        agent_id = agent["id"]
        inner_prompt, outer_prompt = prompt_values[agent_id]
        agent["name"] = f"OBSERVER AGENT {agent_id} NAME"
        agent["inner_context"]["system_prompt"] = inner_prompt
        agent["outer_context"]["system_prompt"] = outer_prompt
    scene = _require_json(client.put(f"/api/scenes/{scene['id']}", json=update))
    scene = _post_event(
        client,
        scene["id"],
        "B",
        "B PRIVATE QUEUE EVENT",
    )
    scene = _post_event(
        client,
        scene["id"],
        "C",
        "C PRIVATE QUEUE EVENT",
    )
    scene = _post_event(client, scene["id"], "A", "A FIRST EVENT")

    first_conversation = ModelConversation(
        system_prompt="A INNER SYSTEM",
        turns=(),
        current_input="外部事件：\nA FIRST EVENT",
    )
    first_preview = _require_json(
        client.get(
            f"/api/scenes/{scene['id']}/agents/A/"
            "model-request-preview?layer=inner"
        )
    )
    assert first_preview["context"] == [
        {"role": "system", "text": "A INNER SYSTEM"},
        {"role": "user", "text": "外部事件：\nA FIRST EVENT"},
    ]
    assert first_preview["request"] == _fake_payload(
        first_conversation,
        FAKE_MODEL,
        1,
    )
    assert backend.generate_calls == []

    calls_before = len(backend.generate_calls)
    inner = _generate(client, scene["id"], "A", "inner")
    assert len(backend.generate_calls) == calls_before + 1
    assert inner["request_snapshot"] == _fake_payload(
        first_conversation,
        FAKE_MODEL,
        1,
    )
    assert inner["reasoning"] == [
        {
            "type": "summary_text",
            "text": "observer-only fake reasoning",
        }
    ]
    calls_before = len(backend.generate_calls)
    _require_json(
        _confirm(
            client,
            scene["id"],
            "A",
            "inner",
            inner,
            content="INNER CONFIRMED EDIT",
        )
    )
    assert len(backend.generate_calls) == calls_before

    outer_input = (
        "外部事件：\nA FIRST EVENT\n\n你内心有一个声音：\nINNER CONFIRMED EDIT"
    )
    outer_conversation = ModelConversation(
        system_prompt="A OUTER SYSTEM",
        turns=(),
        current_input=outer_input,
    )
    outer_preview = _require_json(
        client.get(
            f"/api/scenes/{scene['id']}/agents/A/"
            "model-request-preview?layer=outer"
        )
    )
    assert outer_preview["context"] == [
        {"role": "system", "text": "A OUTER SYSTEM"},
        {"role": "user", "text": outer_input},
    ]
    assert outer_preview["request"] == _fake_payload(
        outer_conversation,
        FAKE_MODEL,
        1,
    )

    calls_before = len(backend.generate_calls)
    outer = _generate(client, scene["id"], "A", "outer")
    assert len(backend.generate_calls) == calls_before + 1
    calls_before = len(backend.generate_calls)
    completed = _require_json(
        _confirm(client, scene["id"], "A", "outer", outer)
    )
    assert len(backend.generate_calls) == calls_before
    assert completed["agents"][1]["pending_events"][0]["content"] == (
        "B PRIVATE QUEUE EVENT"
    )
    assert completed["agents"][1]["pending_events"][1]["content"] == (
        "From A: PUBLIC OUTER OUTPUT"
    )

    completed = _post_event(
        client,
        scene["id"],
        "A",
        "A SECOND EVENT",
    )
    later_input = (
        "外层人格：\nTo B: PUBLIC OUTER OUTPUT\n\n外部事件：\nA SECOND EVENT"
    )
    later_conversation = ModelConversation(
        system_prompt="A INNER SYSTEM",
        turns=(
            ModelTurn(
                input="外部事件：\nA FIRST EVENT",
                output="INNER CONFIRMED EDIT",
            ),
        ),
        current_input=later_input,
    )
    later_preview = _require_json(
        client.get(
            f"/api/scenes/{scene['id']}/agents/A/"
            "model-request-preview?layer=inner"
        )
    )
    assert later_preview["context"] == [
        {"role": "system", "text": "A INNER SYSTEM"},
        {"role": "user", "text": "外部事件：\nA FIRST EVENT"},
        {"role": "assistant", "text": "INNER CONFIRMED EDIT"},
        {"role": "user", "text": later_input},
    ]
    assert later_preview["request"] == _fake_payload(
        later_conversation,
        FAKE_MODEL,
        1,
    )
    assert backend.conversations[-1] == later_conversation
    assert len(backend.generate_calls) == 2

    serialized_request = json.dumps(
        later_preview["request"], ensure_ascii=False
    )
    forbidden_values = [
        "OBSERVER SCENE NAME",
        "OBSERVER AGENT A NAME",
        "OBSERVER AGENT B NAME",
        "OBSERVER AGENT C NAME",
        "A OUTER SYSTEM",
        "B INNER PRIVATE SYSTEM",
        "B OUTER PRIVATE SYSTEM",
        "C INNER PRIVATE SYSTEM",
        "C OUTER PRIVATE SYSTEM",
        "B PRIVATE QUEUE EVENT",
        "C PRIVATE QUEUE EVENT",
        "observer-only fake reasoning",
        completed["id"],
        *(reference["call_id"] for reference in completed["rollback_stack"]),
        *(event["id"] for event in completed["agents"][1]["pending_events"]),
        *(event["id"] for event in completed["agents"][2]["pending_events"]),
    ]
    assert all(value not in serialized_request for value in forbidden_values)
    assert _json_keys(later_preview["request"]) == {
        "axiom",
        "echoes",
        "machine",
        "pending",
        "reply",
        "ritual",
        "stimulus",
    }


def test_valid_draft_confirms_after_restart_without_bound_model(
    tmp_path: Path,
) -> None:
    """Unavailable models block new calls but never an existing confirmation."""
    storage, _client, backend, scene, draft = _new_pending_inner_draft(tmp_path)
    other_backend = FakeThirdProtocolBackend([], model=OTHER_MODEL)
    restarted = TestClient(create_app(storage, {OTHER_MODEL: other_backend}))

    assert (
        restarted.get(
            f"/api/scenes/{scene['id']}/agents/A/"
            "model-request-preview?layer=inner"
        ).status_code
        == 409
    )
    assert (
        restarted.post(
            f"/api/scenes/{scene['id']}/agents/A/inner-drafts"
        ).status_code
        == 409
    )
    calls_before = len(backend.generate_calls)

    confirmed = _require_json(
        _confirm(restarted, scene["id"], "A", "inner", draft)
    )

    assert len(backend.generate_calls) == calls_before
    assert other_backend.generate_calls == []
    assert (
        confirmed["agents"][0]["inner_context"]["turns"][0]["output"]
        == "pending inner answer"
    )


def test_payload_versions_share_state_token_for_identical_business_state(
    tmp_path: Path,
) -> None:
    """Adapter-only payload changes do not invalidate browser drafts."""
    storage = SceneStorage(tmp_path / "scenes")
    first_backend = FakeThirdProtocolBackend(["same answer"], payload_version=1)
    first_client = TestClient(create_app(storage, {FAKE_MODEL: first_backend}))
    scene = _post_scene(first_client)
    scene = _post_event(first_client, scene["id"], "A", "same event")
    first_draft = _generate(first_client, scene["id"], "A", "inner")

    second_backend = FakeThirdProtocolBackend(
        ["same answer"], payload_version=2
    )
    second_client = TestClient(
        create_app(storage, {FAKE_MODEL: second_backend})
    )
    second_draft = _generate(second_client, scene["id"], "A", "inner")

    assert first_draft["request_snapshot"] != second_draft["request_snapshot"]
    assert first_draft["state_token"] == second_draft["state_token"]
    assert len(first_backend.generate_calls) == 1
    assert len(second_backend.generate_calls) == 1


def test_changed_event_rejects_old_draft_over_http(tmp_path: Path) -> None:
    """The complete current event participates in stale-draft detection."""
    _storage, client, backend, scene, draft = _new_pending_inner_draft(tmp_path)
    event_id = scene["agents"][0]["pending_events"][0]["id"]
    edited = client.put(
        f"/api/scenes/{scene['id']}/agents/A/events/{event_id}",
        json={"content": "changed event"},
    )
    assert edited.status_code == 200
    calls_before = len(backend.generate_calls)

    response = _confirm(client, scene["id"], "A", "inner", draft)

    assert response.status_code == 409
    assert len(backend.generate_calls) == calls_before


def test_changed_prompt_rejects_old_draft_over_http(tmp_path: Path) -> None:
    """The selected layer's system prompt participates in the state token."""
    _storage, client, backend, scene, draft = _new_pending_inner_draft(tmp_path)
    update = _scene_update(scene)
    update["agents"][0]["inner_context"]["system_prompt"] = (
        "changed inner prompt"
    )
    saved = client.put(f"/api/scenes/{scene['id']}", json=update)
    assert saved.status_code == 200
    calls_before = len(backend.generate_calls)

    response = _confirm(client, scene["id"], "A", "inner", draft)

    assert response.status_code == 409
    assert len(backend.generate_calls) == calls_before


def test_changed_model_rejects_old_draft_over_http(tmp_path: Path) -> None:
    """The immutable scene model identity participates in the state token."""
    storage, client, backend, scene, draft = _new_pending_inner_draft(tmp_path)
    scene_id = UUID(scene["id"])
    storage.mutate(
        scene_id,
        lambda current: current.model_copy(update={"model": OTHER_MODEL}),
    )
    calls_before = len(backend.generate_calls)

    response = _confirm(client, scene["id"], "A", "inner", draft)

    assert response.status_code == 409
    assert len(backend.generate_calls) == calls_before


def test_changed_same_layer_history_rejects_old_draft_over_http(
    tmp_path: Path,
) -> None:
    """Every confirmed turn in the selected layer participates in the token."""
    storage = SceneStorage(tmp_path / "scenes")
    backend = FakeThirdProtocolBackend(
        ["first inner", "To B: first outer", "second inner"]
    )
    client = TestClient(create_app(storage, {FAKE_MODEL: backend}))
    scene = _post_scene(client)
    scene = _post_event(client, scene["id"], "A", "first event")
    first_inner = _generate(client, scene["id"], "A", "inner")
    scene = _require_json(
        _confirm(client, scene["id"], "A", "inner", first_inner)
    )
    first_outer = _generate(client, scene["id"], "A", "outer")
    scene = _require_json(
        _confirm(client, scene["id"], "A", "outer", first_outer)
    )
    scene = _post_event(client, scene["id"], "A", "second event")
    second_inner = _generate(client, scene["id"], "A", "inner")

    def change_inner_history(current: Scene) -> Scene:
        """Replace one confirmed inner output while preserving scene shape."""
        agent = current.agents[0]
        changed_turn = agent.inner_context.turns[0].model_copy(
            update={"output": "changed prior inner output"}
        )
        changed_agent = agent.model_copy(
            update={
                "inner_context": agent.inner_context.model_copy(
                    update={
                        "turns": [
                            changed_turn,
                            *agent.inner_context.turns[1:],
                        ]
                    }
                )
            }
        )
        return current.model_copy(
            update={"agents": [changed_agent, *current.agents[1:]]}
        )

    storage.mutate(UUID(scene["id"]), change_inner_history)
    calls_before = len(backend.generate_calls)

    response = _confirm(
        client,
        scene["id"],
        "A",
        "inner",
        second_inner,
    )

    assert response.status_code == 409
    assert len(backend.generate_calls) == calls_before
