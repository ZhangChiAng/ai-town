"""HTTP integration tests for protocol-neutral model backend workflows."""

import json
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from app.main import create_app
from app.model_backends import JsonObject, ModelConversation, ModelTurn
from app.storage import SceneStorage
from tests import helpers
from tests.client import TestClient

FakeBackend = helpers.FakeBackend
_confirm = helpers.confirm_draft
_generate = helpers.generate_draft
_post_event = helpers.post_event
_require_json = helpers.require_json

FAKE_MODEL = "third-protocol/Case-Sensitive"
OTHER_MODEL = "other/available"


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


def _backend(
    outputs: list[str],
    *,
    model: str = FAKE_MODEL,
    version: int = 1,
) -> FakeBackend:
    """Build a fake third-protocol backend with a versioned ritual payload."""
    return FakeBackend(
        outputs,
        model=model,
        payload_builder=lambda c, m: _fake_payload(c, m, version),
    )


def _post_scene(
    client: TestClient,
    *,
    name: str = "HTTP integration",
) -> dict[str, Any]:
    """Create one bound scene through the public API."""
    return helpers.post_scene(client, model=FAKE_MODEL, name=name)


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
    FakeBackend,
    dict[str, Any],
    dict[str, Any],
]:
    """Create a persisted scene with one unconfirmed inner draft."""
    storage = SceneStorage(tmp_path / "scenes")
    backend = _backend([output])
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
    backend = _backend(["INNER MODEL OUTPUT", "To B: PUBLIC OUTER OUTPUT"])
    other_backend = _backend([], model=OTHER_MODEL)
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
        "外层人格上一轮对 Agent B（OBSERVER AGENT B NAME）说：\n"
        "PUBLIC OUTER OUTPUT\n\n外部事件：\nA SECOND EVENT"
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
    other_backend = _backend([], model=OTHER_MODEL)
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


def _mutate_event(
    client: TestClient,
    storage: SceneStorage,
    scene: dict[str, Any],
) -> None:
    """Replace the queued event content after the draft was generated."""
    event_id = scene["agents"][0]["pending_events"][0]["id"]
    edited = client.put(
        f"/api/scenes/{scene['id']}/agents/A/events/{event_id}",
        json={"content": "changed event"},
    )
    assert edited.status_code == 200


def _mutate_prompt(
    client: TestClient,
    storage: SceneStorage,
    scene: dict[str, Any],
) -> None:
    """Change the selected layer's system prompt after the draft."""
    update = _scene_update(scene)
    update["agents"][0]["inner_context"]["system_prompt"] = (
        "changed inner prompt"
    )
    saved = client.put(f"/api/scenes/{scene['id']}", json=update)
    assert saved.status_code == 200


def _mutate_model(
    client: TestClient,
    storage: SceneStorage,
    scene: dict[str, Any],
) -> None:
    """Rebind the scene to a different model after the draft."""
    storage.mutate(
        UUID(scene["id"]),
        lambda current: current.model_copy(update={"model": OTHER_MODEL}),
    )


@pytest.mark.parametrize(
    "mutate",
    [_mutate_event, _mutate_prompt, _mutate_model],
    ids=["event", "prompt", "model"],
)
def test_changed_business_state_rejects_old_draft_over_http(
    tmp_path: Path,
    mutate,
) -> None:
    """A stale draft is rejected with 409 without a backend call."""
    storage, client, backend, scene, draft = _new_pending_inner_draft(tmp_path)
    mutate(client, storage, scene)
    calls_before = len(backend.generate_calls)

    response = _confirm(client, scene["id"], "A", "inner", draft)

    assert response.status_code == 409
    assert len(backend.generate_calls) == calls_before
