"""Structured logging, request correlation, and redaction tests."""

import io
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from uuid import UUID

from app.main import create_app
from app.storage import SceneStorage
from app.structured_logging import (
    REDACTED,
    JsonFormatter,
    bind_log_context,
    log_event,
    register_secrets,
)
from tests.client import TestClient
from tests.helpers import (
    FakeBackend,
    fill_prompts,
    generate_draft,
    make_client,
    post_event,
    post_scene,
)

MODEL = "structured-log-model"


@contextmanager
def _captured_json_logs() -> Iterator[io.StringIO]:
    """Capture application records through the production JSON formatter."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    previous_level = root.level
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    try:
        yield stream
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)


def _documents(stream: io.StringIO) -> list[dict[str, object]]:
    """Parse every non-empty captured line as one JSON object."""
    return [json.loads(line) for line in stream.getvalue().splitlines() if line]


def test_json_formatter_emits_stable_unicode_event_and_full_stack() -> None:
    """Every event is one JSON line with UTC time and all context fields."""
    logger = logging.getLogger("tests.structured")
    long_message = "故障正文" * 1000

    with _captured_json_logs() as stream:
        try:
            raise RuntimeError(long_message)
        except RuntimeError as error:
            with bind_log_context(request_id="request-1", layer="inner"):
                log_event(
                    logger,
                    logging.ERROR,
                    "test.failed",
                    "中文消息",
                    exception=error,
                )

    [document] = _documents(stream)
    assert document["timestamp"].endswith("Z")
    assert document["level"] == "ERROR"
    assert document["logger"] == "tests.structured"
    assert document["event"] == "test.failed"
    assert document["message"] == "中文消息"
    assert document["request_id"] == "request-1"
    assert document["layer"] == "inner"
    assert document["scene_id"] is None
    assert long_message in document["exception"]
    assert "RuntimeError" in document["exception"]


def test_redaction_covers_nested_auth_url_body_and_exception() -> None:
    """Registered keys and credential fields never survive serialization."""
    secret = "test-key-redaction-%2F-value"
    register_secrets([secret])
    logger = logging.getLogger("tests.redaction")

    with _captured_json_logs() as stream:
        try:
            raise RuntimeError(
                f"https://provider.test?api_key={secret}&safe=kept"
            )
        except RuntimeError as error:
            log_event(
                logger,
                logging.ERROR,
                "redaction.failed",
                f"provider body contains {secret}",
                exception=error,
                headers={
                    "Authorization": "Bearer arbitrary-auth-value",
                    "X-Api-Key": "another-unregistered-key",
                },
                nested={
                    "api_key": secret,
                    "provider_body": {"safe": f"before-{secret}-after"},
                },
            )

    serialized = stream.getvalue()
    assert secret not in serialized
    assert "arbitrary-auth-value" not in serialized
    assert "another-unregistered-key" not in serialized
    assert serialized.count(REDACTED) >= 5
    document = _documents(stream)[0]
    assert document["nested"]["provider_body"]["safe"] == (
        f"before-{REDACTED}-after"
    )


def test_http_request_id_is_server_generated_and_success_bodies_are_absent(
    tmp_path,
) -> None:
    """HTTP and business events correlate without logging success bodies."""
    client = make_client(tmp_path, FakeBackend([], model=MODEL))
    client_request_id = "00000000-0000-0000-0000-000000000000"
    scene_name = "成功正文不得进入日志"

    with _captured_json_logs() as stream:
        response = client.post(
            "/api/scenes",
            json={"name": scene_name, "model": MODEL},
            headers={"X-Request-ID": client_request_id},
        )

    assert response.status_code == 201
    response_request_id = response.headers["X-Request-ID"]
    assert response_request_id != client_request_id
    UUID(response_request_id)

    documents = _documents(stream)
    relevant = [
        document
        for document in documents
        if document["event"]
        in {"http.request.started", "scene.created", "http.request.completed"}
    ]
    assert len(relevant) == 3
    assert {document["request_id"] for document in relevant} == {
        response_request_id
    }
    completion = next(
        document
        for document in relevant
        if document["event"] == "http.request.completed"
    )
    assert completion["status_code"] == 201
    assert "request" not in completion
    assert "response" not in completion
    assert scene_name not in stream.getvalue()


def test_http_error_logs_complete_bodies_and_redacts_auth(tmp_path) -> None:
    """A 4xx event retains full bodies but strips credential field values."""
    client = make_client(tmp_path, FakeBackend([], model=MODEL))
    content = "完整错误请求正文" * 700

    with _captured_json_logs() as stream:
        response = client.post(
            "/api/scenes",
            json={
                "name": content,
                "model": MODEL,
                "authorization": "body-auth-secret",
            },
            headers={"Authorization": "Bearer header-auth-secret"},
        )

    assert response.status_code == 422
    completion = next(
        document
        for document in _documents(stream)
        if document["event"] == "http.request.completed"
    )
    assert completion["level"] == "WARNING"
    assert completion["request"]["body"]["name"] == content
    assert completion["request"]["body"]["authorization"] == REDACTED
    logged_detail = completion["response"]["body"]["detail"][0]
    assert logged_detail["loc"] == ["body", "authorization"]
    assert logged_detail["input"] == REDACTED
    serialized = stream.getvalue()
    assert "body-auth-secret" not in serialized
    assert "header-auth-secret" not in serialized


def test_model_failure_logs_full_context_with_call_and_request_ids(
    tmp_path,
) -> None:
    """Workflow failures keep diagnostic content server-side and correlated."""
    secret = "workflow-provider-secret"
    register_secrets([secret])
    backend = FakeBackend(
        [RuntimeError(f"provider exploded with {secret}")],
        model=MODEL,
    )
    client = make_client(tmp_path, backend)
    scene = fill_prompts(client, post_scene(client, model=MODEL))
    event_content = "模型失败时必须保留的实验内容" * 300
    scene = post_event(client, scene["id"], "A", event_content)

    with _captured_json_logs() as stream:
        response = client.post(
            f"/api/scenes/{scene['id']}/agents/A/inner-drafts"
        )

    assert response.status_code == 502
    assert response.json() == {"detail": "Model request failed."}
    response_request_id = response.headers["X-Request-ID"]
    documents = _documents(stream)
    started = next(
        document
        for document in documents
        if document["event"] == "model.call.started"
    )
    failed = next(
        document
        for document in documents
        if document["event"] == "model.call.failed"
    )
    assert started["call_id"] == failed["call_id"]
    UUID(started["call_id"])
    assert started["request_id"] == response_request_id
    assert failed["request_id"] == response_request_id
    assert failed["conversation"]["current_input"] == (
        f"外部事件：\n{event_content}"
    )
    assert "RuntimeError" in failed["exception"]
    assert secret not in stream.getvalue()


def test_model_success_logs_usage_hashes_and_final_call_id(tmp_path) -> None:
    """Successful generation logs metrics but no request or response text."""
    visible_output = "只允许浏览器看到的成功模型正文"
    backend = FakeBackend([visible_output], model=MODEL)
    client = make_client(tmp_path, backend)
    scene = fill_prompts(client, post_scene(client, model=MODEL))
    scene = post_event(client, scene["id"], "A", "成功模型输入正文")

    with _captured_json_logs() as stream:
        draft = generate_draft(client, scene["id"], "A", "inner")

    documents = _documents(stream)
    completed = next(
        document
        for document in documents
        if document["event"] == "model.call.completed"
    )
    assert completed["call_id"] == draft["call_id"]
    assert completed["input_tokens"] == 10
    assert completed["output_tokens"] == 4
    assert completed["cache_creation_input_tokens"] == 2
    assert completed["cache_read_input_tokens"] == 3
    assert completed["reasoning_count"] == 1
    assert completed["output_length"] == len(visible_output)
    assert "output_sha256" in completed
    assert visible_output not in stream.getvalue()


def test_unhandled_exception_returns_correlated_500_and_full_stack(
    tmp_path,
) -> None:
    """Unexpected failures retain the default 500 body and request ID."""
    application = create_app(
        SceneStorage(tmp_path),
        FakeBackend([], model=MODEL),
    )

    @application.post("/api/test-unhandled")
    async def raise_unhandled() -> None:
        """Raise one deterministic application error for the middleware."""
        raise RuntimeError("unhandled diagnostic message")

    client = TestClient(application)
    with _captured_json_logs() as stream:
        response = client.post(
            "/api/test-unhandled",
            json={"complete": "unhandled request body"},
        )

    assert response.status_code == 500
    assert response.text == "Internal Server Error"
    request_id = response.headers["X-Request-ID"]
    UUID(request_id)
    documents = _documents(stream)
    unhandled = next(
        document
        for document in documents
        if document["event"] == "http.request.unhandled_exception"
    )
    completion = next(
        document
        for document in documents
        if document["event"] == "http.request.completed"
    )
    assert unhandled["request_id"] == request_id
    assert completion["request_id"] == request_id
    assert unhandled["request"]["body"] == {
        "complete": "unhandled request body"
    }
    assert "RuntimeError" in unhandled["exception"]
    assert completion["status_code"] == 500
    assert completion["response"]["body"] == "Internal Server Error"
