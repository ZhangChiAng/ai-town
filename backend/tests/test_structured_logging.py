"""Structured logging, request correlation, and redaction tests."""

import io
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID

import pytest

from app.main import create_app
from app.model_backends.contracts import (
    LoggedModelError,
    ModelConversation,
    ModelGeneration,
)
from app.storage import SceneStorage
from app.structured_logging import (
    JSON_LOG_DISPLAY_PATH,
    JSON_LOG_PATH,
    REDACTED,
    HumanConsoleHandler,
    HumanFormatter,
    JsonFormatter,
    JsonlRotatingFileHandler,
    bind_log_context,
    log_event,
    register_secrets,
)
from tests.client import TestClient
from tests.helpers import (
    FakeBackend,
    confirm_draft,
    fill_prompts,
    generate_draft,
    make_client,
    post_event,
    post_scene,
)

MODEL = "structured-log-model"


class _TtyStream(io.StringIO):
    """In-memory stream that reports terminal color support."""

    def isatty(self) -> bool:
        """Pretend this capture is connected to an interactive terminal."""
        return True


class _DetailedFailureBackend(FakeBackend):
    """Backend double that logs its own detailed failure before raising."""

    async def generate(
        self,
        conversation: ModelConversation,
    ) -> ModelGeneration:
        """Emit one adapter-style event and raise its safe marker."""
        self.generate_calls.append(conversation)
        log_event(
            logging.getLogger("tests.detailed_backend"),
            logging.ERROR,
            "model.provider_request.failed",
            "Upstream model request timed out.",
            failure_category="upstream_timeout",
            exception_type="ReadTimeout",
        )
        raise LoggedModelError("detailed diagnostics already logged")


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


def test_human_and_json_formats_share_full_redacted_document() -> None:
    """Both channels use the same safe values without terminal truncation."""
    secret = "shared-format-secret"
    register_secrets([secret])
    human_stream = io.StringIO()
    json_stream = io.StringIO()
    human_handler = HumanConsoleHandler(human_stream)
    human_handler.setFormatter(HumanFormatter())
    json_handler = logging.StreamHandler(json_stream)
    json_handler.setFormatter(JsonFormatter())
    logger = logging.Logger("tests.shared_formats", level=logging.DEBUG)
    logger.addHandler(human_handler)
    logger.addHandler(json_handler)
    visible_output = "完整模型输出" * 1500

    with bind_log_context(
        request_id="request-human",
        scene_id="",
        call_id="call-human",
    ):
        log_event(
            logger,
            logging.ERROR,
            "model.workflow.failed",
            "外层格式错误",
            failure_category="outer_protocol",
            visible_output=visible_output,
            nested={
                "Authorization": f"Bearer {secret}",
                "safe": f"before-{secret}-after",
            },
        )

    human = human_stream.getvalue()
    [document] = _documents(json_stream)
    assert "外层格式错误" in human
    assert visible_output in human
    assert document["visible_output"] == visible_output
    assert document["nested"] == {
        "Authorization": REDACTED,
        "safe": f"before-{REDACTED}-after",
    }
    assert secret not in human
    assert secret not in json_stream.getvalue()
    assert 'request_id="request-human"' in human
    assert 'call_id="call-human"' in human
    assert f'json_log="{JSON_LOG_DISPLAY_PATH}"' in human
    assert "scene_id=" not in human
    assert len(json_stream.getvalue().splitlines()) == 1


def test_human_terminal_color_is_tty_only_and_honors_no_color(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ANSI styling follows TTY detection and the standard opt-out."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    color_stream = _TtyStream()
    color_handler = HumanConsoleHandler(color_stream)
    color_handler.setFormatter(HumanFormatter())
    logger = logging.Logger("tests.color", level=logging.INFO)
    logger.addHandler(color_handler)
    log_event(logger, logging.INFO, "test.colored", "彩色消息")
    assert "\x1b[" in color_stream.getvalue()

    monkeypatch.setenv("NO_COLOR", "1")
    plain_stream = _TtyStream()
    plain_handler = HumanConsoleHandler(plain_stream)
    plain_handler.setFormatter(HumanFormatter())
    logger.handlers = [plain_handler]
    log_event(logger, logging.INFO, "test.plain", "无颜色消息")
    assert "\x1b[" not in plain_stream.getvalue()


def test_jsonl_handler_rotates_five_backups_and_deletes_oldest(
    tmp_path: Path,
) -> None:
    """JSONL remains parseable while bounded rotation removes old files."""
    path = tmp_path / "nested" / "ai-town.jsonl"
    handler = JsonlRotatingFileHandler(
        filename=path,
        max_bytes=500,
        backup_count=5,
    )
    handler.setFormatter(JsonFormatter())
    logger = logging.Logger("tests.rotation", level=logging.INFO)
    logger.addHandler(handler)
    try:
        for sequence in range(10):
            log_event(
                logger,
                logging.INFO,
                "rotation.event",
                "轮转测试",
                sequence=sequence,
                payload="x" * 600,
            )
    finally:
        handler.close()

    files = [path, *[Path(f"{path}.{index}") for index in range(1, 6)]]
    assert all(file.exists() for file in files)
    assert not Path(f"{path}.6").exists()
    documents = [
        json.loads(line)
        for file in files
        for line in file.read_text(encoding="utf-8").splitlines()
    ]
    assert {document["sequence"] for document in documents} == set(range(4, 10))


def test_production_config_archives_app_debug_but_not_uvicorn() -> None:
    """The default routing keeps reload-process output terminal-only."""
    repository_root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (repository_root / "backend" / "logging.json").read_text(
            encoding="utf-8"
        )
    )
    assert repository_root / JSON_LOG_DISPLAY_PATH == JSON_LOG_PATH
    assert JSON_LOG_PATH.is_absolute()
    assert config["root"] == {
        "handlers": ["terminal", "jsonl"],
        "level": "DEBUG",
    }
    assert config["handlers"]["terminal"]["level"] == "INFO"
    assert config["handlers"]["jsonl"]["level"] == "DEBUG"
    assert config["handlers"]["jsonl"]["filters"] == ["structured_event"]
    assert config["loggers"]["uvicorn"]["handlers"] == ["terminal"]


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
    assert len(relevant) == 2
    assert all(
        document["event"] != "http.request.started" for document in documents
    )
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
    assert failed["failure_category"] == "internal"
    assert "RuntimeError" in failed["exception"]
    assert secret not in stream.getvalue()


def test_outer_protocol_failure_logs_reason_and_full_visible_output(
    tmp_path: Path,
) -> None:
    """Invalid outer text remains complete beside its validation reason."""
    visible_output = "不带收件人协议的原始输出" * 1200
    backend = FakeBackend(["内层判断", visible_output], model=MODEL)
    client = make_client(tmp_path, backend)
    scene = fill_prompts(client, post_scene(client, model=MODEL))
    scene = post_event(client, scene["id"], "A", "触发外层格式校验")
    inner = generate_draft(client, scene["id"], "A", "inner")
    confirmation = confirm_draft(
        client,
        scene["id"],
        "A",
        "inner",
        inner,
        reasoning=inner["reasoning"],
    )
    assert confirmation.status_code == 200

    with _captured_json_logs() as stream:
        response = client.post(
            f"/api/scenes/{scene['id']}/agents/A/outer-drafts"
        )

    assert response.status_code == 502
    failure = next(
        document
        for document in _documents(stream)
        if document["event"] == "model.workflow.failed"
    )
    assert failure["failure_category"] == "outer_protocol"
    assert failure["visible_output"] == visible_output
    assert failure["validation_error"]
    assert "ValueError" in failure["exception"]


def test_detailed_backend_failure_does_not_add_generic_duplicate(
    tmp_path: Path,
) -> None:
    """Adapter diagnostics suppress only the workflow's generic fallback."""
    backend = _DetailedFailureBackend([], model=MODEL)
    client = make_client(tmp_path, backend)
    scene = fill_prompts(client, post_scene(client, model=MODEL))
    scene = post_event(client, scene["id"], "A", "触发详细适配器错误")

    with _captured_json_logs() as stream:
        response = client.post(
            f"/api/scenes/{scene['id']}/agents/A/inner-drafts"
        )

    assert response.status_code == 502
    documents = _documents(stream)
    detailed = [
        document
        for document in documents
        if document["event"] == "model.provider_request.failed"
    ]
    assert len(detailed) == 1
    assert detailed[0]["failure_category"] == "upstream_timeout"
    assert all(
        document["event"] != "model.call.failed" for document in documents
    )


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
