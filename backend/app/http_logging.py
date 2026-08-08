"""ASGI request correlation and structured HTTP lifecycle logging."""

import json
import logging
from collections.abc import Awaitable, Callable
from contextlib import suppress
from time import perf_counter
from typing import Any
from urllib.parse import parse_qsl
from uuid import uuid4

from app.structured_logging import (
    bind_log_context,
    bind_redaction_secrets,
    is_credential_field,
    log_event,
)

LOGGER = logging.getLogger(__name__)

type Scope = dict[str, Any]
type Message = dict[str, Any]
type Receive = Callable[[], Awaitable[Message]]
type Send = Callable[[Message], Awaitable[None]]
type AsgiApp = Callable[[Scope, Receive, Send], Awaitable[None]]


class RequestLoggingMiddleware:
    """Generate trusted request IDs and log every HTTP request lifecycle."""

    def __init__(self, app: AsgiApp) -> None:
        """Wrap one ASGI application."""
        self._app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        """Buffer one request, replay it, and observe the complete response."""
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        request_id = str(uuid4())
        started_at = perf_counter()
        request_messages = await _read_request_messages(receive)
        request_body = b"".join(
            message.get("body", b"")
            for message in request_messages
            if message["type"] == "http.request"
        )
        request_credentials = _request_credentials(scope, request_body)
        scope.setdefault("state", {})["request_body_for_logging"] = request_body
        replay_index = 0
        response_status = 500
        response_headers: list[tuple[bytes, bytes]] = []
        response_body_parts: list[bytes] = []
        response_completed = False
        response_started = False

        async def replay_receive() -> Message:
            nonlocal replay_index
            if replay_index < len(request_messages):
                message = request_messages[replay_index]
                replay_index += 1
                return message
            # The complete body has already been replayed; later reads are
            # disconnect probes and must not block on an exhausted transport.
            return {"type": "http.disconnect"}

        async def observe_send(message: Message) -> None:
            nonlocal response_status, response_headers, response_completed
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                response_status = message["status"]
                original_headers = list(message.get("headers", []))
                response_headers = [
                    header
                    for header in original_headers
                    if header[0].lower() != b"x-request-id"
                ]
                response_headers.append(
                    (b"x-request-id", request_id.encode("ascii"))
                )
                message = {**message, "headers": response_headers}
            elif message["type"] == "http.response.body":
                if response_status >= 400:
                    response_body_parts.append(message.get("body", b""))
                if not message.get("more_body", False):
                    response_completed = True
            await send(message)

        request_fields = _request_metadata(scope)
        with (
            bind_redaction_secrets(request_credentials),
            bind_log_context(request_id=request_id),
        ):
            log_event(
                LOGGER,
                logging.DEBUG,
                "http.request.started",
                "HTTP request started.",
                **request_fields,
            )
            try:
                await self._app(scope, replay_receive, observe_send)
            except Exception as error:
                log_event(
                    LOGGER,
                    logging.ERROR,
                    "http.request.unhandled_exception",
                    "HTTP request raised an unhandled exception.",
                    exception=error,
                    request=_request_details(scope, request_body),
                    duration_ms=_duration_ms(started_at),
                )
                if response_started:
                    raise
                await _send_internal_server_error(scope, observe_send)

            if response_completed:
                completion_fields: dict[str, object] = {
                    **_request_metadata(scope),
                    "status_code": response_status,
                    "duration_ms": _duration_ms(started_at),
                }
                if response_status >= 400:
                    completion_fields.update(
                        {
                            "request": _request_details(scope, request_body),
                            "response": {
                                "headers": _headers_for_logging(
                                    response_headers
                                ),
                                "body": _body_for_logging(
                                    b"".join(response_body_parts),
                                    response_headers,
                                ),
                            },
                        }
                    )
                log_event(
                    LOGGER,
                    _status_level(response_status),
                    "http.request.completed",
                    "HTTP request completed.",
                    **completion_fields,
                )


async def _read_request_messages(receive: Receive) -> list[Message]:
    """Read the complete request once so failures always retain its body."""
    messages: list[Message] = []
    while True:
        message = await receive()
        messages.append(message)
        if message["type"] != "http.request" or not message.get(
            "more_body", False
        ):
            return messages


async def _send_internal_server_error(
    scope: Scope,
    send: Send,
) -> None:
    """Preserve Starlette's default plain-text 500 response semantics."""
    del scope
    body = b"Internal Server Error"
    await send(
        {
            "type": "http.response.start",
            "status": 500,
            "headers": [
                (b"content-length", str(len(body)).encode("ascii")),
                (b"content-type", b"text/plain; charset=utf-8"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def _request_metadata(scope: Scope) -> dict[str, object]:
    """Return body-free fields shared by request start and completion."""
    route = scope.get("route")
    route_path = getattr(route, "path", scope.get("path", ""))
    client = scope.get("client")
    return {
        **_path_context(scope),
        "method": scope.get("method"),
        "route": route_path,
        "path": scope.get("path"),
        "query_string": scope.get("query_string", b"").decode(
            "utf-8", errors="replace"
        ),
        "client": ({"host": client[0], "port": client[1]} if client else None),
    }


def _path_context(scope: Scope) -> dict[str, object]:
    """Extract known correlation fields without trusting client headers."""
    path_parameters = scope.get("path_params", {})
    parts = str(scope.get("path", "")).strip("/").split("/")
    context: dict[str, object] = {}
    if "scene_id" in path_parameters:
        context["scene_id"] = path_parameters["scene_id"]
    elif len(parts) >= 3 and parts[:2] == ["api", "scenes"]:
        context["scene_id"] = parts[2]
    if "agent_id" in path_parameters:
        context["agent_id"] = path_parameters["agent_id"]
    elif len(parts) >= 5 and parts[3] == "agents":
        context["agent_id"] = parts[4]
    if parts:
        if parts[-1].startswith("inner-"):
            context["layer"] = "inner"
        elif parts[-1].startswith("outer-"):
            context["layer"] = "outer"
    return context


def _request_details(scope: Scope, body: bytes) -> dict[str, object]:
    """Return the untruncated request fields reserved for error events."""
    headers = list(scope.get("headers", []))
    return {
        **_request_metadata(scope),
        "headers": _headers_for_logging(headers),
        "body": _body_for_logging(body, headers),
    }


def _headers_for_logging(
    headers: list[tuple[bytes, bytes]],
) -> dict[str, object]:
    """Preserve duplicate headers in a key-aware redactable mapping."""
    grouped: dict[str, object] = {}
    for raw_name, raw_value in headers:
        name = raw_name.decode("latin-1")
        value = raw_value.decode("latin-1")
        current = grouped.get(name)
        if current is None:
            grouped[name] = value
        elif isinstance(current, list):
            current.append(value)
        else:
            grouped[name] = [current, value]
    return grouped


def _body_for_logging(
    body: bytes,
    headers: list[tuple[bytes, bytes]],
) -> object:
    """Decode a complete body and preserve nested JSON for redaction."""
    if not body:
        return ""
    text = body.decode("utf-8", errors="replace")
    content_type = next(
        (
            value.decode("latin-1").casefold()
            for name, value in headers
            if name.lower() == b"content-type"
        ),
        "",
    )
    if "json" not in content_type:
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _request_credentials(scope: Scope, body: bytes) -> list[str]:
    """Return request auth values so echoed error details are also safe."""
    values: list[str] = []
    headers = list(scope.get("headers", []))
    for name, value in headers:
        if is_credential_field(name.decode("latin-1")):
            values.append(value.decode("latin-1"))
    for name, value in parse_qsl(
        scope.get("query_string", b"").decode("utf-8", errors="replace"),
        keep_blank_values=True,
    ):
        if is_credential_field(name):
            values.append(value)

    content_type = next(
        (
            value.decode("latin-1").casefold()
            for name, value in headers
            if name.lower() == b"content-type"
        ),
        "",
    )
    if body and "json" in content_type:
        with suppress(json.JSONDecodeError, UnicodeDecodeError):
            _collect_credentials(json.loads(body), values)
    return values


def _collect_credentials(value: object, destination: list[str]) -> None:
    """Collect nested JSON credential values without changing the request."""
    if isinstance(value, dict):
        for key, item in value.items():
            if is_credential_field(str(key)):
                if isinstance(item, str):
                    destination.append(item)
            else:
                _collect_credentials(item, destination)
    elif isinstance(value, list):
        for item in value:
            _collect_credentials(item, destination)


def _status_level(status_code: int) -> int:
    """Map HTTP outcomes to the requested operational severity."""
    if status_code >= 500:
        return logging.ERROR
    if status_code >= 400:
        return logging.WARNING
    return logging.INFO


def _duration_ms(started_at: float) -> float:
    """Return a stable millisecond duration without rounding to zero."""
    return round((perf_counter() - started_at) * 1000, 3)
