"""Single-line JSON logging, correlation context, and secret redaction."""

import hashlib
import json
import logging
import re
import threading
import traceback
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from urllib.parse import quote, quote_plus
from uuid import UUID

REDACTED = "[REDACTED]"

CONTEXT_FIELDS = (
    "request_id",
    "scene_id",
    "agent_id",
    "layer",
    "call_id",
    "model",
    "provider",
)
_RESERVED_FIELDS = frozenset(
    {"timestamp", "level", "logger", "event", "message", *CONTEXT_FIELDS}
)
_LOG_CONTEXT: ContextVar[dict[str, object] | None] = ContextVar(
    "structured_log_context",
    default=None,
)
_SECRET_LOCK = threading.Lock()
_SECRET_VALUES: set[str] = set()
_LOCAL_SECRET_VALUES: ContextVar[tuple[str, ...] | None] = ContextVar(
    "structured_log_local_secrets",
    default=None,
)
_CREDENTIAL_KEYS = frozenset(
    {
        "authorization",
        "proxyauthorization",
        "apikey",
        "xapikey",
        "token",
        "accesstoken",
        "authtoken",
        "bearertoken",
        "refreshtoken",
        "idtoken",
        "clientsecret",
        "cookie",
        "setcookie",
    }
)
_INLINE_CREDENTIAL_PATTERN = re.compile(
    r"(?i)([\"']?(?:authorization|proxy[-_]?authorization|x?[-_]?api[-_]?key"
    r"|access[-_]?token|auth[-_]?token|bearer[-_]?token|refresh[-_]?token"
    r"|id[-_]?token|client[-_]?secret)[\"']?\s*[:=]\s*[\"']?)"
    r"([^\"'&,}\s]+)"
)
_BEARER_PATTERN = re.compile(r"(?i)(\bbearer\s+)([^\s,;]+)")


def register_secrets(values: Iterable[str]) -> None:
    """Register resolved credentials that must never reach serialized logs."""
    variants = _secret_variants(values)
    with _SECRET_LOCK:
        _SECRET_VALUES.update(variant for variant in variants if variant)


@contextmanager
def bind_redaction_secrets(values: Iterable[str]) -> Iterator[None]:
    """Limit request-derived authentication redaction to one async context."""
    current = _LOCAL_SECRET_VALUES.get() or ()
    token = _LOCAL_SECRET_VALUES.set(
        tuple({*current, *_secret_variants(values)})
    )
    try:
        yield
    finally:
        _LOCAL_SECRET_VALUES.reset(token)


@contextmanager
def bind_log_context(**values: object) -> Iterator[None]:
    """Temporarily add correlation values to every structured event."""
    current = _LOG_CONTEXT.get() or {}
    updated = {
        **current,
        **{
            key: value
            for key, value in values.items()
            if key in CONTEXT_FIELDS and value is not None
        },
    }
    token = _LOG_CONTEXT.set(updated)
    try:
        yield
    finally:
        _LOG_CONTEXT.reset(token)


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    message: str,
    *,
    exception: BaseException | None = None,
    **fields: object,
) -> None:
    """Emit one already-redacted structured event.

    Exception stacks are materialized before the record reaches handlers so
    alternate handlers cannot accidentally format an unsanitized exception.
    """
    combined = {**(_LOG_CONTEXT.get() or {}), **fields}
    if exception is not None:
        combined["exception"] = "".join(traceback.format_exception(exception))
    logger.log(
        level,
        redact_text(message),
        extra={
            "event_name": event,
            "event_fields": sanitize_for_logging(combined),
        },
    )


def text_metadata(prefix: str, value: str) -> dict[str, object]:
    """Return non-content metadata for one successful text value."""
    return {
        f"{prefix}_length": len(value),
        f"{prefix}_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
    }


def sanitize_for_logging(value: object) -> object:
    """Recursively make a value JSON-safe and remove credentials."""
    return _sanitize(value, key=None, seen=set())


def redact_text(value: str) -> str:
    """Remove registered secrets and inline authentication assignments."""
    redacted = value
    with _SECRET_LOCK:
        secrets = sorted(_SECRET_VALUES, key=len, reverse=True)
    secrets.extend(_LOCAL_SECRET_VALUES.get() or ())
    for secret in secrets:
        redacted = redacted.replace(secret, REDACTED)
    redacted = _INLINE_CREDENTIAL_PATTERN.sub(
        lambda match: f"{match.group(1)}{REDACTED}",
        redacted,
    )
    return _BEARER_PATTERN.sub(
        lambda match: f"{match.group(1)}{REDACTED}",
        redacted,
    )


class JsonFormatter(logging.Formatter):
    """Serialize every log record as one UTF-8-friendly JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        """Return a stable single-line event with all context keys present."""
        raw_fields = getattr(record, "event_fields", {})
        fields = (
            raw_fields
            if isinstance(raw_fields, Mapping)
            else {"data": raw_fields}
        )
        context = _LOG_CONTEXT.get() or {}
        document: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event_name", "log.message"),
            "message": redact_text(record.getMessage()),
        }
        for field_name in CONTEXT_FIELDS:
            document[field_name] = fields.get(
                field_name,
                context.get(field_name),
            )
        for key, value in fields.items():
            if key not in _RESERVED_FIELDS:
                document[str(key)] = value
        if record.exc_info is not None:
            document["exception"] = redact_text(
                "".join(traceback.format_exception(*record.exc_info))
            )
        elif record.exc_text:
            document["exception"] = redact_text(record.exc_text)
        return json.dumps(
            sanitize_for_logging(document),
            ensure_ascii=False,
            separators=(",", ":"),
        )


def _sanitize(
    value: object,
    *,
    key: str | None,
    seen: set[int],
) -> object:
    """Recursive implementation with cycle protection."""
    if key is not None and is_credential_field(key):
        return REDACTED
    if value is None or type(value) in {bool, int}:
        return value
    if isinstance(value, float):
        return (
            value
            if value == value and abs(value) != float("inf")
            else str(value)
        )
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, bytes):
        return redact_text(value.decode("utf-8", errors="replace"))
    if isinstance(value, UUID | Path | Enum):
        return redact_text(str(value))

    identity = id(value)
    if identity in seen:
        return "[CIRCULAR]"
    seen.add(identity)
    try:
        if isinstance(value, Mapping):
            return {
                redact_text(str(item_key)): _sanitize(
                    item_value,
                    key=str(item_key),
                    seen=seen,
                )
                for item_key, item_value in value.items()
            }
        if isinstance(value, list | tuple | set | frozenset):
            return [_sanitize(item, key=None, seen=seen) for item in value]
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return _sanitize(model_dump(mode="python"), key=key, seen=seen)
        if is_dataclass(value) and not isinstance(value, type):
            return _sanitize(asdict(value), key=key, seen=seen)
        if isinstance(value, BaseException):
            details = {
                "type": type(value).__name__,
                "message": str(value),
            }
            if hasattr(value, "__dict__"):
                details["attributes"] = vars(value)
            return _sanitize(details, key=key, seen=seen)
        if hasattr(value, "__dict__"):
            return _sanitize(vars(value), key=key, seen=seen)
        return redact_text(repr(value))
    finally:
        seen.remove(identity)


def is_credential_field(key: str) -> bool:
    """Return whether an object member conventionally stores credentials."""
    compact = re.sub(r"[^a-z0-9]", "", key.casefold())
    return (
        compact in _CREDENTIAL_KEYS
        or compact.endswith("apikey")
        or compact.endswith("token")
    )


def _secret_variants(values: Iterable[str]) -> set[str]:
    """Expand literal credentials to the URL forms that logs may contain."""
    variants: set[str] = set()
    for value in values:
        if not value:
            continue
        variants.add(value)
        variants.add(quote(value, safe=""))
        variants.add(quote_plus(value, safe=""))
    return variants
