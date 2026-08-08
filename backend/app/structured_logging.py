"""Human and JSONL logging with shared redaction and correlation context."""

import hashlib
import json
import logging
import os
import re
import sys
import threading
import traceback
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TextIO, cast
from urllib.parse import quote, quote_plus
from uuid import UUID

REDACTED = "[REDACTED]"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
JSON_LOG_PATH = REPOSITORY_ROOT / "logs" / "ai-town.jsonl"
JSON_LOG_DISPLAY_PATH = "logs/ai-town.jsonl"
JSON_LOG_MAX_BYTES = 10 * 1024 * 1024
JSON_LOG_BACKUP_COUNT = 5

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
    {
        "timestamp",
        "level",
        "logger",
        "event",
        "message",
        "json_log",
        *CONTEXT_FIELDS,
    }
)
_DOCUMENT_ATTRIBUTE = "_ai_town_log_document"
_HUMAN_COLOR_ATTRIBUTE = "_ai_town_human_color"
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
        combined.setdefault("exception_type", type(exception).__name__)
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
        return json.dumps(
            _log_document(record),
            ensure_ascii=False,
            separators=(",", ":"),
        )


class HumanFormatter(logging.Formatter):
    """Render one redacted event for quick terminal diagnosis."""

    _LEVEL_COLORS = {
        "DEBUG": "90",
        "INFO": "32",
        "WARNING": "33",
        "ERROR": "31",
        "CRITICAL": "1;31",
    }

    def format(self, record: logging.LogRecord) -> str:
        """Return a compact header followed by readable diagnostic blocks."""
        document = _log_document(record)
        use_color = bool(getattr(record, _HUMAN_COLOR_ATTRIBUTE, False))
        timestamp = _color(str(document["timestamp"]), "90", use_color)
        level_name = str(document["level"])
        level = _color(
            f"{level_name:<8}",
            self._LEVEL_COLORS.get(level_name, "37"),
            use_color,
        )
        event = _color(str(document["event"]), "36", use_color)
        message = _escape_terminal_text(str(document["message"]))
        lines = [f"{timestamp} {level} {event} — {message}"]

        correlation = [
            f"{field}={_human_scalar(document[field])}"
            for field in CONTEXT_FIELDS
            if document.get(field) is not None and document.get(field) != ""
        ]
        if document.get("json_log"):
            correlation.append(
                f"json_log={_human_scalar(document['json_log'])}"
            )
        if correlation:
            lines.append(f"  {' '.join(correlation)}")

        scalar_fields: list[str] = []
        block_fields: list[tuple[str, object]] = []
        for key, value in document.items():
            if key in _RESERVED_FIELDS or value is None:
                continue
            if _is_block_value(value):
                block_fields.append((key, value))
            else:
                scalar_fields.append(f"{key}={_human_scalar(value)}")
        if scalar_fields:
            lines.append(f"  {' '.join(scalar_fields)}")
        for key, value in block_fields:
            lines.extend(_human_block(key, value))
        return "\n".join(lines)


class HumanConsoleHandler(logging.StreamHandler):
    """Enable ANSI colors only when the configured output is an actual TTY."""

    def __init__(self, stream: TextIO | None = None) -> None:
        """Default to stdout so terminal and JSONL have separate channels."""
        super().__init__(stream or sys.stdout)

    def emit(self, record: logging.LogRecord) -> None:
        """Annotate one record with this handler's current color capability."""
        had_attribute = hasattr(record, _HUMAN_COLOR_ATTRIBUTE)
        previous = getattr(record, _HUMAN_COLOR_ATTRIBUTE, None)
        setattr(record, _HUMAN_COLOR_ATTRIBUTE, self._supports_color())
        try:
            super().emit(record)
        finally:
            if had_attribute:
                setattr(record, _HUMAN_COLOR_ATTRIBUTE, previous)
            else:
                delattr(record, _HUMAN_COLOR_ATTRIBUTE)

    def _supports_color(self) -> bool:
        """Honor NO_COLOR and tolerate streams without a usable isatty."""
        if "NO_COLOR" in os.environ:
            return False
        try:
            return bool(self.stream.isatty())
        except AttributeError, OSError:
            return False


class JsonlRotatingFileHandler(RotatingFileHandler):
    """Write structured events to the repository log with bounded backups."""

    def __init__(
        self,
        filename: str | Path | None = None,
        max_bytes: int = JSON_LOG_MAX_BYTES,
        backup_count: int = JSON_LOG_BACKUP_COUNT,
    ) -> None:
        """Create the destination directory and defer opening until needed."""
        path = Path(filename) if filename is not None else JSON_LOG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(
            path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
            delay=True,
        )


class StructuredEventFilter(logging.Filter):
    """Allow only application events emitted through :func:`log_event`."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Exclude Uvicorn and third-party records from the JSONL archive."""
        return hasattr(record, "event_name")


def _log_document(record: logging.LogRecord) -> dict[str, object]:
    """Build and cache the one sanitized document shared by both formats."""
    cached = getattr(record, _DOCUMENT_ATTRIBUTE, None)
    if isinstance(cached, dict):
        return cast(dict[str, object], cached)

    raw_fields = getattr(record, "event_fields", {})
    fields = (
        raw_fields if isinstance(raw_fields, Mapping) else {"data": raw_fields}
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
    if record.levelno >= logging.ERROR and hasattr(record, "event_name"):
        document["json_log"] = JSON_LOG_DISPLAY_PATH
    for key, value in fields.items():
        if key not in _RESERVED_FIELDS:
            document[str(key)] = value
    if record.exc_info is not None:
        document["exception"] = "".join(
            traceback.format_exception(*record.exc_info)
        )
    elif record.exc_text:
        document["exception"] = record.exc_text

    sanitized = sanitize_for_logging(document)
    assert isinstance(sanitized, dict)
    setattr(record, _DOCUMENT_ATTRIBUTE, sanitized)
    return cast(dict[str, object], sanitized)


def _human_scalar(value: object) -> str:
    """Render one scalar without losing Unicode or type distinctions."""
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _is_block_value(value: object) -> bool:
    """Return whether a field benefits from an indented terminal block."""
    return isinstance(value, Mapping | list) or (
        isinstance(value, str) and "\n" in value
    )


def _human_block(key: str, value: object) -> list[str]:
    """Render one complex value without truncating any diagnostic content."""
    if isinstance(value, str):
        rendered_lines = value.split("\n")
    else:
        rendered_lines = json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        ).splitlines()
    return [
        f"  {key}:",
        *[f"    {_escape_terminal_text(line)}" for line in rendered_lines],
    ]


def _escape_terminal_text(value: str) -> str:
    """Keep readable whitespace while neutralizing terminal controls."""
    return "".join(
        character
        if character == "\t" or ord(character) >= 32
        else f"\\x{ord(character):02x}"
        for character in value
    )


def _color(value: str, code: str, enabled: bool) -> str:
    """Wrap one terminal token in ANSI color only when explicitly enabled."""
    if not enabled:
        return value
    return f"\x1b[{code}m{value}\x1b[0m"


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
