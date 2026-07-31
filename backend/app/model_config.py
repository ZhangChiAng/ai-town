"""Load ordered, protocol-neutral model configuration."""

import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_FILE = REPOSITORY_ROOT / "models.toml"
DEFAULT_ENV_FILE = REPOSITORY_ROOT / ".env"

_MODEL_FIELDS = frozenset({"model", "protocol", "base_url", "api_key_env"})
_ENVIRONMENT_VARIABLE_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class ModelConfigError(RuntimeError):
    """Raised when model configuration is absent or invalid."""


@dataclass(frozen=True, slots=True)
class ModelSettings:
    """Validated settings used to create one protocol backend."""

    model: str
    protocol: str
    base_url: str
    api_key: str


def load_model_settings(
    models_file: Path = DEFAULT_MODELS_FILE,
    env_file: Path | None = DEFAULT_ENV_FILE,
    environ: Mapping[str, str] | None = None,
) -> tuple[ModelSettings, ...]:
    """Load ordered model settings and resolve their referenced secrets.

    Args:
        models_file: TOML file containing one or more ``[[models]]`` tables.
        env_file: Optional dotenv fallback file. A missing file is ignored.
        environ: Optional process environment override, primarily for tests.

    Returns:
        Settings in the exact order declared by the TOML file.

    Raises:
        ModelConfigError: If either input cannot be read or any configuration
            shape or value is invalid. Error messages never include configured
            values or resolved secrets.
    """
    document = _load_toml(Path(models_file))
    file_environment = _load_dotenv(env_file)
    process_environment = os.environ if environ is None else environ
    model_tables = _require_model_tables(document)

    settings: list[ModelSettings] = []
    seen_models: set[str] = set()
    for index, table in enumerate(model_tables):
        settings.append(
            _load_one_model(
                table,
                index,
                seen_models,
                process_environment,
                file_environment,
            )
        )
    return tuple(settings)


def _load_toml(models_file: Path) -> dict[str, object]:
    """Read TOML and replace parser and filesystem details with categories."""
    try:
        with models_file.open("rb") as file_handle:
            return tomllib.load(file_handle)
    except FileNotFoundError:
        raise ModelConfigError(
            "Invalid model configuration: models file is missing"
        ) from None
    except tomllib.TOMLDecodeError:
        raise ModelConfigError(
            "Invalid model configuration: malformed TOML"
        ) from None
    except OSError:
        raise ModelConfigError(
            "Invalid model configuration: models file is unreadable"
        ) from None


def _load_dotenv(env_file: Path | None) -> Mapping[str, str | None]:
    """Return dotenv fallback values without interpolating process state."""
    if env_file is None:
        return {}

    dotenv_path = Path(env_file)
    if not dotenv_path.is_file():
        return {}
    try:
        return dotenv_values(dotenv_path=dotenv_path, interpolate=False)
    except OSError:
        raise ModelConfigError(
            "Invalid model configuration: environment file is unreadable"
        ) from None


def _require_model_tables(document: dict[str, object]) -> list[object]:
    """Validate the strict top-level document shape."""
    unknown_fields = document.keys() - {"models"}
    if unknown_fields:
        raise ModelConfigError(
            "Invalid model configuration at root: unknown field"
        )
    if "models" not in document:
        raise ModelConfigError(
            "Invalid model configuration at root.models: missing field"
        )

    model_tables = document["models"]
    if type(model_tables) is not list:
        raise ModelConfigError(
            "Invalid model configuration at root.models: expected array"
        )
    if not model_tables:
        raise ModelConfigError(
            "Invalid model configuration at root.models: empty array"
        )
    return model_tables


def _load_one_model(
    raw_table: object,
    index: int,
    seen_models: set[str],
    process_environment: Mapping[str, str],
    file_environment: Mapping[str, str | None],
) -> ModelSettings:
    """Validate and resolve one model table without exposing its values."""
    location = f"models[{index}]"
    if type(raw_table) is not dict:
        raise ModelConfigError(
            f"Invalid model configuration at {location}: expected table"
        )
    table: dict[object, object] = raw_table

    unknown_fields = table.keys() - _MODEL_FIELDS
    if unknown_fields:
        raise ModelConfigError(
            f"Invalid model configuration at {location}: unknown field"
        )
    missing_fields = _MODEL_FIELDS - table.keys()
    if missing_fields:
        raise ModelConfigError(
            f"Invalid model configuration at {location}: missing field"
        )

    model = _require_string(table, "model", location).strip()
    if not model:
        raise ModelConfigError(
            f"Invalid model configuration at {location}.model: blank value"
        )
    if model in seen_models:
        raise ModelConfigError(
            f"Invalid model configuration at {location}.model: duplicate value"
        )
    seen_models.add(model)

    protocol = _require_string(table, "protocol", location)
    if not protocol.strip():
        raise ModelConfigError(
            f"Invalid model configuration at {location}.protocol: blank value"
        )

    base_url = _require_string(table, "base_url", location)
    if not _is_valid_http_url(base_url):
        raise ModelConfigError(
            f"Invalid model configuration at {location}.base_url: "
            "invalid HTTP(S) URL"
        )

    api_key_env = _require_string(table, "api_key_env", location)
    if _ENVIRONMENT_VARIABLE_PATTERN.fullmatch(api_key_env) is None:
        raise ModelConfigError(
            f"Invalid model configuration at {location}.api_key_env: "
            "invalid environment variable name"
        )

    # An explicitly present process value wins even when it is invalid.
    raw_api_key = (
        process_environment[api_key_env]
        if api_key_env in process_environment
        else file_environment.get(api_key_env)
    )
    if not isinstance(raw_api_key, str) or not raw_api_key.strip():
        raise ModelConfigError(
            f"Invalid model configuration at {location}.api_key_env: "
            "referenced secret is missing or blank"
        )

    return ModelSettings(
        model=model,
        protocol=protocol,
        base_url=base_url,
        api_key=raw_api_key.strip(),
    )


def _require_string(
    table: Mapping[object, object],
    field_name: str,
    location: str,
) -> str:
    """Return one string field or raise a value-free shape error."""
    value = table[field_name]
    if type(value) is not str:
        raise ModelConfigError(
            f"Invalid model configuration at {location}.{field_name}: "
            "expected string"
        )
    return value


def _is_valid_http_url(value: str) -> bool:
    """Return whether a value is an absolute HTTP(S) URL."""
    if any(character.isspace() for character in value):
        return False
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme.casefold() in {"http", "https"}
        and bool(parsed.netloc)
        and parsed.hostname is not None
    )
