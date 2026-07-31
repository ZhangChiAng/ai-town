"""Model configuration loading for the AI Town backend."""

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = REPOSITORY_ROOT / ".env"
ANTHROPIC_VARIABLES = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
)
RESPONSES_VARIABLES = (
    "RESPONSES_BASE_URL",
    "RESPONSES_API_KEY",
    "RESPONSES_MODEL",
)
MODEL_VARIABLES = ANTHROPIC_VARIABLES + RESPONSES_VARIABLES


class ModelConfigError(RuntimeError):
    """Raised when required model configuration is absent or invalid."""


@dataclass(frozen=True)
class ModelSettings:
    """Validated settings used to create one model SDK client."""

    base_url: str
    api_key: str
    model: str


@dataclass(frozen=True)
class ConfiguredModels:
    """The two protocol configurations required by one application process."""

    anthropic: ModelSettings
    responses: ModelSettings


def load_model_settings(
    env_file: Path = DEFAULT_ENV_FILE,
    environ: dict[str, str] | None = None,
) -> ConfiguredModels:
    """Load model settings with process environment values taking priority.

    Args:
        env_file: Repository-root dotenv file used for fallback values.
        environ: Optional environment mapping for tests. Defaults to
            ``os.environ``.

    Returns:
        Validated model settings.

    Raises:
        ModelConfigError: If a value is missing or blank, a base URL is not a
            valid HTTP(S) URL, or a model conflicts with its protocol. The
            error only identifies variable names.
    """
    environment = os.environ if environ is None else environ
    file_values = dotenv_values(env_file) if env_file.is_file() else {}
    values: dict[str, str] = {}
    invalid_names: list[str] = []

    for name in MODEL_VARIABLES:
        raw_value = environment.get(name, file_values.get(name))
        if not isinstance(raw_value, str) or not raw_value.strip():
            invalid_names.append(name)
            continue
        values[name] = raw_value.strip()

    for name in ("ANTHROPIC_BASE_URL", "RESPONSES_BASE_URL"):
        if name in values and not _is_valid_http_url(values[name]):
            invalid_names.append(name)

    anthropic_model = values.get("ANTHROPIC_MODEL")
    if (
        anthropic_model is not None
        and "claude" not in anthropic_model.casefold()
    ):
        invalid_names.append("ANTHROPIC_MODEL")

    responses_model = values.get("RESPONSES_MODEL")
    if responses_model is not None and "claude" in responses_model.casefold():
        invalid_names.append("RESPONSES_MODEL")

    if invalid_names:
        names = ", ".join(
            name for name in MODEL_VARIABLES if name in invalid_names
        )
        raise ModelConfigError(f"Invalid model configuration: {names}")

    return ConfiguredModels(
        anthropic=ModelSettings(
            base_url=values["ANTHROPIC_BASE_URL"],
            api_key=values["ANTHROPIC_API_KEY"],
            model=values["ANTHROPIC_MODEL"],
        ),
        responses=ModelSettings(
            base_url=values["RESPONSES_BASE_URL"],
            api_key=values["RESPONSES_API_KEY"],
            model=values["RESPONSES_MODEL"],
        ),
    )


def _is_valid_http_url(value: str) -> bool:
    """Return whether *value* is an absolute HTTP(S) URL."""
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError:
        return False

    return parsed.scheme in {"http", "https"} and parsed.hostname is not None
