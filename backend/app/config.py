"""Model configuration loading for the AI Town backend."""

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = REPOSITORY_ROOT / ".env"
MODEL_VARIABLES = ("BASE_URL", "API_KEY", "MODEL")


class ModelConfigError(RuntimeError):
    """Raised when required model configuration is absent or invalid."""


@dataclass(frozen=True)
class ModelSettings:
    """Validated settings used to create the Anthropic client."""

    base_url: str
    api_key: str
    model: str


def load_model_settings(
    env_file: Path = DEFAULT_ENV_FILE,
    environ: dict[str, str] | None = None,
) -> ModelSettings:
    """Load model settings with process environment values taking priority.

    Args:
        env_file: Repository-root dotenv file used for fallback values.
        environ: Optional environment mapping for tests. Defaults to
            ``os.environ``.

    Returns:
        Validated model settings.

    Raises:
        ModelConfigError: If a value is missing, blank, or BASE_URL is not a
            valid HTTP(S) URL. The error only identifies variable names.
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

    if "BASE_URL" in values and not _is_valid_http_url(values["BASE_URL"]):
        invalid_names.append("BASE_URL")

    if invalid_names:
        names = ", ".join(
            name for name in MODEL_VARIABLES if name in invalid_names
        )
        raise ModelConfigError(f"Invalid model configuration: {names}")

    return ModelSettings(
        base_url=values["BASE_URL"],
        api_key=values["API_KEY"],
        model=values["MODEL"],
    )


def _is_valid_http_url(value: str) -> bool:
    """Return whether *value* is an absolute HTTP(S) URL."""
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError:
        return False

    return parsed.scheme in {"http", "https"} and parsed.hostname is not None
