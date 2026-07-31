"""Tests for ordered TOML model configuration loading."""

from pathlib import Path

import pytest

from app.model_backends.contracts import ModelBackendSettings
from app.model_config import (
    ModelConfigError,
    ModelSettings,
    load_model_settings,
)


def _write_models_file(path: Path, content: str) -> Path:
    """Write one isolated TOML fixture and return its path."""
    path.write_text(content, encoding="utf-8")
    return path


def _two_model_toml() -> str:
    """Return valid configuration with two protocols in a stable order."""
    return """
[[models]]
model = "Vendor/Case-Sensitive"
protocol = "first_protocol"
base_url = "https://first.example/api"
api_key_env = "FIRST_API_KEY"

[[models]]
model = "vendor/case-sensitive"
protocol = "second_protocol"
base_url = "http://localhost:8080/v1"
api_key_env = "SECOND_API_KEY"
"""


def test_loads_one_or_more_models_in_toml_order(tmp_path: Path) -> None:
    """Model identity stays case-sensitive and follows declaration order."""
    models_file = _write_models_file(
        tmp_path / "models.toml", _two_model_toml()
    )

    settings = load_model_settings(
        models_file,
        tmp_path / "missing.env",
        {
            "FIRST_API_KEY": " first-secret ",
            "SECOND_API_KEY": "second-secret",
        },
    )

    assert settings == (
        ModelSettings(
            model="Vendor/Case-Sensitive",
            protocol="first_protocol",
            base_url="https://first.example/api",
            api_key="first-secret",
        ),
        ModelSettings(
            model="vendor/case-sensitive",
            protocol="second_protocol",
            base_url="http://localhost:8080/v1",
            api_key="second-secret",
        ),
    )
    assert all(isinstance(item, ModelBackendSettings) for item in settings)


def test_process_environment_overrides_dotenv(tmp_path: Path) -> None:
    """An explicitly present process value wins over the dotenv fallback."""
    models_file = _write_models_file(
        tmp_path / "models.toml",
        """
[[models]]
model = "model"
protocol = "protocol"
base_url = "https://example.test"
api_key_env = "MODEL_API_KEY"
""",
    )
    env_file = tmp_path / ".env"
    env_file.write_text("MODEL_API_KEY=file-secret\n", encoding="utf-8")

    settings = load_model_settings(
        models_file,
        env_file,
        {"MODEL_API_KEY": "process-secret"},
    )

    assert settings[0].api_key == "process-secret"


def test_dotenv_is_a_fallback_and_is_optional(tmp_path: Path) -> None:
    """Dotenv supplies absent values, while a missing file needs no handling."""
    models_file = _write_models_file(
        tmp_path / "models.toml",
        """
[[models]]
model = "first"
protocol = "protocol"
base_url = "https://first.example"
api_key_env = "FIRST_KEY"

[[models]]
model = "second"
protocol = "protocol"
base_url = "https://second.example"
api_key_env = "SECOND_KEY"
""",
    )
    env_file = tmp_path / ".env"
    env_file.write_text('FIRST_KEY="file-secret"\n', encoding="utf-8")

    settings = load_model_settings(
        models_file,
        env_file,
        {"SECOND_KEY": "process-secret"},
    )
    without_file = load_model_settings(
        _write_models_file(
            tmp_path / "one-model.toml",
            """
[[models]]
model = "only"
protocol = "protocol"
base_url = "https://only.example"
api_key_env = "ONLY_KEY"
""",
        ),
        tmp_path / "missing.env",
        {"ONLY_KEY": "only-secret"},
    )

    assert [item.api_key for item in settings] == [
        "file-secret",
        "process-secret",
    ]
    assert without_file[0].api_key == "only-secret"


@pytest.mark.parametrize(
    "toml_content",
    [
        "models = []",
        "[models]\nmodel = 'wrong-shape'",
        "models = ['wrong-item-shape']",
        "unrelated = true",
        "[[models]]\nmodel = 'missing-fields'",
        "[[models]\nmodel = 'malformed'",
    ],
)
def test_rejects_empty_unknown_and_invalid_toml_shapes(
    tmp_path: Path,
    toml_content: str,
) -> None:
    """Only a non-empty array of strict model tables is accepted."""
    models_file = _write_models_file(tmp_path / "models.toml", toml_content)

    with pytest.raises(ModelConfigError):
        load_model_settings(models_file, None, {})


@pytest.mark.parametrize(
    "extra_line",
    [
        'unexpected = "do-not-report-this-value"',
        'model = "duplicate-field-value"',
    ],
)
def test_rejects_unknown_or_duplicate_model_fields(
    tmp_path: Path,
    extra_line: str,
) -> None:
    """Tables cannot add fields or redefine one of the four known fields."""
    models_file = _write_models_file(
        tmp_path / "models.toml",
        f"""
[[models]]
model = "model"
protocol = "protocol"
base_url = "https://example.test"
api_key_env = "KEY"
{extra_line}
""",
    )

    with pytest.raises(ModelConfigError) as caught:
        load_model_settings(models_file, None, {"KEY": "secret"})

    assert "do-not-report-this-value" not in str(caught.value)
    assert "duplicate-field-value" not in str(caught.value)


def test_rejects_duplicate_models_after_trimming(tmp_path: Path) -> None:
    """Whitespace normalization cannot make two model keys ambiguous."""
    models_file = _write_models_file(
        tmp_path / "models.toml",
        """
[[models]]
model = "ExactModel"
protocol = "protocol"
base_url = "https://first.example"
api_key_env = "FIRST_KEY"

[[models]]
model = "  ExactModel  "
protocol = "protocol"
base_url = "https://second.example"
api_key_env = "SECOND_KEY"
""",
    )

    with pytest.raises(ModelConfigError, match="duplicate value") as caught:
        load_model_settings(
            models_file,
            None,
            {"FIRST_KEY": "first-secret", "SECOND_KEY": "second-secret"},
        )

    assert "ExactModel" not in str(caught.value)


@pytest.mark.parametrize(
    "base_url",
    [
        "not-a-url",
        "ftp://example.test/api",
        "https:///missing-host",
        "https://example.test:invalid-port",
        "https://example.test/has a space",
    ],
)
def test_rejects_invalid_absolute_http_urls(
    tmp_path: Path,
    base_url: str,
) -> None:
    """Every backend root must be an absolute HTTP(S) URL."""
    models_file = _write_models_file(
        tmp_path / "models.toml",
        f"""
[[models]]
model = "model"
protocol = "protocol"
base_url = "{base_url}"
api_key_env = "KEY"
""",
    )

    with pytest.raises(ModelConfigError) as caught:
        load_model_settings(models_file, None, {"KEY": "secret"})

    assert base_url not in str(caught.value)
    assert "base_url" in str(caught.value)


@pytest.mark.parametrize(
    "variable_name",
    ["", "1KEY", "KEY-WITH-DASH", "KEY WITH SPACE", "KÉY"],
)
def test_rejects_invalid_api_key_environment_names(
    tmp_path: Path,
    variable_name: str,
) -> None:
    """Secret references use portable environment variable identifiers."""
    models_file = _write_models_file(
        tmp_path / "models.toml",
        f"""
[[models]]
model = "model"
protocol = "protocol"
base_url = "https://example.test"
api_key_env = "{variable_name}"
""",
    )

    with pytest.raises(ModelConfigError) as caught:
        load_model_settings(models_file, None, {})

    if variable_name:
        assert variable_name not in str(caught.value)
    assert "api_key_env" in str(caught.value)


def test_blank_process_secret_does_not_fall_back_to_dotenv(
    tmp_path: Path,
) -> None:
    """Process precedence is deterministic even for an invalid blank value."""
    models_file = _write_models_file(
        tmp_path / "models.toml",
        """
[[models]]
model = "model"
protocol = "protocol"
base_url = "https://example.test"
api_key_env = "KEY"
""",
    )
    env_file = tmp_path / ".env"
    env_file.write_text("KEY=file-secret\n", encoding="utf-8")

    with pytest.raises(ModelConfigError, match="missing or blank") as caught:
        load_model_settings(models_file, env_file, {"KEY": "  "})

    assert "file-secret" not in str(caught.value)


def test_missing_secret_and_legacy_variables_do_not_leak_or_fallback(
    tmp_path: Path,
) -> None:
    """The former six-variable configuration is never read implicitly."""
    models_file = _write_models_file(
        tmp_path / "models.toml",
        """
[[models]]
model = "new-model"
protocol = "new-protocol"
base_url = "https://new.example"
api_key_env = "AI_TOWN_NEW_KEY"
""",
    )
    legacy_environment = {
        "ANTHROPIC_BASE_URL": "https://legacy-anthropic.example",
        "ANTHROPIC_API_KEY": "legacy-anthropic-secret",
        "ANTHROPIC_MODEL": "legacy-anthropic-model",
        "RESPONSES_BASE_URL": "https://legacy-responses.example",
        "RESPONSES_API_KEY": "legacy-responses-secret",
        "RESPONSES_MODEL": "legacy-responses-model",
    }

    with pytest.raises(ModelConfigError) as caught:
        load_model_settings(models_file, None, legacy_environment)

    error = str(caught.value)
    assert "AI_TOWN_NEW_KEY" not in error
    assert all(value not in error for value in legacy_environment.values())


def test_wrong_field_types_are_rejected_without_value_leaks(
    tmp_path: Path,
) -> None:
    """TOML scalar types are strict and diagnostics expose only field names."""
    models_file = _write_models_file(
        tmp_path / "models.toml",
        """
[[models]]
model = 12345
protocol = "protocol"
base_url = "https://example.test"
api_key_env = "KEY"
""",
    )

    with pytest.raises(ModelConfigError) as caught:
        load_model_settings(models_file, None, {"KEY": "secret-value"})

    error = str(caught.value)
    assert "model" in error
    assert "12345" not in error
    assert "secret-value" not in error
