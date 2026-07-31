"""Static tests for model-backend architecture boundaries."""

import ast
import re
from collections.abc import Iterable
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = BACKEND_ROOT.parent
APP_ROOT = BACKEND_ROOT / "app"
MODEL_BACKENDS_ROOT = APP_ROOT / "model_backends"
FRONTEND_ROOT = REPOSITORY_ROOT / "frontend" / "src"

ANTHROPIC_ADAPTER = MODEL_BACKENDS_ROOT / "anthropic_messages.py"
RESPONSES_ADAPTER = MODEL_BACKENDS_ROOT / "openai_responses.py"
ADAPTER_PATHS = (ANTHROPIC_ADAPTER, RESPONSES_ADAPTER)
SDK_OWNERS = {
    "anthropic": ANTHROPIC_ADAPTER,
    "openai": RESPONSES_ADAPTER,
}
PROVIDER_MARKERS = (
    "anthropic",
    "claude",
    "gemini",
    "openai",
    "responses",
    "gpt",
)


def _parse(path: Path) -> ast.Module:
    """Parse one Python module with its filename in syntax diagnostics."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _import_targets(tree: ast.AST) -> Iterable[tuple[str, str | None]]:
    """Yield imported module targets and explicitly imported symbols."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, None
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                yield module, alias.name
                if module:
                    yield f"{module}.{alias.name}", alias.name


def _identifiers(node: ast.AST) -> set[str]:
    """Collect case-folded names and attributes below one AST node."""
    values: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            values.add(child.id.casefold())
        elif isinstance(child, ast.Attribute):
            values.add(child.attr.casefold())
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            values.add(child.name.casefold())
    return values


def _strings(node: ast.AST) -> set[str]:
    """Collect case-folded string literals below one AST node."""
    return {
        child.value.casefold()
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    }


def _condition_expressions(tree: ast.AST) -> Iterable[ast.AST]:
    """Yield branch expressions that could route by model identity."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.IfExp, ast.While)):
            yield node.test
        elif isinstance(node, ast.comprehension):
            yield from node.ifs
        elif isinstance(node, ast.Match):
            yield node


def test_provider_sdk_imports_stay_in_their_concrete_adapters() -> None:
    """Neither SDK may cross from its adapter into neutral application code."""
    for path in APP_ROOT.rglob("*.py"):
        for target, _symbol in _import_targets(_parse(path)):
            sdk_name = target.partition(".")[0]
            if sdk_name in SDK_OWNERS:
                assert path == SDK_OWNERS[sdk_name], (
                    f"{sdk_name} SDK import escaped its adapter: {path}"
                )


def test_concrete_adapters_do_not_depend_on_business_or_storage_layers() -> (
    None
):
    """Adapters point inward only to frozen model-backend contracts."""
    forbidden_modules = {
        "app.draft_workflow",
        "app.main",
        "app.models",
        "app.storage",
        "draft_workflow",
        "main",
        "models",
        "storage",
    }
    for path in ADAPTER_PATHS:
        imports = tuple(_import_targets(_parse(path)))
        for target, symbol in imports:
            assert not any(
                target == module or target.startswith(f"{module}.")
                for module in forbidden_modules
            ), f"adapter imports forbidden module {target}: {path}"
            assert symbol != "Scene", f"adapter imports Scene: {path}"


def test_draft_workflow_has_no_provider_routing_or_wire_vocabulary() -> None:
    """The draft workflow consumes only protocol-neutral ports and values."""
    tree = _parse(APP_ROOT / "draft_workflow.py")
    backend_imports = (
        target
        for target, _symbol in _import_targets(tree)
        if target == "app.model_backends"
        or target.startswith("app.model_backends.")
    )
    assert all(
        target == "app.model_backends.contracts"
        or target.startswith("app.model_backends.contracts.")
        for target in backend_imports
    ), "draft_workflow must depend directly on the neutral contracts"
    vocabulary = _identifiers(tree) | _strings(tree)

    for value in vocabulary:
        assert not any(marker in value for marker in PROVIDER_MARKERS), (
            f"provider vocabulary leaked into draft_workflow: {value}"
        )
    for wire_field in {"cache_control", "input_text", "instructions"}:
        assert wire_field not in vocabulary, (
            f"wire field leaked into draft_workflow: {wire_field}"
        )


def test_business_modules_do_not_guess_protocol_from_model_names() -> None:
    """Main and scene-domain branches never infer transport from model text."""
    inference_methods = {"casefold", "endswith", "lower", "startswith"}
    for filename in ("main.py", "models.py"):
        path = APP_ROOT / filename
        tree = _parse(path)
        for condition in _condition_expressions(tree):
            identifiers = _identifiers(condition)
            strings = _strings(condition)
            references_model = any("model" in name for name in identifiers)
            references_provider = any(
                marker in value
                for marker in PROVIDER_MARKERS
                for value in identifiers | strings
            )
            assert not (references_model and references_provider), (
                f"model-name protocol branch in {path}:{condition.lineno}"
            )
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in inference_methods:
                continue
            assert not any(
                "model" in name for name in _identifiers(node.func.value)
            ), f"model-name inference in {path}:{node.lineno}"


def test_removed_legacy_model_modules_do_not_return() -> None:
    """The obsolete protocol-branching implementation has no fallback path."""
    assert not (APP_ROOT / "config.py").exists()
    assert not (APP_ROOT / "drafting.py").exists()


def test_frontend_uses_neutral_model_options_and_preview_context() -> None:
    """Frontend readable views use context, never provider request fields."""
    paths = {
        name: FRONTEND_ROOT / name for name in ("App.vue", "api.ts", "types.ts")
    }
    sources = {
        name: path.read_text(encoding="utf-8") for name, path in paths.items()
    }

    for name, source in sources.items():
        assert "ModelProtocol" not in source, f"protocol enum in {name}"
        assert re.search(r"\.protocol\b|\bprotocol\s*:", source) is None, (
            f"protocol exposed in {name}"
        )
        for marker in PROVIDER_MARKERS:
            assert re.search(rf"\b{marker}\b", source, re.IGNORECASE) is None, (
                f"provider name {marker} exposed in {name}"
            )

    app_source = sources["App.vue"]
    for wire_field in ("cache_control", "input_text", "instructions"):
        assert wire_field not in app_source, (
            f"App.vue parses provider wire field {wire_field}"
        )
    assert re.search(r"selectedPreview\s*\.\s*context", app_source)
    assert re.search(
        r"prettyJson\(\s*selectedPreview\s*\.\s*request\s*\)",
        app_source,
    )

    api_source = sources["api.ts"]
    assert re.search(r"Array\.isArray\(\s*value\.context\s*\)", api_source)
    assert re.search(r"value\.context\.every\(", api_source)

    types_source = sources["types.ts"]
    assert re.search(
        r"context\s*:\s*ModelRequestContextItem\[\]",
        types_source,
    )
