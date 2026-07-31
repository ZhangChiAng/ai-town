"""Contract tests using a deliberately provider-unlike fake backend."""

from dataclasses import dataclass

import pytest

from app.model_backends import (
    BackendFactory,
    ModelBackend,
    ModelBackendSettings,
    ModelConversation,
    ModelGeneration,
    ModelReasoning,
    ModelTurn,
    ModelUsage,
    PreparedModelRequest,
)
from tests.model_backends.contract import (
    BackendContractCase,
    BackendContractTests,
)

MODEL = "fake/case-Sensitive"
SECRET = "never-place-this-secret-in-a-payload"
CONVERSATION = ModelConversation(
    system_prompt="SYSTEM exact",
    turns=(
        ModelTurn(input="old user", output="old assistant"),
        ModelTurn(input="older user", output="older assistant"),
    ),
    current_input="current user",
)
EXPECTED_PAYLOAD = {
    "engine": MODEL,
    "envelope": {
        "rules": "SYSTEM exact",
        "exchanges": [
            ["old user", "old assistant"],
            ["older user", "older assistant"],
        ],
        "next": "current user",
    },
}
EXPECTED_GENERATION = ModelGeneration(
    content="fake output",
    reasoning=(ModelReasoning(type="thinking", text="safe thought"),),
    usage=ModelUsage(
        input_tokens=11,
        output_tokens=4,
        cache_creation_input_tokens=2,
        cache_read_input_tokens=3,
    ),
)


@dataclass(frozen=True, slots=True)
class FakeSettings:
    """Resolved factory settings used only by the fake backend test."""

    model: str
    base_url: str
    api_key: str


class FakeBackend:
    """Small fake proving the port is independent of provider wire shapes."""

    def __init__(self, settings: ModelBackendSettings) -> None:
        """Capture only initialization data needed after client creation."""
        self._model = settings.model
        self.upstream_requests: list[PreparedModelRequest] = []
        self.close_calls = 0

    @property
    def model(self) -> str:
        """Return the configured model name exactly."""
        return self._model

    def prepare(
        self,
        conversation: ModelConversation,
    ) -> PreparedModelRequest:
        """Map neutral conversation fields to an unrelated fake envelope."""
        return PreparedModelRequest(
            payload={
                "engine": self.model,
                "envelope": {
                    "rules": conversation.system_prompt,
                    "exchanges": [
                        [turn.input, turn.output] for turn in conversation.turns
                    ],
                    "next": conversation.current_input,
                },
            }
        )

    def generate(
        self,
        prepared: PreparedModelRequest,
    ) -> ModelGeneration:
        """Represent one and only one upstream interaction."""
        self.upstream_requests.append(prepared)
        return EXPECTED_GENERATION

    def close(self) -> None:
        """Record one resource-release request."""
        self.close_calls += 1


def fake_backend_factory(settings: ModelBackendSettings) -> ModelBackend:
    """Create one fake backend through the frozen factory signature."""
    return FakeBackend(settings)


def build_contract_case() -> BackendContractCase:
    """Return one isolated fake implementation for the shared suite."""
    backend = FakeBackend(
        FakeSettings(
            model=MODEL,
            base_url="https://fake.example/api",
            api_key=SECRET,
        )
    )
    return BackendContractCase(
        backend=backend,
        expected_model=MODEL,
        conversation=CONVERSATION,
        expected_payload=EXPECTED_PAYLOAD,
        expected_generation=EXPECTED_GENERATION,
        upstream_call_count=lambda: len(backend.upstream_requests),
        close_call_count=lambda: backend.close_calls,
        forbidden_payload_values=(SECRET,),
    )


class TestFakeBackendContract(BackendContractTests):
    """Apply every reusable backend check to the fake protocol."""

    contract_case_factory = staticmethod(build_contract_case)


def test_factory_and_backend_protocols_are_structurally_satisfied() -> None:
    """Plain callables and classes can implement the frozen ports."""
    factory: BackendFactory = fake_backend_factory
    backend = factory(
        FakeSettings(
            model=MODEL,
            base_url="https://fake.example/api",
            api_key=SECRET,
        )
    )

    assert isinstance(backend, ModelBackend)
    assert isinstance(
        FakeSettings(MODEL, "https://fake.example/api", SECRET),
        ModelBackendSettings,
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"bad": float("nan")},
        {"bad": ("tuple",)},
        {1: "non-string key"},
    ],
)
def test_prepared_request_rejects_non_json_safe_values(
    payload: object,
) -> None:
    """The request wrapper rejects values JSON would coerce or distort."""
    with pytest.raises(ValueError, match="payload"):
        PreparedModelRequest(payload=payload)  # type: ignore[arg-type]
