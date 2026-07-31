"""Reusable pytest contract suite for protocol-specific model backends."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import pytest

from app.model_backends import (
    JsonObject,
    ModelBackend,
    ModelConversation,
    ModelGeneration,
)


@dataclass(frozen=True, slots=True)
class BackendContractCase:
    """One isolated backend fixture consumed by the shared test suite."""

    backend: ModelBackend
    expected_model: str
    conversation: ModelConversation
    expected_payload: JsonObject
    expected_generation: ModelGeneration
    upstream_call_count: Callable[[], int]
    close_call_count: Callable[[], int]
    forbidden_payload_values: tuple[str, ...] = ()


class BackendContractFactory(Protocol):
    """Build a fresh backend contract fixture for each inherited test."""

    def __call__(self) -> BackendContractCase:
        """Return one isolated backend and its observable test doubles."""


class BackendContractTests:
    """Reusable behavioral tests inherited by every adapter test class."""

    contract_case_factory: BackendContractFactory

    @pytest.fixture
    def contract_case(self) -> BackendContractCase:
        """Create a fresh contract case supplied by the concrete test class."""
        return self.contract_case_factory()

    def test_model_identity_is_exact(
        self,
        contract_case: BackendContractCase,
    ) -> None:
        """Backends expose their case-sensitive configured model name."""
        assert contract_case.backend.model == contract_case.expected_model

    def test_prepare_is_pure_deterministic_and_credential_free(
        self,
        contract_case: BackendContractCase,
    ) -> None:
        """Preparing performs no I/O and yields a safe preview payload."""
        calls_before = contract_case.upstream_call_count()

        first = contract_case.backend.prepare(contract_case.conversation)
        second = contract_case.backend.prepare(contract_case.conversation)

        assert contract_case.upstream_call_count() == calls_before
        assert first.payload == contract_case.expected_payload
        assert second.payload == contract_case.expected_payload
        assert (
            json.loads(json.dumps(first.payload, allow_nan=False))
            == first.payload
        )
        serialized = json.dumps(first.payload, ensure_ascii=False)
        assert all(
            forbidden not in serialized
            for forbidden in contract_case.forbidden_payload_values
        )

    def test_generate_calls_upstream_once_for_prepared_request(
        self,
        contract_case: BackendContractCase,
    ) -> None:
        """Generating consumes the passed request in one upstream call."""
        prepared = contract_case.backend.prepare(contract_case.conversation)
        calls_before = contract_case.upstream_call_count()

        result = contract_case.backend.generate(prepared)

        assert contract_case.upstream_call_count() == calls_before + 1
        assert result == contract_case.expected_generation

    def test_close_releases_owned_resource_once(
        self,
        contract_case: BackendContractCase,
    ) -> None:
        """One lifecycle close delegates once to the owned resource."""
        closes_before = contract_case.close_call_count()

        contract_case.backend.close()

        assert contract_case.close_call_count() == closes_before + 1
