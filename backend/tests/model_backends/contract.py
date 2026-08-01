"""Reusable pytest contract suite for protocol-specific model backends."""

import asyncio
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol

import pytest

from app.model_backends.contracts import (
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
    expected_generation: ModelGeneration
    upstream_call_count: Callable[[], int]
    close_call_count: Callable[[], int]
    forbidden_snapshot_values: tuple[str, ...] = ()


class BackendContractFactory(Protocol):
    """Build a fresh backend contract fixture for each inherited test."""

    def __call__(self) -> BackendContractCase:
        """Return one isolated backend and its observable test doubles."""


class BackendContractTests:
    """Reusable behavioral tests inherited by every adapter test class."""

    contract_case_factory: BackendContractFactory

    @pytest.fixture
    def contract_case(self) -> Iterator[BackendContractCase]:
        """Create and always close one fresh concrete contract case."""
        contract_case = self.contract_case_factory()
        try:
            yield contract_case
        finally:
            asyncio.run(contract_case.backend.aclose())

    def test_model_identity_is_exact(
        self,
        contract_case: BackendContractCase,
    ) -> None:
        """Backends expose their case-sensitive configured model name."""
        assert contract_case.backend.model == contract_case.expected_model

    def test_generate_calls_upstream_once_for_conversation(
        self,
        contract_case: BackendContractCase,
    ) -> None:
        """Generating maps the neutral conversation in one upstream call."""
        calls_before = contract_case.upstream_call_count()

        async def generate_and_close() -> ModelGeneration:
            try:
                return await contract_case.backend.generate(
                    contract_case.conversation
                )
            finally:
                await contract_case.backend.aclose()

        result = asyncio.run(generate_and_close())

        assert contract_case.upstream_call_count() == calls_before + 1
        assert result == contract_case.expected_generation

    def test_generation_snapshot_is_json_safe_and_credential_free(
        self,
        contract_case: BackendContractCase,
    ) -> None:
        """Successful calls expose only a portable, credential-free body."""

        async def generate_and_close() -> ModelGeneration:
            try:
                return await contract_case.backend.generate(
                    contract_case.conversation
                )
            finally:
                await contract_case.backend.aclose()

        result = asyncio.run(generate_and_close())

        assert (
            json.loads(json.dumps(result.request_snapshot, allow_nan=False))
            == result.request_snapshot
        )
        serialized = json.dumps(result.request_snapshot, ensure_ascii=False)
        assert all(
            forbidden not in serialized
            for forbidden in contract_case.forbidden_snapshot_values
        )

    def test_close_releases_owned_resource_once(
        self,
        contract_case: BackendContractCase,
    ) -> None:
        """Repeated lifecycle close delegates once to the owned resource."""
        closes_before = contract_case.close_call_count()

        async def close_twice() -> None:
            await contract_case.backend.aclose()
            await contract_case.backend.aclose()

        asyncio.run(close_twice())

        assert contract_case.close_call_count() == closes_before + 1
