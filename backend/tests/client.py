"""Thread-free ASGI test client for the Python 3.14 test environment."""

import asyncio
from typing import Any

import httpx
from fastapi import FastAPI


class TestClient:
    """Small synchronous facade over httpx's asynchronous ASGI transport."""

    __test__ = False

    def __init__(self, application: FastAPI) -> None:
        """Store the application exercised by each isolated request."""
        self._application = application

    def get(
        self,
        path: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Send a GET request."""
        return self._request("GET", path, headers=headers)

    def post(
        self,
        path: str,
        *,
        json: Any | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Send a POST request."""
        return self._request("POST", path, json=json, headers=headers)

    def put(
        self,
        path: str,
        *,
        json: Any | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Send a PUT request."""
        return self._request("PUT", path, json=json, headers=headers)

    def delete(
        self,
        path: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Send a DELETE request."""
        return self._request("DELETE", path, headers=headers)

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        async def send() -> httpx.Response:
            transport = httpx.ASGITransport(app=self._application)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.request(
                    method,
                    path,
                    json=json,
                    headers=headers,
                )

        return asyncio.run(send())
