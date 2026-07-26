from typing import Literal, TypedDict

from fastapi import FastAPI


class HealthResponse(TypedDict):
    status: Literal["ok"]


app = FastAPI(title="AI Town API")


@app.get("/api/health")
def health() -> HealthResponse:
    return {"status": "ok"}
