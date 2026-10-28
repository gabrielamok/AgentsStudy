import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import httpx
from fastapi import FastAPI
from openai import OpenAI

from agent.service import AzureAgentService, AzureSettings
from api.main import create_app


def provider_response(request: httpx.Request) -> httpx.Response:
    payload = json.loads(request.content)
    user_input = json.loads(payload["messages"][1]["content"])
    if "invalid-ai-output" in user_input["request"]:
        content = "invalid-provider-json: private fixture diagnostic"
    else:
        content = json.dumps({
            "date": user_input["selected_date"],
            "tasks": [
                {"title": "AI-103 study", "priority": "high", "type": "flexible",
                 "start_time": None, "end_time": None, "duration_minutes": 60, "deadline": "11:00"},
                {"title": "Lunch", "priority": "medium", "type": "fixed",
                 "start_time": "12:00", "end_time": "13:00", "duration_minutes": None, "deadline": None},
            ],
        })
    return httpx.Response(200, json={
        "id": "e2e-completion", "object": "chat.completion", "created": 0,
        "model": "e2e-model", "choices": [{"index": 0, "finish_reason": "stop",
        "message": {"role": "assistant", "content": content}}],
    })


def create_e2e_app() -> FastAPI:
    database_path = Path(os.environ["PLANNER_E2E_DB_PATH"])
    settings = AzureSettings(
        endpoint="https://e2e.invalid", api_key="e2e-placeholder-not-a-secret", deployment="e2e-model",
    )
    client = OpenAI(
        base_url="https://e2e.invalid/openai/v1/", api_key=settings.api_key.get_secret_value(),
        max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(provider_response)),
    )
    app = create_app(database_path, AzureAgentService(settings, client))
    application_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        try:
            async with application_lifespan(application):
                yield
        finally:
            client.close()

    app.router.lifespan_context = lifespan
    return app