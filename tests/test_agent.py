import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from openai import OpenAI

from agent.service import AzureAgentService, AzureSettings, AgentServiceError, DeterministicAgentService, PlannerRequest, create_agent_service
from api.main import create_app
from api.models import Task


class AzureAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = AzureSettings(endpoint="https://example.openai.azure.com", api_key="test-key-not-a-secret", deployment="test-deployment")
        self.request = PlannerRequest(request="Help me plan tomorrow", date="2026-10-28")
        self.calls: list[httpx.Request] = []

    def intent(self, **changes):
        return {"title": "Study AI-103", "priority": "high", "type": "flexible", "start_time": None, "end_time": None, "duration_minutes": 60, "deadline": "11:00", **changes}

    def service(self, tasks=None, content=None, finish_reason="stop", refusal=None, status=200, timeout=False):
        content = content if content is not None else json.dumps({"date": "2026-10-28", "tasks": tasks if tasks is not None else [self.intent()]})

        def respond(request: httpx.Request) -> httpx.Response:
            self.calls.append(request)
            if timeout:
                raise httpx.ReadTimeout("private timeout diagnostic", request=request)
            if status != 200:
                return httpx.Response(status, json={"error": {"message": "private provider diagnostic", "type": "test_error"}})
            return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 0, "model": "test", "choices": [{"index": 0, "finish_reason": finish_reason, "message": {"role": "assistant", "content": content, "refusal": refusal}}]})

        http_client = httpx.Client(transport=httpx.MockTransport(respond))
        client = OpenAI(base_url="https://example.openai.azure.com/openai/v1/", api_key=self.settings.api_key.get_secret_value(), http_client=http_client, max_retries=0)
        self.addCleanup(client.close)
        return AzureAgentService(self.settings, client)

    def test_strict_json_request_and_application_owned_scheduling(self) -> None:
        existing = Task.model_validate({"id": 7, "title": "Private appointment", "start_time": "2026-10-28T09:00", "end_time": "2026-10-28T10:00", "priority": "high", "type": "fixed"})
        service = self.service(tasks=[self.intent(), self.intent(title="Overlapping fixed", type="fixed", start_time="09:30", end_time="10:30", duration_minutes=None, deadline=None)])
        result = service.plan(self.request, [existing])
        self.assertEqual(len(result.tasks), 0)
        self.assertEqual(len(result.warnings), 2)
        payload = json.loads(self.calls[0].content)
        self.assertEqual(payload["model"], "test-deployment")
        schema = payload["response_format"]["json_schema"]
        self.assertTrue(schema["strict"])
        self.assertFalse(schema["schema"]["additionalProperties"])
        fields = schema["schema"]["$defs"]["TaskIntent"]
        self.assertEqual(set(fields["required"]), set(fields["properties"]))
        self.assertIn("title", fields["properties"])
        self.assertNotIn("Private appointment", json.dumps(payload))
        self.assertNotIn("minimum", json.dumps(schema))
        self.assertNotIn('"format"', json.dumps(schema))

    def test_flexible_slot_buffer_and_missing_details(self) -> None:
        tasks = [self.intent(deadline=None), self.intent(title="Needs details", duration_minutes=None), self.intent(title="Lunch", type="fixed", start_time="09:00", end_time="10:00", duration_minutes=None, deadline=None)]
        result = self.service(tasks=tasks).plan(self.request, [])
        self.assertEqual(result.tasks[1].start_time.isoformat(), "2026-10-28T10:10:00")
        self.assertEqual(len(result.warnings), 1)

    def test_invalid_outputs_are_rejected(self) -> None:
        invalid = ["not json", "{}", json.dumps({"date": "2026-10-29", "tasks": []})]
        for changes in ({"title": " "}, {"duration_minutes": -1}, {"duration_minutes": "60"}, {"priority": "urgent"}, {"conflict": False}, {"start_time": "09:00"}, {"deadline": "11:00+02:00"}, {"type": "fixed", "start_time": "13:00", "end_time": "12:00", "duration_minutes": None, "deadline": None}):
            invalid.append(json.dumps({"date": "2026-10-28", "tasks": [self.intent(**changes)]}))
        for content in invalid:
            with self.subTest(content=content), self.assertRaises(AgentServiceError) as caught:
                self.service(content=content).plan(self.request, [])
            self.assertEqual(caught.exception.status_code, 502)

    def test_refused_or_truncated_output_is_rejected(self) -> None:
        for arguments in ({"refusal": "private refusal text"}, {"finish_reason": "length"}, {"finish_reason": "content_filter"}):
            with self.assertRaises(AgentServiceError):
                self.service(**arguments).plan(self.request, [])

    def test_upstream_errors_are_sanitized(self) -> None:
        for status in (401, 429, 500):
            with self.assertRaises(AgentServiceError) as caught:
                self.service(status=status).plan(self.request, [])
            self.assertEqual(caught.exception.status_code, 503)
            self.assertNotIn("private", str(caught.exception))
            self.assertNotIn("test-key-not-a-secret", str(caught.exception))

    def test_timeout_is_sanitized(self) -> None:
        with self.assertRaises(AgentServiceError) as caught:
            self.service(timeout=True).plan(self.request, [])
        self.assertEqual(caught.exception.status_code, 503)
        self.assertNotIn("private", str(caught.exception))

    def test_model_fixed_tasks_are_checked_against_each_other(self) -> None:
        first = self.intent(title="First", type="fixed", start_time="12:00", end_time="13:00", duration_minutes=None, deadline=None)
        second = {**first, "title": "Second", "start_time": "12:30", "end_time": "13:30"}
        adjacent = {**first, "title": "Adjacent", "start_time": "13:00", "end_time": "14:00"}
        result = self.service(tasks=[first, second, adjacent]).plan(self.request, [])
        self.assertEqual([task.title for task in result.tasks], ["First", "Adjacent"])
        self.assertEqual(len(result.warnings), 1)

    def test_api_error_handling_and_no_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for service, expected in ((self.service(content="invalid"), 502), (self.service(status=429), 503), (self.service(), 200)):
                with TestClient(create_app(Path(directory) / "test.sqlite3", service)) as client:
                    result = client.post("/planner", json={"request": "Plan my day", "date": "2026-10-28"})
                    self.assertEqual(result.status_code, expected)
                    self.assertEqual(client.get("/tasks").json(), [])
                    self.assertNotIn("private", result.text)

    def test_configuration_is_environment_only_and_validated(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsInstance(create_agent_service(), DeterministicAgentService)
            os.environ["PLANNER_AGENT_PROVIDER"] = "azure"
            with self.assertRaisesRegex(ValueError, "AZURE_OPENAI_API_KEY"):
                create_agent_service()
            os.environ.update(AZURE_OPENAI_ENDPOINT="https://example.openai.azure.com", AZURE_OPENAI_API_KEY="environment-test-key", AZURE_OPENAI_DEPLOYMENT="deployment")
            settings = AzureSettings.from_environment()
            self.assertNotIn("environment-test-key", repr(settings))
            service = create_agent_service()
            self.addCleanup(service.close)
            self.assertIsInstance(service, AzureAgentService)
            os.environ["AZURE_OPENAI_ENDPOINT"] = "http://example.openai.azure.com"
            with self.assertRaisesRegex(ValueError, "Invalid Azure configuration"):
                create_agent_service()