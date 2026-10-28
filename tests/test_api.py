import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from agent.rag.contracts import RAGHealthResponse, RAGReindexResponse, RAGResponse
from agent.rag.service import RAGService
from agent.rag.tracing import RAGTraceRecord, RAGTracer
from agent.service import DeterministicAgentService
from api.main import create_app
from api.models import Task, TaskInput, overlaps


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.client = self.enterContext(TestClient(create_app(Path(directory.name) / "test.sqlite3", DeterministicAgentService())))
        self.task = {"title": "Lunch", "start_time": "2026-10-28T12:00", "end_time": "2026-10-28T13:00", "priority": "medium", "type": "fixed"}

    def test_crud_and_conflict(self) -> None:
        response = self.client.post("/tasks", json=self.task)
        self.assertEqual(response.status_code, 201)
        task_id = response.json()["id"]
        self.assertEqual(self.client.post("/tasks", json=self.task).status_code, 409)
        self.assertEqual(len(self.client.get("/tasks").json()), 1)
        self.assertEqual(self.client.put(f"/tasks/{task_id}", json={**self.task, "title": "Lunch break"}).json()["title"], "Lunch break")
        self.assertEqual(self.client.delete(f"/tasks/{task_id}").status_code, 204)
        self.assertEqual(self.client.delete(f"/tasks/{task_id}").status_code, 404)
        self.assertEqual(self.client.put("/tasks/999", json=self.task).status_code, 404)

    def test_validation(self) -> None:
        for changes in ({"title": " "}, {"end_time": self.task["start_time"]}, {"priority": "urgent"}, {"type": "unknown"}, {"start_time": "2026-10-28T12:00Z"}, {"end_time": "2026-10-29T13:00"}):
            with self.subTest(changes=changes):
                self.assertEqual(self.client.post("/tasks", json={**self.task, **changes}).status_code, 422)
        self.assertEqual(self.client.post("/planner", json={"request": " "}).status_code, 422)

    def test_planner_preserves_existing_and_does_not_save_proposals(self) -> None:
        self.client.post("/tasks", json=self.task)
        result = self.client.post("/planner", json={
            "date": "2026-10-28", "request": "Study AI-103 for 1 hour; Review PR for 30 minutes before 11 AM [high priority]; Meeting from 3 PM to 4 PM",
        })
        self.assertEqual(result.status_code, 200)
        body = result.json()
        self.assertEqual(body["warnings"], [])
        self.assertEqual(len(body["tasks"]), 3)
        self.assertEqual(body["tasks"][0]["title"], "Review PR")
        self.assertLessEqual(body["tasks"][0]["end_time"], "2026-10-28T11:00:00")
        self.assertEqual(len(self.client.get("/tasks").json()), 1)

    def test_planner_reports_conflicts_missing_duration_and_infeasible_work(self) -> None:
        self.client.post("/tasks", json=self.task)
        body = self.client.post("/planner", json={"date": "2026-10-28", "request": "Meeting from 12 PM to 1 PM; Review PR before 11 AM; Study for 12 hours"}).json()
        self.assertEqual(body["tasks"], [])
        self.assertEqual(len(body["warnings"]), 3)

    def test_planner_buffer_deadline_and_fixed_times(self) -> None:
        saved = self.client.post("/tasks", json={**self.task, "start_time": "2026-10-28T09:00", "end_time": "2026-10-28T10:00"}).json()
        result = self.client.post("/planner", json={
            "date": "2026-10-28", "request": "Review PR for 30 minutes before 11 AM; Meeting from 3 PM to 4 PM; Study for 1 hour",
        }).json()
        self.assertEqual(result["warnings"], [])
        self.assertEqual(result["tasks"][0]["start_time"], "2026-10-28T10:10:00")
        self.assertEqual(result["tasks"][0]["end_time"], "2026-10-28T10:40:00")
        meeting = next(task for task in result["tasks"] if task["title"] == "Meeting")
        self.assertEqual(meeting["start_time"], "2026-10-28T15:00:00")
        tasks = [Task.model_validate(saved), *[TaskInput.model_validate(task) for task in result["tasks"]]]
        for index, first in enumerate(tasks):
            for second in tasks[index + 1:]:
                self.assertFalse(overlaps(first.start_time, first.end_time, second))
        self.assertEqual(self.client.get("/tasks").json(), [saved])

    def test_planner_rejects_unsupported_duration_suffix(self) -> None:
        result = self.client.post("/planner", json={"request": "Study for 1 hour 30 minutes; Review for 20 minutes tomorrow"}).json()
        self.assertEqual(result["tasks"], [])
        self.assertEqual(len(result["warnings"]), 2)

    def test_planner_window_validation(self) -> None:
        for changes in ({"work_start": "18:00", "work_end": "09:00"}, {"buffer_minutes": -1}, {"work_start": "09:00:00+02:00"}):
            self.assertEqual(self.client.post("/planner", json={"request": "Study for 1 hour", **changes}).status_code, 422)


class RAGApiTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        knowledge = self.root / "knowledge"
        knowledge.mkdir()
        self.document = knowledge / "rules.md"
        self.document.write_text("# Planner\n\n## Persistence\n\nProposals are persisted only after acceptance.\n", encoding="utf-8")
        self.service = RAGService(knowledge)
        self.client = self.enterContext(TestClient(create_app(
            self.root / "test.sqlite3", DeterministicAgentService(), rag_service=self.service,
        )))

    def test_query_contract_and_no_internal_paths(self) -> None:
        response = self.client.post("/rag/query", json={"query": "Are proposals persisted?", "top_k": 3})
        self.assertEqual(response.status_code, 200)
        result = RAGResponse.model_validate(response.json())
        self.assertTrue(result.success)
        self.assertTrue(result.grounded)
        self.assertFalse(result.insufficient_context)
        self.assertIsNone(result.error)
        self.assertTrue(result.citations)
        self.assertLessEqual(len(result.retrieved_chunks), 3)
        self.assertEqual({citation.chunk_id for citation in result.citations}, {chunk.chunk_id for chunk in result.retrieved_chunks})
        self.assertNotIn(str(self.root), response.text)
        self.assertNotIn("source_path", response.text)
        self.assertNotIn("metadata", response.text)

    def test_insufficient_context_is_successful(self) -> None:
        response = self.client.post("/rag/query", json={"query": "quasar spectroscopy", "top_k": 3})
        self.assertEqual(response.status_code, 200)
        result = RAGResponse.model_validate(response.json())
        self.assertTrue(result.success)
        self.assertTrue(result.insufficient_context)
        self.assertFalse(result.grounded)
        self.assertEqual(result.citations, [])
        self.assertEqual(result.retrieved_chunks, [])

    def test_trace_correlation_id_propagates_through_api_and_failures(self) -> None:
        class Sink:
            def __init__(self) -> None:
                self.records: list[RAGTraceRecord] = []

            def emit(self, record: RAGTraceRecord) -> None:
                self.records.append(record)

        sink = Sink()
        self.service.tracer = RAGTracer(sink, correlation_id_factory=lambda: "generated-123")
        response = self.client.post(
            "/rag/query", json={"query": "proposals", "top_k": 1}, headers={"X-Correlation-ID": "caller-123"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Correlation-ID"], "caller-123")
        self.assertEqual(sink.records[-1].correlation_id, "caller-123")
        with patch.object(self.service, "retrieve", side_effect=RuntimeError("fictional-secret C:\\private\\source.md")):
            response = self.client.post("/rag/query", json={"query": "proposals", "top_k": 1})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.headers["X-Correlation-ID"], "generated-123")
        self.assertEqual(sink.records[-1].correlation_id, "generated-123")
        self.assertEqual(sink.records[-1].failure_layer, "Retrieval")
        self.assertNotIn("fictional-secret", response.text)
        self.assertNotIn("Traceback", response.text)
        self.assertNotIn("private", response.text)

    def test_invalid_correlation_id_is_rejected(self) -> None:
        for identifier in ("", "contains spaces", "C:\\private\\source.md", "x" * 129):
            response = self.client.post(
                "/rag/query", json={"query": "proposals", "top_k": 1}, headers={"X-Correlation-ID": identifier},
            )
            self.assertEqual(response.status_code, 422)

    def test_optional_filters_are_forwarded(self) -> None:
        for section, grounded in (("Persistence", True), ("Missing", False)):
            with self.subTest(section=section):
                response = self.client.post("/rag/query", json={
                    "query": "proposals", "top_k": 3, "filters": {"section": section},
                })
                self.assertEqual(response.status_code, 200)
                result = RAGResponse.model_validate(response.json())
                self.assertEqual(result.grounded, grounded)
                self.assertTrue(all(chunk.section == section for chunk in result.retrieved_chunks))

    def test_query_validation(self) -> None:
        invalid = [
            {}, {"query": " ", "top_k": 3}, {"query": "test"},
            {"query": "test", "top_k": 0}, {"query": "test", "top_k": -1},
            {"query": "test", "top_k": True}, {"query": "test", "top_k": "3"},
            {"query": "test", "top_k": 1.5}, {"query": 42, "top_k": 3},
            {"query": "test", "top_k": 3, "filters": []},
            {"query": "test", "top_k": 3, "path": "outside.md"},
        ]
        for body in invalid:
            with self.subTest(body=body):
                self.assertEqual(self.client.post("/rag/query", json=body).status_code, 422)
        self.assertEqual(self.client.post("/rag/query", content="{", headers={"Content-Type": "application/json"}).status_code, 422)

    def test_health_reindex_and_updated_documents(self) -> None:
        health = RAGHealthResponse.model_validate(self.client.get("/rag/health").json())
        self.assertTrue(health.success)
        self.assertFalse(health.indexed)
        self.assertEqual(health.indexed_chunks, 0)
        response = self.client.post("/rag/reindex")
        self.assertEqual(response.status_code, 200)
        indexed = RAGReindexResponse.model_validate(response.json())
        self.assertTrue(indexed.success)
        self.assertGreater(indexed.indexed_chunks, 0)
        health = RAGHealthResponse.model_validate(self.client.get("/rag/health").json())
        self.assertTrue(health.indexed)
        self.assertEqual(health.indexed_chunks, indexed.indexed_chunks)
        self.document.write_text("# Updated\n\n## Astronomy\n\nQuasar spectroscopy studies light.\n", encoding="utf-8")
        self.assertEqual(self.client.post("/rag/reindex").status_code, 200)
        result = self.client.post("/rag/query", json={"query": "spectroscopy", "top_k": 3}).json()
        self.assertTrue(result["grounded"])
        self.assertIn("Quasar spectroscopy", result["answer"])
        self.assertEqual(self.client.get("/tasks").json(), [])

    def test_processing_failure_is_sanitized(self) -> None:
        self.document.write_bytes(b"\xff")
        response = self.client.post("/rag/query", json={"query": "proposals", "top_k": 3})
        self.assertEqual(response.status_code, 503)
        result = RAGResponse.model_validate(response.json())
        self.assertFalse(result.success)
        self.assertEqual(result.error, "RAG_PROCESSING_ERROR")
        self.assertNotIn(str(self.root), response.text)
        self.assertNotIn("Traceback", response.text)
        response = self.client.post("/rag/reindex")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "RAG reindex unavailable. Please try again."})

    def test_unexpected_and_invalid_service_outputs_are_sanitized(self) -> None:
        for method, url, body in (
            ("query", "/rag/query", {"query": "proposals", "top_k": 3}),
            ("index_documents", "/rag/reindex", None),
            ("health", "/rag/health", None),
        ):
            for failure in (True, False):
                with self.subTest(method=method, failure=failure):
                    options = {"side_effect": RuntimeError(str(self.root))} if failure else {"return_value": {"internal_path": str(self.root)}}
                    with patch.object(self.service, method, **options):
                        response = self.client.get(url) if method == "health" else self.client.post(url, json=body)
                    self.assertEqual(response.status_code, 503)
                    self.assertNotIn(str(self.root), response.text)
                    self.assertNotIn("Traceback", response.text)

    def test_worker_threads_can_query_and_reindex(self) -> None:
        self.assertEqual(self.client.post("/rag/reindex").status_code, 200)

        def request(index: int) -> int:
            if index % 3 == 0:
                return self.client.post("/rag/reindex").status_code
            response = self.client.post("/rag/query", json={"query": "proposals", "top_k": 3})
            self.assertTrue(response.json()["grounded"])
            return response.status_code

        with ThreadPoolExecutor(max_workers=4) as executor:
            self.assertEqual(list(executor.map(request, range(12))), [200] * 12)

    def test_shutdown_closes_index(self) -> None:
        with TestClient(create_app(
            self.root / "other.sqlite3", DeterministicAgentService(), rag_service=self.service,
        )) as client:
            self.assertEqual(client.post("/rag/reindex").status_code, 200)
            self.assertTrue(self.service.health().indexed)
        self.assertFalse(self.service.health().indexed)