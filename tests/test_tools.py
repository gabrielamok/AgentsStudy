import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from agent.contracts import AgentResponse, Intent
from agent.tools import create_task, delete_task, find_conflicts, list_tasks, update_task
from api.repository import TaskRepository


class ToolTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.repository = TaskRepository(Path(directory.name) / "tasks.sqlite3")
        self.repository.initialize()
        self.task = {"title": "Review PR", "start_time": "2026-10-28T09:00:00", "end_time": "2026-10-28T10:00:00", "priority": "high", "type": "flexible"}

    def create(self, **changes):
        result = create_task({**self.task, **changes}, repository=self.repository)
        self.assertTrue(result.success, result.model_dump())
        return result.data

    def test_create_and_list_structured_results(self) -> None:
        result = create_task(self.task, repository=self.repository)
        self.assertTrue(result.success)
        self.assertEqual(result.intent, Intent.CREATE_TASK)
        self.assertEqual(result.tool_called, "create_task")
        self.assertEqual(AgentResponse.model_validate_json(result.model_dump_json()), result)
        self.assertEqual(list_tasks({}, repository=self.repository).data, [result.data])
        self.assertIsNone(result.error)

    def test_invalid_create_never_calls_repository(self) -> None:
        invalid = [None, [], {}, {**self.task, "title": " "}, {**self.task, "title": " Review PR "},
                   {**self.task, "title": 123}, {**self.task, "priority": "urgent"},
                   {**self.task, "type": "unknown"}, {**self.task, "id": 1},
                   {**self.task, "end_time": self.task["start_time"]},
                   {**self.task, "start_time": 12345}, {**self.task, "start_time": "2026-10-28"},
                   {**self.task, "start_time": "2026-10-28T09:00:00Z"},
                   {**self.task, "end_time": "2026-10-29T10:00:00"}]
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                repository = Mock(spec=TaskRepository)
                result = create_task(arguments, repository=repository)
                self.assertFalse(result.success)
                self.assertEqual(result.error, "VALIDATION_ERROR")
                repository.save.assert_not_called()
        self.assertEqual(self.repository.list(), [])

    def test_each_create_parameter_is_required(self) -> None:
        for field in self.task:
            with self.subTest(field=field):
                arguments = {key: value for key, value in self.task.items() if key != field}
                self.assertEqual(create_task(arguments, repository=self.repository).error, "VALIDATION_ERROR")
        self.assertEqual(self.repository.list(), [])

    def test_conflicting_create_does_not_persist_and_adjacent_does(self) -> None:
        original = self.create()
        result = create_task({**self.task, "start_time": "2026-10-28T09:30:00"}, repository=self.repository)
        self.assertEqual(result.error, "SCHEDULE_CONFLICT")
        self.assertEqual(result.data["conflicts"], [original])
        self.assertEqual(len(self.repository.list()), 1)
        self.create(start_time="2026-10-28T10:00:00", end_time="2026-10-28T11:00:00")
        self.assertEqual(len(self.repository.list()), 2)

    def test_update_validates_and_replaces_identified_task(self) -> None:
        original = self.create()
        arguments = {**self.task, "task_id": original["id"], "title": "Updated review"}
        result = update_task(arguments, repository=self.repository)
        self.assertTrue(result.success)
        self.assertEqual(result.data["id"], original["id"])
        self.assertEqual(result.data["title"], "Updated review")
        self.assertEqual(len(self.repository.list()), 1)
        for changes in ({"end_time": self.task["start_time"]}, {"task_id": "1"}, {"task_id": True}):
            self.assertEqual(update_task({**arguments, **changes}, repository=self.repository).error, "VALIDATION_ERROR")
        self.assertEqual(self.repository.list()[0].title, "Updated review")
        self.assertEqual(update_task({"task_id": original["id"], "title": "Partial"}, repository=self.repository).error, "VALIDATION_ERROR")
        self.assertEqual(update_task({**arguments, "task_id": 999}, repository=self.repository).error, "TASK_NOT_FOUND")

    def test_conflicting_update_preserves_original(self) -> None:
        first = self.create()
        second = self.create(start_time="2026-10-28T10:00:00", end_time="2026-10-28T11:00:00")
        result = update_task({**self.task, "task_id": second["id"]}, repository=self.repository)
        self.assertEqual(result.error, "SCHEDULE_CONFLICT")
        self.assertEqual(list_tasks({}, repository=self.repository).data, [first, second])

    def test_delete_requires_valid_id_and_reports_missing_tasks(self) -> None:
        original = self.create()
        for arguments in ({}, {"task_id": 0}, {"task_id": True}, {"task_id": "1"}, {"task_id": original["id"], "extra": True}):
            self.assertEqual(delete_task(arguments, repository=self.repository).error, "VALIDATION_ERROR")
        self.assertEqual(len(self.repository.list()), 1)
        result = delete_task({"task_id": original["id"]}, repository=self.repository)
        self.assertTrue(result.success)
        self.assertEqual(result.data, {"task_id": original["id"]})
        self.assertEqual(self.repository.list(), [])
        self.assertEqual(delete_task({"task_id": original["id"]}, repository=self.repository).error, "TASK_NOT_FOUND")

    def test_list_filters_explicit_date_without_mutation(self) -> None:
        first = self.create()
        self.create(start_time="2026-10-29T09:00:00", end_time="2026-10-29T10:00:00")
        self.assertEqual(list_tasks({"date": "2026-10-28"}, repository=self.repository).data, [first])
        self.assertEqual(list_tasks({"date": "2026-10-30"}, repository=self.repository).data, [])
        for arguments in ({"date": "tomorrow"}, {"date": 12345}, {"extra": True}):
            self.assertEqual(list_tasks(arguments, repository=self.repository).error, "VALIDATION_ERROR")
        self.assertEqual(len(self.repository.list()), 2)

    def test_find_conflicts_is_deterministic_and_read_only(self) -> None:
        first = self.create()
        query = {"start_time": "2026-10-28T09:30:00", "end_time": "2026-10-28T10:30:00"}
        result = find_conflicts(query, repository=self.repository)
        self.assertTrue(result.success)
        self.assertEqual(result.intent, Intent.CHECK_CONFLICTS)
        self.assertEqual(result.data, {"has_conflicts": True, "conflicts": [first]})
        self.assertEqual(find_conflicts(query, repository=self.repository), result)
        self.assertEqual(find_conflicts({**query, "exclude_task_id": first["id"]}, repository=self.repository).data, {"has_conflicts": False, "conflicts": []})
        adjacent = {"start_time": "2026-10-28T10:00:00", "end_time": "2026-10-28T11:00:00"}
        self.assertEqual(find_conflicts(adjacent, repository=self.repository).data["has_conflicts"], False)
        self.assertEqual(list_tasks({}, repository=self.repository).data, [first])

    def test_invalid_conflict_queries_and_unknown_exclusion_are_explicit(self) -> None:
        query = {"start_time": self.task["start_time"], "end_time": self.task["end_time"]}
        for arguments in ({}, {"start_time": query["start_time"]}, {**query, "end_time": query["start_time"]}, {**query, "exclude_task_id": "1"}, {**query, "end_time": "2026-10-29T10:00:00"}):
            self.assertEqual(find_conflicts(arguments, repository=self.repository).error, "VALIDATION_ERROR")
        self.assertEqual(find_conflicts({**query, "exclude_task_id": 999}, repository=self.repository).error, "TASK_NOT_FOUND")

    def test_storage_errors_do_not_leak_details(self) -> None:
        repository = Mock(spec=TaskRepository)
        repository.save.side_effect = sqlite3.OperationalError("private database path")
        result = create_task(self.task, repository=repository)
        self.assertFalse(result.success)
        self.assertEqual(result.error, "STORAGE_ERROR")
        self.assertNotIn("private", result.model_dump_json())