import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from api.models import TaskInput
from api.repository import ScheduleConflict, TaskRepository


class RepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.repository = TaskRepository(Path(self.directory.name) / "tasks.sqlite3")
        self.repository.initialize()

    def task(self, start: str = "09:00", end: str = "10:00") -> TaskInput:
        return TaskInput.model_validate({
            "title": "Review PR", "start_time": f"2026-10-28T{start}",
            "end_time": f"2026-10-28T{end}", "priority": "high", "type": "flexible",
        })

    def test_overlap_rejected_and_adjacent_allowed(self) -> None:
        self.repository.save(self.task())
        with self.assertRaises(ScheduleConflict):
            self.repository.save(self.task("09:30", "10:30"))
        self.repository.save(self.task("10:00", "11:00"))
        self.assertEqual(len(self.repository.list()), 2)

    def test_update_excludes_itself_and_preserves_conflicting_original(self) -> None:
        first = self.repository.save(self.task())
        second = self.repository.save(self.task("10:00", "11:00"))
        self.repository.save(self.task(), first.id)
        with self.assertRaises(ScheduleConflict):
            self.repository.save(self.task("09:30", "10:30"), second.id)
        self.assertEqual(self.repository.list()[1], second)

    def test_persistence(self) -> None:
        saved = self.repository.save(self.task())
        reopened = TaskRepository(self.repository.path)
        self.assertEqual(reopened.list(), [saved])

    def test_concurrent_conflicting_writes_only_save_once(self) -> None:
        def save() -> bool:
            try:
                self.repository.save(self.task())
                return True
            except ScheduleConflict:
                return False

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda index: save(), range(2)))
        self.assertEqual(sum(results), 1)
        self.assertEqual(len(self.repository.list()), 1)