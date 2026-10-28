import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from api.models import Task, TaskInput


class TaskNotFound(Exception):
    pass


class ScheduleConflict(Exception):
    def __init__(self, tasks: list[Task]) -> None:
        self.tasks = tasks
        super().__init__("This time overlaps an existing task")


class TaskRepository:
    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS tasks ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, "
                "start_time TEXT NOT NULL, end_time TEXT NOT NULL, "
                "priority TEXT NOT NULL, type TEXT NOT NULL)"
            )

    def list(self) -> list[Task]:
        with self.connection() as connection:
            rows = connection.execute("SELECT * FROM tasks ORDER BY start_time, id").fetchall()
        return [Task.model_validate(dict(row)) for row in rows]

    def save(self, task: TaskInput, task_id: int | None = None) -> Task:
        values = task.model_dump(mode="json")
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if task_id is not None:
                exists = connection.execute("SELECT id FROM tasks WHERE id = ?", (task_id,)).fetchone()
                if exists is None:
                    raise TaskNotFound()
            rows = connection.execute(
                "SELECT * FROM tasks WHERE start_time < ? AND end_time > ? "
                "AND (? IS NULL OR id != ?)",
                (values["end_time"], values["start_time"], task_id, task_id),
            ).fetchall()
            if rows:
                raise ScheduleConflict([Task.model_validate(dict(row)) for row in rows])
            if task_id is None:
                cursor = connection.execute(
                    "INSERT INTO tasks (title, start_time, end_time, priority, type) "
                    "VALUES (:title, :start_time, :end_time, :priority, :type)", values,
                )
                task_id = cursor.lastrowid
            else:
                connection.execute(
                    "UPDATE tasks SET title=:title, start_time=:start_time, "
                    "end_time=:end_time, priority=:priority, type=:type WHERE id=:id",
                    {**values, "id": task_id},
                )
        return Task.model_validate({**values, "id": task_id})

    def delete(self, task_id: int) -> None:
        with self.connection() as connection:
            cursor = connection.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            if cursor.rowcount == 0:
                raise TaskNotFound()