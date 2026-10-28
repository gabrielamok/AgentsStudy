import socket
from collections.abc import Callable
from pathlib import Path

import pytest

from agent.contracts import AgentResponse, Intent
from agent.tools import create_task, delete_task, find_conflicts, list_tasks, update_task
from api.models import Task, TaskInput
from api.repository import TaskRepository


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Deterministic tool tests must not use the network or AI services")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)


@pytest.fixture
def repository(tmp_path: Path) -> TaskRepository:
    repository = TaskRepository(tmp_path / "tasks.sqlite3")
    repository.initialize()
    return repository


@pytest.fixture
def task_arguments() -> dict[str, object]:
    return {
        "title": "AI-103 Study", "start_time": "2026-11-02T09:00:00",
        "end_time": "2026-11-02T10:00:00", "priority": "high", "type": "flexible",
    }


@pytest.fixture
def existing_task(repository: TaskRepository, task_arguments: dict[str, object]) -> Task:
    return repository.save(TaskInput.model_validate(task_arguments))


def test_create_task_persists_validated_fields(repository: TaskRepository, task_arguments: dict[str, object]) -> None:
    # Arrange
    original_arguments = task_arguments.copy()

    # Act
    result = create_task(task_arguments, repository=repository)

    # Assert
    assert result.success is True
    assert result.intent == Intent.CREATE_TASK
    assert result.tool_called == "create_task"
    assert result.error is None
    assert isinstance(result.data, dict)
    assert result.data == {**task_arguments, "id": repository.list()[0].id}
    assert task_arguments == original_arguments
    assert AgentResponse.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("missing_field", ["title", "start_time", "end_time", "priority", "type"])
def test_create_task_requires_each_field(repository: TaskRepository, task_arguments: dict[str, object], missing_field: str) -> None:
    # Arrange
    arguments = {key: value for key, value in task_arguments.items() if key != missing_field}

    # Act
    result = create_task(arguments, repository=repository)

    # Assert
    assert result.success is False
    assert result.error == "VALIDATION_ERROR"
    assert repository.list() == []


@pytest.mark.parametrize("changes", [
    {"title": ""}, {"title": " "}, {"title": " AI-103 Study "}, {"title": 123},
    {"priority": "urgent"}, {"type": "unknown"}, {"unexpected": True}, {"id": 1},
    {"start_time": "not-a-time"}, {"start_time": "2026-11-02T25:00:00"},
    {"start_time": "2026-11-02"}, {"start_time": 12345},
    {"start_time": "2026-11-02T09:00:00Z"},
])
def test_create_task_rejects_invalid_input_without_correction(repository: TaskRepository, task_arguments: dict[str, object], changes: dict[str, object]) -> None:
    # Arrange
    arguments = {**task_arguments, **changes}
    before = arguments.copy()

    # Act
    result = create_task(arguments, repository=repository)

    # Assert
    assert result.success is False
    assert result.error == "VALIDATION_ERROR"
    assert arguments == before
    assert repository.list() == []


def test_duplicated_task_is_rejected(repository: TaskRepository, task_arguments: dict[str, object], existing_task: Task) -> None:
    # Arrange
    before = repository.list()

    # Act
    result = create_task(task_arguments, repository=repository)

    # Assert
    assert result.success is False
    assert result.error == "SCHEDULE_CONFLICT"
    assert result.data == {"conflicts": [existing_task.model_dump(mode="json")]}
    assert repository.list() == before


@pytest.mark.parametrize(("start", "end"), [
    ("08:30", "09:30"), ("09:30", "10:30"), ("08:00", "11:00"), ("09:15", "09:45"),
])
def test_create_task_rejects_overlapping_schedules(repository: TaskRepository, task_arguments: dict[str, object], existing_task: Task, start: str, end: str) -> None:
    # Arrange
    arguments = {**task_arguments, "title": "Fictional review", "start_time": f"2026-11-02T{start}:00", "end_time": f"2026-11-02T{end}:00"}

    # Act
    result = create_task(arguments, repository=repository)

    # Assert
    assert result.success is False
    assert result.error == "SCHEDULE_CONFLICT"
    assert repository.list() == [existing_task]


@pytest.mark.parametrize(("start", "end"), [("08:00", "09:00"), ("10:00", "11:00")])
def test_create_task_allows_adjacent_intervals(repository: TaskRepository, task_arguments: dict[str, object], existing_task: Task, start: str, end: str) -> None:
    # Arrange
    arguments = {**task_arguments, "start_time": f"2026-11-02T{start}:00", "end_time": f"2026-11-02T{end}:00"}

    # Act
    result = create_task(arguments, repository=repository)

    # Assert
    assert result.success is True
    assert len(repository.list()) == 2
    assert existing_task in repository.list()


def test_update_task_preserves_id_and_replaces_fields(repository: TaskRepository, task_arguments: dict[str, object], existing_task: Task) -> None:
    # Arrange
    arguments = {**task_arguments, "task_id": existing_task.id, "title": "Updated AI-103 Study", "start_time": "2026-11-02T10:00:00", "end_time": "2026-11-02T11:00:00", "priority": "low", "type": "fixed"}

    # Act
    result = update_task(arguments, repository=repository)

    # Assert
    assert result.success is True
    assert result.intent == Intent.UPDATE_TASK
    assert result.tool_called == "update_task"
    assert result.data == {key: value for key, value in arguments.items() if key != "task_id"} | {"id": existing_task.id}
    assert [task.model_dump(mode="json") for task in repository.list()] == [result.data]


@pytest.mark.parametrize("changes", [{"task_id": 0}, {"task_id": -1}, {"task_id": "1"}, {"task_id": True}, {"title": " "}, {"unexpected": 1}])
def test_update_task_invalid_input_preserves_original(repository: TaskRepository, task_arguments: dict[str, object], existing_task: Task, changes: dict[str, object]) -> None:
    # Arrange
    arguments = {**task_arguments, "task_id": existing_task.id, **changes}

    # Act
    result = update_task(arguments, repository=repository)

    # Assert
    assert result.success is False
    assert result.error == "VALIDATION_ERROR"
    assert repository.list() == [existing_task]


@pytest.mark.parametrize("missing_field", ["task_id", "title", "start_time", "end_time", "priority", "type"])
def test_update_task_requires_complete_replacement(repository: TaskRepository, task_arguments: dict[str, object], existing_task: Task, missing_field: str) -> None:
    # Arrange
    arguments = {**task_arguments, "task_id": existing_task.id}
    del arguments[missing_field]

    # Act
    result = update_task(arguments, repository=repository)

    # Assert
    assert result.success is False
    assert result.error == "VALIDATION_ERROR"
    assert repository.list() == [existing_task]


def test_update_task_rejects_conflict_without_mutation(repository: TaskRepository, task_arguments: dict[str, object], existing_task: Task) -> None:
    # Arrange
    second = repository.save(TaskInput.model_validate({**task_arguments, "title": "Fictional meeting", "start_time": "2026-11-02T11:00:00", "end_time": "2026-11-02T12:00:00"}))
    before = repository.list()

    # Act
    result = update_task({**task_arguments, "task_id": second.id}, repository=repository)

    # Assert
    assert result.success is False
    assert result.error == "SCHEDULE_CONFLICT"
    assert result.data == {"conflicts": [existing_task.model_dump(mode="json")]}
    assert repository.list() == before


def test_update_task_does_not_conflict_with_itself(repository: TaskRepository, task_arguments: dict[str, object], existing_task: Task) -> None:
    # Arrange
    arguments = {**task_arguments, "task_id": existing_task.id}

    # Act
    result = update_task(arguments, repository=repository)

    # Assert
    assert result.success is True
    assert repository.list() == [existing_task]


def test_update_nonexistent_task_does_not_create_one(repository: TaskRepository, task_arguments: dict[str, object], existing_task: Task) -> None:
    # Arrange
    arguments = {**task_arguments, "task_id": existing_task.id + 100}

    # Act
    result = update_task(arguments, repository=repository)

    # Assert
    assert result.success is False
    assert result.error == "TASK_NOT_FOUND"
    assert repository.list() == [existing_task]


def test_delete_task_removes_only_identified_record(repository: TaskRepository, task_arguments: dict[str, object], existing_task: Task) -> None:
    # Arrange
    survivor = repository.save(TaskInput.model_validate({**task_arguments, "title": "Fictional meeting", "start_time": "2026-11-02T11:00:00", "end_time": "2026-11-02T12:00:00"}))

    # Act
    result = delete_task({"task_id": existing_task.id}, repository=repository)

    # Assert
    assert result.success is True
    assert result.intent == Intent.DELETE_TASK
    assert result.tool_called == "delete_task"
    assert result.data == {"task_id": existing_task.id}
    assert repository.list() == [survivor]


@pytest.mark.parametrize("arguments", [{}, {"task_id": 0}, {"task_id": -1}, {"task_id": "1"}, {"task_id": True}, {"task_id": None}, {"task_id": 1, "extra": True}])
def test_delete_task_rejects_invalid_identifiers(repository: TaskRepository, existing_task: Task, arguments: dict[str, object]) -> None:
    # Arrange
    before = repository.list()

    # Act
    result = delete_task(arguments, repository=repository)

    # Assert
    assert result.success is False
    assert result.error == "VALIDATION_ERROR"
    assert repository.list() == before


def test_delete_nonexistent_task_preserves_existing_tasks(repository: TaskRepository, existing_task: Task) -> None:
    # Arrange
    arguments = {"task_id": existing_task.id + 100}

    # Act
    result = delete_task(arguments, repository=repository)

    # Assert
    assert result.success is False
    assert result.error == "TASK_NOT_FOUND"
    assert repository.list() == [existing_task]


def test_list_tasks_returns_empty_list(repository: TaskRepository) -> None:
    # Arrange
    arguments: dict[str, object] = {}

    # Act
    result = list_tasks(arguments, repository=repository)

    # Assert
    assert result.success is True
    assert result.intent == Intent.LIST_TASKS
    assert result.tool_called == "list_tasks"
    assert result.data == []


def test_list_tasks_is_ordered_and_read_only(repository: TaskRepository, task_arguments: dict[str, object]) -> None:
    # Arrange
    later = repository.save(TaskInput.model_validate({**task_arguments, "start_time": "2026-11-03T09:00:00", "end_time": "2026-11-03T10:00:00"}))
    earlier = repository.save(TaskInput.model_validate(task_arguments))
    before = repository.list()

    # Act
    result = list_tasks({}, repository=repository)

    # Assert
    assert result.success is True
    assert result.data == [earlier.model_dump(mode="json"), later.model_dump(mode="json")]
    assert repository.list() == before


@pytest.mark.parametrize(("date", "matches"), [("2026-11-02", True), ("2026-11-04", False)])
def test_list_tasks_filters_by_explicit_date(repository: TaskRepository, existing_task: Task, date: str, matches: bool) -> None:
    # Arrange
    arguments = {"date": date}

    # Act
    result = list_tasks(arguments, repository=repository)

    # Assert
    assert result.success is True
    assert result.data == ([existing_task.model_dump(mode="json")] if matches else [])
    assert repository.list() == [existing_task]


@pytest.mark.parametrize("arguments", [{"date": "tomorrow"}, {"date": "2026-02-30"}, {"date": 12345}, {"date": True}, {"extra": True}])
def test_list_tasks_rejects_invalid_filters(repository: TaskRepository, existing_task: Task, arguments: dict[str, object]) -> None:
    # Arrange
    before = repository.list()

    # Act
    result = list_tasks(arguments, repository=repository)

    # Assert
    assert result.success is False
    assert result.error == "VALIDATION_ERROR"
    assert repository.list() == before


@pytest.mark.parametrize(("start", "end", "has_conflicts"), [
    ("09:00", "10:00", True), ("09:30", "10:30", True), ("08:00", "11:00", True),
    ("09:15", "09:45", True), ("08:00", "09:00", False), ("10:00", "11:00", False),
])
def test_find_conflicts_is_deterministic_and_read_only(repository: TaskRepository, existing_task: Task, start: str, end: str, has_conflicts: bool) -> None:
    # Arrange
    arguments = {"start_time": f"2026-11-02T{start}:00", "end_time": f"2026-11-02T{end}:00"}

    # Act
    result = find_conflicts(arguments, repository=repository)
    repeated = find_conflicts(arguments, repository=repository)

    # Assert
    assert result.success is True
    assert result.intent == Intent.CHECK_CONFLICTS
    assert result.tool_called == "find_conflicts"
    assert result.data == {"has_conflicts": has_conflicts, "conflicts": [existing_task.model_dump(mode="json")] if has_conflicts else []}
    assert repeated == result
    assert repository.list() == [existing_task]


def test_find_conflicts_excludes_only_requested_task(repository: TaskRepository, existing_task: Task) -> None:
    # Arrange
    arguments = {"start_time": "2026-11-02T09:30:00", "end_time": "2026-11-02T10:30:00", "exclude_task_id": existing_task.id}

    # Act
    result = find_conflicts(arguments, repository=repository)

    # Assert
    assert result.success is True
    assert result.data == {"has_conflicts": False, "conflicts": []}
    assert repository.list() == [existing_task]


@pytest.mark.parametrize("changes", [{"exclude_task_id": "1"}, {"exclude_task_id": 0}, {"exclude_task_id": True}, {"extra": True}])
def test_find_conflicts_rejects_invalid_arguments(repository: TaskRepository, existing_task: Task, changes: dict[str, object]) -> None:
    # Arrange
    arguments = {"start_time": "2026-11-02T09:00:00", "end_time": "2026-11-02T10:00:00", **changes}

    # Act
    result = find_conflicts(arguments, repository=repository)

    # Assert
    assert result.success is False
    assert result.error == "VALIDATION_ERROR"
    assert repository.list() == [existing_task]


def test_find_conflicts_reports_nonexistent_exclusion(repository: TaskRepository, existing_task: Task) -> None:
    # Arrange
    arguments = {"start_time": "2026-11-02T09:00:00", "end_time": "2026-11-02T10:00:00", "exclude_task_id": existing_task.id + 100}

    # Act
    result = find_conflicts(arguments, repository=repository)

    # Assert
    assert result.success is False
    assert result.error == "TASK_NOT_FOUND"
    assert repository.list() == [existing_task]


@pytest.mark.parametrize("tool", [create_task, update_task, find_conflicts], ids=["create", "update", "conflicts"])
@pytest.mark.parametrize(("start", "end"), [
    ("2026-11-02T10:00:00", "2026-11-02T10:00:00"),
    ("2026-11-02T11:00:00", "2026-11-02T10:00:00"),
    ("2026-11-02T23:00:00", "2026-11-03T01:00:00"),
    ("2026-11-02T25:00:00", "2026-11-02T26:00:00"),
])
def test_invalid_time_ranges_never_change_storage(repository: TaskRepository, task_arguments: dict[str, object], existing_task: Task, tool: Callable[..., AgentResponse], start: str, end: str) -> None:
    # Arrange
    arguments = {} if tool is find_conflicts else task_arguments.copy()
    arguments.update(start_time=start, end_time=end)
    if tool is update_task:
        arguments["task_id"] = existing_task.id
    before = repository.list()

    # Act
    result = tool(arguments, repository=repository)

    # Assert
    assert result.success is False
    assert result.error == "VALIDATION_ERROR"
    assert repository.list() == before