import sqlite3
from collections.abc import Callable, Mapping
from datetime import date as Date, datetime
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, JsonValue, NaiveDatetime, ValidationError, field_validator, model_validator

from agent.contracts import AgentResponse, Intent
from api.models import TaskInput, overlaps
from api.repository import ScheduleConflict, TaskNotFound, TaskRepository


def _parse_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value == value.strip() and ("T" in value or " " in value):
        return datetime.fromisoformat(value)
    raise ValueError("Use an explicit ISO datetime, not a date or numeric timestamp")


LocalDatetime = Annotated[NaiveDatetime, BeforeValidator(_parse_datetime)]
TaskId = Annotated[int, Field(strict=True, gt=0)]


class _Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _TaskArguments(TaskInput):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    title: str = Field(strict=True, min_length=1, max_length=200)
    start_time: LocalDatetime
    end_time: LocalDatetime

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("title must not be blank or contain surrounding whitespace")
        return value


class _UpdateArguments(_TaskArguments):
    task_id: TaskId


class _IdArguments(_Arguments):
    task_id: TaskId


class _ListArguments(_Arguments):
    date: Date | None = None

    @field_validator("date", mode="before")
    @classmethod
    def validate_date(cls, value: object) -> Date | None:
        if value is None or isinstance(value, Date) and not isinstance(value, datetime):
            return value
        if isinstance(value, str) and len(value) == 10 and value[4] == "-" and value[7] == "-":
            return Date.fromisoformat(value)
        raise ValueError("date must be a YYYY-MM-DD date")


class _ConflictArguments(_Arguments):
    start_time: LocalDatetime
    end_time: LocalDatetime
    exclude_task_id: TaskId | None = None

    @model_validator(mode="after")
    def validate_interval(self) -> "_ConflictArguments":
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be after start_time")
        if self.start_time.date() != self.end_time.date():
            raise ValueError("A daily interval must start and end on the same date")
        return self


def _execute(tool_name: str, intent: Intent, operation: Callable[[], JsonValue]) -> AgentResponse:
    try:
        data = operation()
        return AgentResponse(success=True, intent=intent, tool_called=tool_name, message=f"{tool_name} completed", data=data)
    except ValidationError as error:
        code = "VALIDATION_ERROR"
        message = "Invalid tool arguments"
        data = {"validation_errors": [
            {"field": ".".join(str(part) for part in item["loc"]), "message": item["msg"], "type": item["type"]}
            for item in error.errors(include_input=False, include_context=False, include_url=False)
        ]}
    except TaskNotFound:
        code, message, data = "TASK_NOT_FOUND", "Task not found", None
    except ScheduleConflict as error:
        code, message = "SCHEDULE_CONFLICT", "Requested times overlap an existing task"
        data = {"conflicts": [task.model_dump(mode="json") for task in error.tasks]}
    except sqlite3.Error:
        code, message, data = "STORAGE_ERROR", "Task storage is unavailable", None
    return AgentResponse(success=False, intent=intent, tool_called=tool_name, message=message, data=data, error=code)


def create_task(arguments: Mapping[str, object], *, repository: TaskRepository) -> AgentResponse:
    """Create from all five task fields; validate before any repository write."""
    def operation() -> JsonValue:
        task = _TaskArguments.model_validate(arguments)
        return repository.save(task).model_dump(mode="json")

    return _execute("create_task", Intent.CREATE_TASK, operation)


def update_task(arguments: Mapping[str, object], *, repository: TaskRepository) -> AgentResponse:
    """Replace an identified task; task_id and all five task fields are required."""
    def operation() -> JsonValue:
        parameters = _UpdateArguments.model_validate(arguments)
        task = TaskInput.model_validate(parameters.model_dump(exclude={"task_id"}))
        return repository.save(task, parameters.task_id).model_dump(mode="json")

    return _execute("update_task", Intent.UPDATE_TASK, operation)


def delete_task(arguments: Mapping[str, object], *, repository: TaskRepository) -> AgentResponse:
    """Delete only the explicitly supplied positive integer task_id."""
    def operation() -> JsonValue:
        parameters = _IdArguments.model_validate(arguments)
        repository.delete(parameters.task_id)
        return {"task_id": parameters.task_id}

    return _execute("delete_task", Intent.DELETE_TASK, operation)


def list_tasks(arguments: Mapping[str, object], *, repository: TaskRepository) -> AgentResponse:
    """List all tasks, optionally filtered by an explicit YYYY-MM-DD date."""
    def operation() -> JsonValue:
        parameters = _ListArguments.model_validate(arguments)
        return [task.model_dump(mode="json") for task in repository.list()
                if parameters.date is None or task.start_time.date() == parameters.date]

    return _execute("list_tasks", Intent.LIST_TASKS, operation)


def find_conflicts(arguments: Mapping[str, object], *, repository: TaskRepository) -> AgentResponse:
    """Check a start/end interval, optionally excluding an existing task ID; never write."""
    def operation() -> JsonValue:
        parameters = _ConflictArguments.model_validate(arguments)
        tasks = repository.list()
        if parameters.exclude_task_id is not None and not any(task.id == parameters.exclude_task_id for task in tasks):
            raise TaskNotFound()
        conflicts = [task.model_dump(mode="json") for task in tasks
                     if task.id != parameters.exclude_task_id and overlaps(parameters.start_time, parameters.end_time, task)]
        return {"has_conflicts": bool(conflicts), "conflicts": conflicts}

    return _execute("find_conflicts", Intent.CHECK_CONFLICTS, operation)