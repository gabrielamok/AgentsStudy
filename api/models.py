from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, NaiveDatetime, model_validator


class Priority(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskType(StrEnum):
    FIXED = "fixed"
    FLEXIBLE = "flexible"


class TaskInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=200)
    start_time: NaiveDatetime
    end_time: NaiveDatetime
    priority: Priority
    type: TaskType

    @model_validator(mode="after")
    def validate_interval(self) -> "TaskInput":
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be after start_time")
        if self.start_time.date() != self.end_time.date():
            raise ValueError("A daily task must start and end on the same date")
        return self


class Task(TaskInput):
    id: int


def overlaps(start: datetime, end: datetime, task: TaskInput) -> bool:
    return start < task.end_time and end > task.start_time