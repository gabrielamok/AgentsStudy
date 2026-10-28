from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from api.models import TaskInput


class Intent(StrEnum):
    CREATE_TASK = "CREATE_TASK"
    UPDATE_TASK = "UPDATE_TASK"
    DELETE_TASK = "DELETE_TASK"
    LIST_TASKS = "LIST_TASKS"
    CHECK_CONFLICTS = "CHECK_CONFLICTS"
    UNKNOWN = "UNKNOWN"


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Task(TaskInput):
    id: int | None = Field(default=None, gt=0)


class AgentRequest(_ContractModel):
    message: str = Field(min_length=1, max_length=4000)
    context: dict[str, JsonValue] = Field(default_factory=dict)


class ToolCall(_ContractModel):
    tool_name: str = Field(min_length=1, max_length=200)
    tool_arguments: dict[str, JsonValue] = Field(default_factory=dict)


class AgentDecision(_ContractModel):
    intent: Intent
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    reasoning_summary: str = Field(min_length=1, max_length=2000)
    tool_name: str | None = Field(default=None, min_length=1, max_length=200)
    tool_arguments: dict[str, JsonValue] = Field(default_factory=dict)


class AgentResponse(_ContractModel):
    success: bool
    intent: Intent
    tool_called: str | None = Field(default=None, min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=4000)
    data: JsonValue = None
    error: str | None = Field(default=None, min_length=1)