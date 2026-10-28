"""Dataset evaluations, separate from production code.

Run: python -m pytest tests/evals/test_agent_evals.py -v
Set PLANNER_EVAL_AGENT=package.module:function to a request-only decision adapter.
The callable receives AgentRequest and returns a JSON-compatible object containing
decision (AgentDecision), should_execute (bool), requires_clarification (bool),
and response (AgentResponse, required only when execution is declined).
The adapter must not execute tools: this runner owns isolated tool execution.
No expectations or evaluation case IDs are passed to the adapter. Missing adapter
configuration fails each case explicitly; no mock predictions, skips, or xfails
are used for dataset evaluations. Override agent_predictor to inject an adapter.
"""

import copy
import importlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, StrictBool

from agent.contracts import AgentDecision, AgentRequest, AgentResponse
from agent.tools import create_task, delete_task, find_conflicts, list_tasks, update_task
from api.models import Task
from api.repository import TaskRepository


DATASET = Path(__file__).with_name("planner-evals.json")
CASES = json.loads(DATASET.read_text(encoding="utf-8"))
TOOLS = {tool.__name__: tool for tool in (
    create_task, update_task, delete_task, list_tasks, find_conflicts,
)}


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: AgentDecision
    should_execute: StrictBool
    requires_clarification: StrictBool
    response: AgentResponse | None = None


@pytest.fixture
def agent_predictor() -> Callable[[AgentRequest], object]:
    def predict(request: AgentRequest) -> object:
        target = os.environ.get("PLANNER_EVAL_AGENT", "")
        module_name, separator, function_name = target.partition(":")
        if not separator or not module_name or not function_name:
            raise RuntimeError(
                "No CRUD decision adapter configured. Set PLANNER_EVAL_AGENT="
                "package.module:function. AgentService.plan returns schedule proposals, "
                "not AgentDecision/tool execution decisions."
            )
        adapter = getattr(importlib.import_module(module_name), function_name)
        if not callable(adapter):
            raise TypeError(f"{target} is not callable")
        return adapter(request)

    return predict


def _snapshot(repository: TaskRepository) -> list[dict[str, Any]]:
    return sorted(
        [task.model_dump(mode="json") for task in repository.list()],
        key=lambda task: task["id"],
    )


def _seed(repository: TaskRepository, initial_tasks: list[dict[str, Any]]) -> None:
    repository.initialize()
    with repository.connection() as connection:
        for item in initial_tasks:
            task = Task.model_validate(item).model_dump(mode="json")
            connection.execute(
                "INSERT INTO tasks (id, title, start_time, end_time, priority, type) "
                "VALUES (:id, :title, :start_time, :end_time, :priority, :type)", task,
            )


def _evaluate(
    case: dict[str, Any], predictor: Callable[[AgentRequest], object],
    repository: TaskRepository,
) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    report: dict[str, Any] = {
        "test_case_id": case["id"], "user_input": case["user_input"],
        "expected_intent": case["expected_intent"], "actual_intent": None,
        "expected_tool": case["expected_tool"], "actual_tool": None,
        "expected_arguments": case["expected_arguments"], "actual_arguments": None,
        "expected_should_execute": case["should_execute"], "actual_should_execute": None,
        "expected_outcome": case["expected_outcome"], "actual_outcome": None,
    }

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    try:
        expected = case["expected_outcome"]
        _seed(repository, expected["initial_tasks"])
        before = _snapshot(repository)
        raw = predictor(AgentRequest(
            message=case["user_input"], context={"tasks": copy.deepcopy(before)},
        ))
        if isinstance(raw, BaseModel):
            raw = raw.model_dump(mode="json")
        report["raw_prediction"] = raw
        if isinstance(raw, dict):
            decision = raw.get("decision", {})
            if isinstance(decision, dict):
                report.update(
                    actual_intent=decision.get("intent"), actual_tool=decision.get("tool_name"),
                    actual_arguments=decision.get("tool_arguments"),
                )
            report["actual_should_execute"] = raw.get("should_execute")
            report["actual_outcome"] = raw.get("response")
        prediction = Prediction.model_validate_json(json.dumps(raw), strict=True)
        decision = prediction.decision
        report.update(
            actual_intent=decision.intent.value, actual_tool=decision.tool_name,
            actual_arguments=decision.tool_arguments,
            actual_should_execute=prediction.should_execute,
        )
        check(_snapshot(repository) == before, "Decision adapter mutated storage before execution")
        check(decision.intent.value == case["expected_intent"], "Intent mismatch")
        check(decision.tool_name == case["expected_tool"], "Tool selection mismatch")
        check(
            json.dumps(decision.tool_arguments, sort_keys=True)
            == json.dumps(case["expected_arguments"], sort_keys=True),
            "Tool argument mismatch",
        )
        check(prediction.should_execute == case["should_execute"], "should_execute mismatch")
        check(
            prediction.requires_clarification == expected["requires_clarification"],
            "Clarification behavior mismatch",
        )

        executed = False
        if prediction.should_execute:
            check(prediction.response is None, "Executable decision supplied an unexecuted response")
            if decision.tool_name not in TOOLS:
                raise ValueError(f"Unknown executable tool: {decision.tool_name!r}")
            executed = True
            response = TOOLS[decision.tool_name](decision.tool_arguments, repository=repository)
        else:
            check(decision.tool_name is None, "Declined execution must not select a tool")
            check(decision.tool_arguments == {}, "Declined execution must not supply tool arguments")
            if prediction.response is None:
                raise ValueError("Declined execution requires a structured AgentResponse")
            response = prediction.response
        response = AgentResponse.model_validate_json(response.model_dump_json(), strict=True)
        after = _snapshot(repository)
        report["actual_outcome"] = {
            **response.model_dump(mode="json"), "executed": executed,
            "requires_clarification": prediction.requires_clarification,
            "persisted_tasks": after,
        }
        check(executed == case["should_execute"], "Tool execution behavior mismatch")
        check(response.intent == decision.intent, "Response intent differs from decision")
        check(response.tool_called == (decision.tool_name if executed else None), "Response tool mismatch")
        check(response.success == expected["success"], "Outcome success mismatch")
        check(response.error == expected["error"], "Outcome error mismatch")

        if "returned_task_ids" in expected:
            check(isinstance(response.data, list), "List response data must be an array")
            if isinstance(response.data, list):
                check(
                    response.data == [task for task in before if task["id"] in expected["returned_task_ids"]],
                    "Returned tasks mismatch",
                )
                check([task["id"] for task in response.data] == expected["returned_task_ids"], "Returned task IDs mismatch")
        if "conflicting_task_ids" in expected or "has_conflicts" in expected:
            check(isinstance(response.data, dict), "Conflict response data must be an object")
            if isinstance(response.data, dict):
                check(
                    response.data.get("conflicts") == [
                        task for task in before if task["id"] in expected.get("conflicting_task_ids", [])
                    ], "Conflicting tasks mismatch",
                )
                if "has_conflicts" in expected:
                    check(response.data.get("has_conflicts") is expected["has_conflicts"], "Conflict flag mismatch")

        state_change = expected["state_change"]
        expected_state = copy.deepcopy(before)
        arguments = case["expected_arguments"]
        if state_change == "create_one":
            added = [task for task in after if task["id"] not in {item["id"] for item in before}]
            check(len(added) == 1, "Creation must persist exactly one new task")
            if len(added) == 1:
                created = {**arguments, "id": added[0]["id"]}
                check(type(created["id"]) is int and created["id"] > 0, "Created ID must be positive")
                expected_state.append(created)
                check(response.data == created, "Created response differs from expected saved task")
        elif state_change == "update_one_preserving_id":
            updated = {key: value for key, value in arguments.items() if key != "task_id"}
            updated["id"] = arguments["task_id"]
            expected_state = [updated if task["id"] == updated["id"] else task for task in before]
            check(response.data == updated, "Updated response differs from expected saved task")
        elif state_change == "delete_only_task_101":
            check(arguments["task_id"] == 101, "Unsupported deletion expectation")
            expected_state = [task for task in before if task["id"] != 101]
            check(response.data == {"task_id": 101}, "Deleted task response mismatch")
        elif state_change != "none":
            raise ValueError(f"Unsupported expected state change: {state_change!r}")
        report["expected_persisted_tasks"] = sorted(expected_state, key=lambda task: task["id"])
        check(after == report["expected_persisted_tasks"], "Final persisted state mismatch")
    except Exception as error:
        failures.append(f"{type(error).__name__}: {error}")
        report["evaluation_error"] = f"{type(error).__name__}: {error}"
        if report["actual_outcome"] is None:
            report["actual_outcome"] = {"evaluation_error": report["evaluation_error"]}
    finally:
        try:
            report["actual_persisted_tasks"] = _snapshot(repository)
        except Exception as error:
            failures.append(f"Cannot inspect final persisted state: {error}")
    report["failures"] = failures
    return failures, report


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_agent_evaluation(
    case: dict[str, Any], agent_predictor: Callable[[AgentRequest], object], tmp_path: Path,
) -> None:
    failures, report = _evaluate(case, agent_predictor, TaskRepository(tmp_path / "eval.sqlite3"))
    if failures:
        pytest.fail(json.dumps(report, indent=2, ensure_ascii=True, default=repr), pytrace=False)


@pytest.fixture
def runner_control_case() -> dict[str, Any]:
    return {
        "id": "runner-control-only", "user_input": "List all tasks.",
        "expected_intent": "LIST_TASKS", "expected_tool": "list_tasks",
        "expected_arguments": {}, "should_execute": True,
        "expected_outcome": {
            "initial_tasks": [], "success": True, "error": None,
            "returned_task_ids": [], "state_change": "none", "requires_clarification": False,
        },
    }


def _control_prediction(request: AgentRequest) -> dict[str, Any]:
    return {
        "decision": {
            "intent": "LIST_TASKS", "confidence": 1.0, "reasoning_summary": "Runner control, not a dataset prediction",
            "tool_name": "list_tasks", "tool_arguments": {},
        },
        "should_execute": True, "requires_clarification": False,
    }


def test_runner_accepts_valid_control(runner_control_case: dict[str, Any], tmp_path: Path) -> None:
    failures, report = _evaluate(runner_control_case, _control_prediction, TaskRepository(tmp_path / "control.sqlite3"))
    assert failures == [], report


@pytest.mark.parametrize(("field", "value", "expected_failure"), [
    ("intent", "UNKNOWN", "Intent mismatch"),
    ("tool_name", "find_conflicts", "Tool selection mismatch"),
    ("tool_arguments", {"date": "2026-11-02"}, "Tool argument mismatch"),
    ("should_execute", False, "should_execute mismatch"),
    ("requires_clarification", True, "Clarification behavior mismatch"),
    ("should_execute", "true", "ValidationError"),
])
def test_runner_reports_bad_predictions(
    runner_control_case: dict[str, Any], tmp_path: Path, field: str, value: Any, expected_failure: str,
) -> None:
    def predict(request: AgentRequest) -> dict[str, Any]:
        prediction = _control_prediction(request)
        target = prediction["decision"] if field in {"intent", "tool_name", "tool_arguments"} else prediction
        target[field] = value
        return prediction

    failures, report = _evaluate(runner_control_case, predict, TaskRepository(tmp_path / "control.sqlite3"))
    assert any(expected_failure in failure for failure in failures), report
    assert all(key in report for key in (
        "test_case_id", "user_input", "expected_intent", "actual_intent",
        "expected_tool", "actual_tool", "expected_arguments", "actual_arguments",
        "expected_outcome", "actual_outcome",
    ))


def test_runner_reports_bad_tool_result_and_persistence(
    runner_control_case: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_list(arguments: Any, *, repository: TaskRepository) -> AgentResponse:
        with repository.connection() as connection:
            connection.execute(
                "INSERT INTO tasks (title, start_time, end_time, priority, type) "
                "VALUES ('Unexpected', '2026-11-02T09:00:00', '2026-11-02T10:00:00', 'low', 'fixed')"
            )
        return AgentResponse(success=False, intent="LIST_TASKS", tool_called="list_tasks", message="Broken control", error="STORAGE_ERROR")

    monkeypatch.setitem(TOOLS, "list_tasks", broken_list)
    failures, report = _evaluate(runner_control_case, _control_prediction, TaskRepository(tmp_path / "control.sqlite3"))
    assert "Outcome success mismatch" in failures, report
    assert "Outcome error mismatch" in failures, report
    assert "Final persisted state mismatch" in failures, report


@pytest.mark.parametrize(("tool_name", "intent", "state_change"), [
    ("create_task", "CREATE_TASK", "create_one"),
    ("update_task", "UPDATE_TASK", "update_one_preserving_id"),
    ("delete_task", "DELETE_TASK", "delete_only_task_101"),
])
def test_runner_checks_successful_write_paths(
    runner_control_case: dict[str, Any], tmp_path: Path,
    tool_name: str, intent: str, state_change: str,
) -> None:
    task = {
        "title": "Runner control", "start_time": "2026-11-02T09:00:00",
        "end_time": "2026-11-02T10:00:00", "priority": "low", "type": "fixed",
    }
    initial = [] if tool_name == "create_task" else [{**task, "id": 101}]
    arguments = {"task_id": 101} if tool_name == "delete_task" else {**task, "title": "Written control"}
    if tool_name == "update_task":
        arguments["task_id"] = 101
    case = {
        **runner_control_case, "expected_intent": intent, "expected_tool": tool_name,
        "expected_arguments": arguments,
        "expected_outcome": {
            "initial_tasks": initial, "success": True, "error": None,
            "state_change": state_change, "requires_clarification": False,
        },
    }

    def predict(request: AgentRequest) -> dict[str, Any]:
        assert request.model_dump(mode="json") == {
            "message": "List all tasks.", "context": {"tasks": initial},
        }
        return {
            "decision": {
                "intent": intent, "confidence": 1.0, "reasoning_summary": "Synthetic write-path control",
                "tool_name": tool_name, "tool_arguments": arguments.copy(),
            },
            "should_execute": True, "requires_clarification": False,
        }

    failures, report = _evaluate(case, predict, TaskRepository(tmp_path / "control.sqlite3"))
    assert failures == [], report


@pytest.mark.parametrize("invalid_response", [False, True])
def test_runner_checks_declined_execution_schema_and_outcome(
    runner_control_case: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    invalid_response: bool,
) -> None:
    case = {
        **runner_control_case, "expected_intent": "UNKNOWN", "expected_tool": None,
        "should_execute": False,
        "expected_outcome": {
            "initial_tasks": [], "success": False, "error": "CLARIFICATION_REQUIRED",
            "state_change": "none", "requires_clarification": True,
        },
    }
    calls: list[str] = []

    def unexpected_call(*args: Any, **kwargs: Any) -> AgentResponse:
        calls.append("unexpected")
        raise AssertionError("Declined execution must not invoke any tool")

    for tool_name in TOOLS:
        monkeypatch.setitem(TOOLS, tool_name, unexpected_call)

    def predict(request: AgentRequest) -> dict[str, Any]:
        return {
            "decision": {
                "intent": "UNKNOWN", "confidence": 0.1, "reasoning_summary": "Synthetic declined control",
                "tool_name": None, "tool_arguments": {},
            },
            "should_execute": False, "requires_clarification": True,
            "response": {
                "success": "false" if invalid_response else False, "intent": "UNKNOWN",
                "tool_called": None, "message": "Please clarify.", "error": "CLARIFICATION_REQUIRED",
            },
        }

    failures, report = _evaluate(case, predict, TaskRepository(tmp_path / "control.sqlite3"))
    assert calls == []
    if invalid_response:
        assert any("ValidationError" in failure for failure in failures), report
    else:
        assert failures == [], report