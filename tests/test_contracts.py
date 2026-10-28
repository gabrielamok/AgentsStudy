import unittest

from pydantic import ValidationError

from agent.contracts import AgentDecision, AgentRequest, AgentResponse, Intent, Task, ToolCall
from agent.rag.contracts import (
    DocumentChunk, GroundedAnswer, KnowledgeDocument, RAGRequest, RAGResponse,
    RetrievalQuery, RetrievalResult, SourceCitation,
)


class ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = {
            "title": "Study AI-103", "start_time": "2026-10-28T09:00:00",
            "end_time": "2026-10-28T10:00:00", "priority": "high", "type": "flexible",
        }

    def test_exact_supported_intents(self) -> None:
        self.assertEqual({intent.value for intent in Intent}, {
            "CREATE_TASK", "UPDATE_TASK", "DELETE_TASK", "LIST_TASKS", "CHECK_CONFLICTS", "UNKNOWN",
        })
        for intent in Intent:
            decision = AgentDecision(intent=intent, confidence=0.8, reasoning_summary="Intent classification")
            self.assertEqual(AgentDecision.model_validate_json(decision.model_dump_json()), decision)
        with self.assertRaises(ValidationError):
            AgentDecision(intent="UNSUPPORTED", confidence=0.8, reasoning_summary="Invalid intent")

    def test_confidence_bounds_and_finite_values(self) -> None:
        for confidence in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(confidence=confidence), self.assertRaises(ValidationError):
                AgentDecision(intent=Intent.UNKNOWN, confidence=confidence, reasoning_summary="Unclear request")
        for confidence in (0, 1):
            self.assertEqual(AgentDecision(intent=Intent.UNKNOWN, confidence=confidence, reasoning_summary="Unclear request").confidence, confidence)

    def test_request_and_tool_payloads_are_json_serializable(self) -> None:
        request = AgentRequest(message="  Create a study task  ", context={"tasks": [self.task]})
        self.assertEqual(request.message, "Create a study task")
        call = ToolCall(tool_name="create_task", tool_arguments=self.task)
        self.assertEqual(ToolCall.model_validate_json(call.model_dump_json()), call)
        with self.assertRaises(ValidationError):
            ToolCall(tool_name="create_task", tool_arguments={"invalid": object()})
        for payload in ({"message": " "}, {"message": "Plan", "extra": True}):
            with self.assertRaises(ValidationError):
                AgentRequest.model_validate(payload)

    def test_mutable_defaults_are_independent(self) -> None:
        first = AgentRequest(message="First")
        second = AgentRequest(message="Second")
        first.context["changed"] = True
        self.assertEqual(second.context, {})
        first_call = ToolCall(tool_name="list_tasks")
        second_call = ToolCall(tool_name="list_tasks")
        first_call.tool_arguments["date"] = "2026-10-28"
        self.assertEqual(second_call.tool_arguments, {})

    def test_task_reuses_existing_validation_and_supports_optional_id(self) -> None:
        self.assertIsNone(Task.model_validate(self.task).id)
        self.assertEqual(Task.model_validate({**self.task, "id": 1}).id, 1)
        for changes in ({"id": 0}, {"title": " "}, {"end_time": self.task["start_time"]}, {"priority": "urgent"}, {"start_time": "2026-10-28T09:00:00Z"}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                Task.model_validate({**self.task, **changes})

    def test_response_success_and_error_shapes(self) -> None:
        task = Task.model_validate({**self.task, "id": 1})
        response = AgentResponse(success=True, intent=Intent.CREATE_TASK, tool_called="create_task", message="Task created", data=task.model_dump(mode="json"))
        self.assertEqual(AgentResponse.model_validate_json(response.model_dump_json()), response)
        self.assertIsNone(response.error)
        failure = AgentResponse(success=False, intent=Intent.UNKNOWN, message="Request not understood", error="Unknown intent")
        self.assertIsNone(failure.tool_called)
        self.assertIsNone(failure.data)
        self.assertEqual(set(response.model_dump()), {"success", "intent", "tool_called", "message", "data", "error"})


class RAGContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.citation = {"source_title": "Planning rules", "section": "Decision Rules", "chunk_id": "rules-1"}
        self.result = {**self.citation, "content": "Proposals are not persisted.", "score": 2.5}
        self.query = {"query": "Are proposals persisted?", "top_k": 3, "filters": {"domain": "agents"}}
        self.payloads = {
            KnowledgeDocument: {
                "document_id": "rules", "title": "Planning rules", "source_path": "knowledge/planning-rules.md",
                "content": "Proposals are not persisted.", "metadata": {"trusted": True, "tags": ["planner"]},
            },
            DocumentChunk: {**self.citation, "document_id": "rules", "content": self.result["content"], "metadata": {}},
            RetrievalQuery: self.query,
            RetrievalResult: self.result,
            SourceCitation: self.citation,
            GroundedAnswer: {
                "answer": "Proposals remain unsaved.", "citations": [self.citation],
                "grounded": True, "insufficient_context": False,
            },
            RAGRequest: self.query,
            RAGResponse: {
                "success": True, "query": self.query["query"], "answer": "Proposals remain unsaved.",
                "retrieved_chunks": [self.result], "citations": [self.citation],
                "grounded": True, "insufficient_context": False, "error": None,
            },
        }

    def test_all_contracts_round_trip_and_build_schemas(self) -> None:
        for model, payload in self.payloads.items():
            with self.subTest(model=model.__name__):
                instance = model.model_validate(payload)
                self.assertEqual(model.model_validate_json(instance.model_dump_json()), instance)
                self.assertEqual(set(model.model_json_schema()["properties"]), set(payload))
        response = RAGResponse.model_validate(self.payloads[RAGResponse])
        self.assertIsInstance(response.citations[0], SourceCitation)
        self.assertIsInstance(response.retrieved_chunks[0], RetrievalResult)

    def test_every_declared_field_is_required(self) -> None:
        for model, payload in self.payloads.items():
            for field in payload:
                with self.subTest(model=model.__name__, field=field), self.assertRaises(ValidationError):
                    model.model_validate({key: value for key, value in payload.items() if key != field})

    def test_blank_identifiers_and_other_required_text_are_rejected(self) -> None:
        for model, payload in self.payloads.items():
            for field, value in payload.items():
                if isinstance(value, str):
                    for blank in ("", " \t\n"):
                        with self.subTest(model=model.__name__, field=field, blank=blank), self.assertRaises(ValidationError):
                            model.model_validate({**payload, field: blank})
        citation = SourceCitation.model_validate({**self.citation, "chunk_id": " rules-1 "})
        self.assertEqual(citation.chunk_id, "rules-1")

    def test_top_k_requires_a_positive_integer_without_coercion(self) -> None:
        for model in (RetrievalQuery, RAGRequest):
            for top_k in (0, -1, True, "3", 1.5):
                with self.subTest(model=model.__name__, top_k=top_k), self.assertRaises(ValidationError):
                    model.model_validate({**self.query, "top_k": top_k})

    def test_scores_are_finite_without_assuming_provider_normalization(self) -> None:
        for score in (-2.0, 0.0, 2.5):
            self.assertEqual(RetrievalResult.model_validate({**self.result, "score": score}).score, score)
        for score in (float("nan"), float("inf"), -float("inf"), True, "0.5"):
            with self.subTest(score=score), self.assertRaises(ValidationError):
                RetrievalResult.model_validate({**self.result, "score": score})

    def test_extra_fields_and_non_json_metadata_are_rejected(self) -> None:
        for model, payload in self.payloads.items():
            with self.subTest(model=model.__name__), self.assertRaises(ValidationError):
                model.model_validate({**payload, "azure_deployment": "not-part-of-the-contract"})
            for field in ("metadata", "filters"):
                if field in payload:
                    with self.subTest(model=model.__name__, field=field), self.assertRaises(ValidationError):
                        model.model_validate({**payload, field: {"invalid": object()}})

    def test_nested_contracts_and_boolean_flags_are_validated(self) -> None:
        payload = self.payloads[RAGResponse]
        for changes in (
            {"success": "true"}, {"grounded": 1}, {"insufficient_context": "false"},
            {"citations": [{**self.citation, "chunk_id": " "}]},
            {"retrieved_chunks": [{**self.result, "score": float("inf")}]},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                RAGResponse.model_validate({**payload, **changes})

    def test_insufficient_context_and_failure_responses_remain_representable(self) -> None:
        payload = self.payloads[RAGResponse]
        unanswered = RAGResponse.model_validate({
            **payload, "answer": "The knowledge base does not contain enough information.",
            "retrieved_chunks": [], "citations": [], "grounded": False, "insufficient_context": True,
        })
        self.assertTrue(unanswered.insufficient_context)
        failure = RAGResponse.model_validate({**unanswered.model_dump(), "success": False, "error": "Retrieval unavailable"})
        self.assertEqual(failure.error, "Retrieval unavailable")