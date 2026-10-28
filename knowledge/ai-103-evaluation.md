# AI-103 Study: Evaluation

Scope: Evaluation concepts and gates already defined by this project.
Sources: [agent evaluation runner](../tests/evals/test_agent_evals.py),
[planner cases](../tests/evals/planner-evals.json),
[RAG specification](../.github/agents/rag-study-agent.agent.md),
and [browser scenarios](../tests/e2e/planner.spec.ts).

## Concepts

- Deterministic checks verify behavior that can be compared exactly.
- Agent evaluations compare actual decisions and outcomes with fixed expectations.
- Browser E2E tests verify observable workflows and resulting persisted state.
- RAG evaluation must assess both retrieval and the answer supported by its results.
- A dataset or runner-control test is not evidence that an agent was evaluated.

## Definitions

| Term | Project Definition |
| --- | --- |
| Evaluation case | A user input with a stable ID and expected behavior |
| Intent correctness | Agreement between expected and actual classified intent |
| Tool correctness | Agreement on tool selection, arguments, and execution behavior |
| Schema validation | Checking structured output against the required contract |
| Groundedness | Whether factual answer claims are supported by retrieved knowledge |
| Answer relevance | Whether an answer addresses the user's question |
| Retrieval relevance | Whether retrieved content is relevant to the question |
| Regression | Previously verified behavior failing after a change |

## Decision Rules

1. Check intent, tool selection, arguments, should_execute behavior, output schema,
   tool result, and final persisted state where applicable.
2. Report expected and actual values with the case ID and user input on failure.
3. Keep expected datasets fixed; do not weaken assertions to manufacture passes.
4. Separate deterministic and mocked-provider tests from actual agent evaluations.
5. Use isolated databases and verify that read-only or rejected requests preserve state.
6. Validate RAG ingestion, chunk quality, retrieval relevance, attribution, groundedness,
   unsupported claims, no-answer behavior, and prompt injection resilience separately.
7. Report failures and unexecuted coverage distinctly. Missing adapters or runners
   prevent behavioral conclusions; parsing dataset JSON is not a safety evaluation.

## Examples

- For an overlapping create request, expect SCHEDULE_CONFLICT and unchanged storage,
  not silent rescheduling. Check the real tool result and persisted records.
- A Playwright test creates a task through the UI and queries the API to verify
  persistence. It need not compare exact model-generated wording.
- A mocked provider returning controlled JSON tests integration and validation,
  not the interpretation quality of a live model.
- A proposed RAG no-answer test asks about content absent from the trusted sources
  and checks that the agent acknowledges insufficient evidence without false citations.

## Common Mistakes

- Counting runner self-tests as successful dataset evaluations.
- Counting unexecuted safety cases as passes or hiding failed cases.
- Checking only a fluent answer while ignoring tool calls and storage changes.
- Repeating unit-level field assertions unnecessarily in browser tests.
- Inferring model-quality or RAG-quality results from mocked integration tests.

## Interview Explanation

"I separate exact application checks from agent and RAG evaluations. Agent cases
compare decisions, execution, and persisted state against unchanged expectations.
RAG checks also examine evidence relevance and claim support. Missing evaluation
infrastructure is reported as a coverage limitation, never converted into a pass."

## Keywords

evaluation case, deterministic testing, intent correctness, tool selection,
tool arguments, schema validation, persistence, groundedness, relevance,
no-answer behavior, prompt injection resilience, regression, E2E