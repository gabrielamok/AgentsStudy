# AI-103 Study: Agents

Scope: Project-defined agent concepts, not a complete Microsoft exam syllabus.
Sources: [Daily Planner specification](../.github/agents/daily-planner.agent.md),
[agent contracts](../agent/contracts.py), [planning rules](planning-rules.md),
and [application routes](../api/main.py).

## Concepts

- An agent interprets a request within defined responsibilities and constraints.
- Structured contracts separate requests, decisions, tool calls, and responses.
- Tools perform application operations; interpreting intent does not itself persist data.
- The planner separates intent extraction from deterministic scheduling and storage.

## Definitions

| Term | Project Definition |
| --- | --- |
| AgentRequest | A message with optional contextual JSON data |
| Intent | CREATE_TASK, UPDATE_TASK, DELETE_TASK, LIST_TASKS, CHECK_CONFLICTS, or UNKNOWN |
| AgentDecision | Intent, confidence, reasoning summary, optional tool name, and arguments |
| ToolCall | A named operation with structured arguments |
| AgentResponse | Success status, intent, optional called tool, message, data, and error |
| Proposal | A suggested schedule that is not yet persisted |

## Decision Rules

1. Keep fixed appointments unchanged and do not invent appointments or missing details.
2. Identify an existing task explicitly before modifying or deleting it.
3. Keep scheduling and overlap decisions in application code, not model output.
4. Validate structured intent before scheduling and validate accepted tasks before storage.
5. Distinguish a proposal from a successful persisted write in the response.
6. Do not equate contract definitions with an implemented agent workflow. The current
   service exposes schedule planning; the CRUD evaluation adapter remains absent.

## Examples

- A request for AI-103 Study with an explicit duration can produce an unsaved
  proposal. Accepting it is a separate task-creation operation.
- "Delete that task" does not identify a target. Request clarification instead
  of choosing a task by recency or title.
- Adjacent intervals, such as 09:00-10:00 and 10:00-11:00, do not overlap under
  the planner's half-open interval rule.

## Common Mistakes

- Treating generated intent as permission to bypass application validation.
- Reporting a task as saved when only a proposal exists.
- Assuming Pydantic contract classes implement natural-language CRUD decisions.
- Moving existing appointments to make a generated proposal fit.

## Interview Explanation

"This project separates interpretation from execution. The planner converts a
request into validated task intent, while deterministic code handles scheduling
and transactional conflict checks. Proposals remain unsaved until accepted.
Structured CRUD contracts exist, but their decision adapter is not implemented."

## Keywords

agent, intent, AgentRequest, AgentDecision, ToolCall, AgentResponse, structured
output, tool execution, deterministic scheduling, proposal, persistence