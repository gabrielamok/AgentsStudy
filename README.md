# AI Daily Planner

A daily planner with plain HTML/CSS/JavaScript, FastAPI, Pydantic validation, and
Python's built-in SQLite driver. Choose offline deterministic interpretation or
Azure OpenAI intent extraction. Both use the same deterministic scheduler.
No frontend build step or ORM is needed. Offline mode requires no credentials.

## Run

Python 3.11+ is required. From the project root on Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

On macOS/Linux, replace `.venv\Scripts\python.exe` with `.venv/bin/python`.
Open http://127.0.0.1:8000. Interactive API documentation is at
http://127.0.0.1:8000/docs. The frontend is served by FastAPI on the same origin.
Optional Google Fonts fall back to local fonts if unavailable.

SQLite is created automatically at `data/planner.sqlite3`. Override its location
with the `PLANNER_DB_PATH` environment variable before starting the server.
This is a local, single-user application without authentication; do not expose it
publicly without adding access controls.

## Structure

```text
agent/       AgentService, offline/Azure interpretation, deterministic scheduler
api/         Validated models, transactional SQLite repository, FastAPI routes
frontend/    HTML, CSS, and browser JavaScript
tests/       Standard-library unittest tests with temporary databases
knowledge/   Explicit planning rules
```

## Tasks And API

| Method | Route | Behavior |
| --- | --- | --- |
| POST | /tasks | Create a task; returns 201 |
| GET | /tasks | List all tasks sorted by start time |
| PUT | /tasks/{id} | Replace a task; all fields required |
| DELETE | /tasks/{id} | Delete an identified task; returns 204 |
| POST | /planner | Generate an unsaved proposal against existing tasks |

Example task payload:

```json
{
  "title": "Review pull request",
  "start_time": "2026-10-28T09:00:00",
  "end_time": "2026-10-28T10:00:00",
  "priority": "high",
  "type": "flexible"
}
```

Priority is `high`, `medium`, or `low`; type is `fixed` or `flexible`. Titles must
contain 1-200 non-whitespace characters. Start/end must be local ISO datetimes
without timezone offsets, on the same date, with end strictly after start.
Invalid payloads return 422, missing tasks 404, conflicting writes 409 (with the
conflicting task records), and database failures 503. Adjacent tasks are allowed.
Updates never silently shift other tasks. The UI identifies tasks before edits
and confirms the task title and times before deletion.

## RAG API

RAG uses the local `knowledge/` documents, an in-memory SQLite FTS5 index, and
extractive answers. It does not call Azure or a language model and does not
modify Planner tasks. Orchestration remains in
[agent/rag/service.py](agent/rag/service.py); routes only validate and delegate.

| Method | Route | Behavior |
| --- | --- | --- |
| POST | /rag/query | Retrieve matching chunks and return an extractive, cited answer |
| POST | /rag/reindex | Rebuild the index from trusted documents; no body required |
| GET | /rag/health | Report availability, index state, and indexed chunk count |

Example query payload:

```json
{
  "query": "What is the difference between RAG and fine-tuning?",
  "top_k": 3
}
```

`query` must be nonblank and `top_k` a positive integer. Optional `filters`
defaults to `{}` and matches metadata exactly, for example
`{"section": "Definitions"}`. Unknown request fields and invalid types return
422. The response includes `success`, `query`, `answer`, `retrieved_chunks`,
`citations`, `grounded`, `insufficient_context`, and nullable `error`.
Results contain chunk IDs, source titles, sections, content, and lexical scores,
not internal file-system paths or ingestion metadata. Scores are ranking
magnitudes, not confidence estimates.

No matches return 200 with `success=true`, `grounded=false`,
`insufficient_context=true`, and empty results and citations. Processing failures
return 503 with a sanitized error, never a stack trace. A handled query failure
retains the query response contract; unexpected failures return a generic
`detail` message.

Health returns `success`, `indexed`, `indexed_chunks`, and nullable `error`.
Before the first query or explicit reindex, health reports `indexed=false` and
zero chunks without building an index. Reindex returns `success`,
`indexed_chunks`, and nullable `error`; a failed rebuild preserves the prior
index and returns 503. The index is process-local, reset on restart, and closed
at shutdown. Queries and rebuilds serialize index access across worker threads.

### Local RAG Tracing

[agent/rag/tracing.py](agent/rag/tracing.py) emits one structured record per
workflow, separate from retrieval and grounding. The default local sink uses
the `agent.rag.tracing` logger at INFO with the record in `LogRecord.rag_trace`;
enable INFO and a formatter/handler that retains that structured attribute to
persist records. It makes no network calls and writes no trace files by itself.
Inject `RAGTracer(sink, correlation_id_factory=..., redactor=...)` into
`RAGService(tracer=...)` for deterministic collection in tests.

Records contain correlation ID, casefolded/whitespace-normalized redacted query,
retrieval strategy, requested `top_k`, candidate IDs/scores in ranked order,
final IDs/scores, grounding/insufficiency flags, citation IDs, processing outcome
and failure layer. No answer/chunk bodies, filters, exception messages, stack
traces, credentials configuration or environment snapshots are captured.
The lexical candidate pool is all positive, filter-matching results before
`top_k`. Custom retrievers can call `capture_candidates` to observe their pool;
otherwise `candidate_scope=retrieval_output` explicitly denotes only the results
they exposed. Set `retrieval_strategy` explicitly for custom implementations.
Trace failures do not change RAG responses. State is context-local and no clock
is used, allowing reproducible records with injected IDs.

`POST /rag/query` accepts an optional `X-Correlation-ID` header (1-128 ASCII
letters, digits, dot, underscore or hyphen) and returns the sanitized/generated
ID in that header, including handled processing failures. The body contract is
unchanged. This is diagnostic correlation, not authentication.

The redactor removes internal absolute paths, labelled credentials, bearer
tokens, URL credentials and assignment values, including environment-style
assignments. It never reads environment variables. Register application secrets
as `TraceRedactor(secret_values=[...])` before querying; registered values are
redacted even when unlabelled. Long opaque query tokens are masked too. No
heuristic can identify every arbitrary unlabelled short secret, so do not submit
credentials in queries or correlation IDs. Retain traces only in trusted local
storage with appropriate access controls.

## Natural-Language Planning

Send JSON to `/planner`:

```json
{
  "request": "Study AI-103 for 1 hour; Review PR for 30 minutes before 11 AM [high priority]; Lunch from 12 PM to 1 PM; Project meeting from 3 PM to 4 PM",
  "date": "2026-10-28",
  "work_start": "09:00",
  "work_end": "17:00",
  "buffer_minutes": 10
}
```

The result contains `date`, proposed `tasks`, and `warnings`. Nothing is saved
until explicitly accepted through `POST /tasks` or the UI's Add to schedule action.

In offline mode, supported English forms are separated by newlines, semicolons,
or sentences:

- `Title from 12 PM to 1 PM` or `Title from 15:00 to 16:00` for fixed tasks.
- `Title for 1 hour` or `Title for 30 minutes` for flexible tasks.
- `Title for 30 minutes before 11 AM` for a deadline.
- Append `[high priority]`, `[medium priority]`, or `[low priority]` (default medium).
- An optional `Tomorrow I need to:` heading is accepted. If `date` is omitted,
  tomorrow is resolved using the server's local date; otherwise today is used.
  An explicit `date` takes precedence; the UI always supplies the selected day.

Defaults are 09:00-17:00 with 10-minute buffers. Fixed ranges can sit outside the
flexible work window. Existing tasks are protected. The planner uses a simple
priority-first greedy algorithm, not a global optimizer. Missing durations are
reported, never invented. General conversation, arbitrary dates in prose, combined
hour/minute durations, recurrence, timezone conversion, and cross-midnight tasks
are not supported in offline mode. Azure mode accepts broader natural language,
including combined durations, but is still limited to single-day local-time
planning. See [planning rules](knowledge/planning-rules.md).

## Azure AI Integration

The provider is selected when the application starts. Offline mode is the
default (`PLANNER_AGENT_PROVIDER=deterministic`). To use Azure, provision an
Azure OpenAI resource and deploy a model supporting strict structured outputs,
such as `gpt-4o` version `2024-08-06` or a supported newer model. The deployment
name is your Azure deployment name, not necessarily the underlying model name.

Set these environment variables in the server process:

| Variable | Value |
| --- | --- |
| `PLANNER_AGENT_PROVIDER` | `azure` |
| `AZURE_OPENAI_ENDPOINT` | HTTPS resource origin, e.g. `https://YOUR-RESOURCE.openai.azure.com` |
| `AZURE_OPENAI_DEPLOYMENT` | Your structured-output-capable deployment name |
| `AZURE_OPENAI_API_KEY` | Resource key, injected securely via environment/secret manager |

After securely configuring `AZURE_OPENAI_API_KEY`, PowerShell startup looks like:

```powershell
$env:PLANNER_AGENT_PROVIDER = "azure"
$env:AZURE_OPENAI_ENDPOINT = "https://YOUR-RESOURCE.openai.azure.com"
$env:AZURE_OPENAI_DEPLOYMENT = "YOUR-DEPLOYMENT"
.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Never put keys in source, frontend code, planning requests, or committed files.
The application reads the process environment; it does not automatically load
`.env` files. Environment files are ignored by Git. Missing/invalid Azure settings
fail startup rather than silently falling back to offline mode. Switch providers
by changing the environment and restarting the server.

`AzureAgentService` in [agent/service.py](agent/service.py) uses the official
OpenAI client against Azure's `/openai/v1/chat/completions` endpoint; no dated API
version setting is needed. The model receives only the planning request, selected
date, and server reference date, not saved tasks or credentials. Request text is
sent to your configured Azure resource, so account for its data-handling policies.

The model must return strict JSON-schema task intent: titles, priority/type,
explicit fixed times, duration, and optional deadline. Pydantic then validates
required fields, types, enums, lengths, time semantics, and the selected date.
Flexible task slots, IDs, conflict flags, and arbitrary extra fields are rejected.
Missing details produce deterministic warnings rather than invented durations.

Only application code assigns flexible slots and checks overlaps, deadlines, and
buffers against both existing and proposed tasks. The model never decides overlap
or sees the saved calendar. Proposals remain unsaved; accepting one still goes
through SQLite's transactional conflict check. Schema validation cannot prove
that a model interpreted every phrase correctly, so review proposals before saving.

Provider calls use a 30-second timeout (5-second connect timeout), an 8192-token
completion cap, and no automatic retries. Refused, truncated, or invalid model
output returns a sanitized 502. Provider/network errors return a sanitized 503.
Raw provider responses, prompts, and credentials are not included in these errors.
Provider-specific prompts, configuration, calls, and validation live in
[agent/service.py](agent/service.py); API routes only delegate to `AgentService`.
Tests can inject an agent with `create_app(agent_service=...)`.

See Microsoft's [structured output documentation](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/structured-outputs)
for compatible deployments and schema restrictions.

## Tests

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover CRUD, interval validation, overlap rejection, adjacent tasks, updates,
SQLite persistence, deadline planning, missing details, and non-mutating proposals.
Azure tests use the real SDK with a mocked HTTP transport: no credentials, network
calls, or Azure charges. They cover strict schemas, invalid output, refusals,
truncation, authentication/rate/server failures, timeouts, environment settings,
and application-owned conflict checks. Databases are temporary and isolated.

## Browser E2E Tests

Playwright with TypeScript covers task creation, today's task list, overlap
rejection, updates, deletion, natural-language planning, and invalid AI output.
Use Node.js 22+ and Python with the project requirements installed.

```powershell
npm ci
npx playwright install chromium
$env:PLAYWRIGHT_PYTHON = (Resolve-Path .venv\Scripts\python.exe).Path
npm run typecheck
npm run test:e2e
```

`PLAYWRIGHT_PYTHON` is optional if `python` on PATH already has the project
dependencies. On macOS/Linux, use `export PLAYWRIGHT_PYTHON="$PWD/.venv/bin/python"`.
For interactive testing and reports:

```powershell
npm run test:e2e:ui
npm run test:e2e:report
```

The suite launches its own FastAPI server on `http://127.0.0.1:8765` and refuses
to reuse an occupied port. Direct process management avoids shell-tree teardown
issues on Windows. The normal application on port 8000 and its database are not
used. Each run gets a temporary SQLite database, with identified test records
removed before each test; tests run sequentially with one worker. Cleanup stops
the test server and removes its database. Browser time is fixed in UTC so today's
task filtering is reproducible.

The test-only app in [tests/e2e_app.py](tests/e2e_app.py) uses the real Azure adapter
and OpenAI SDK with `httpx.MockTransport`. Successful intent is controlled JSON;
the `invalid-ai-output` fixture request produces malformed provider JSON that must
be rejected by backend validation with a sanitized 502. These are not live-model
quality tests, and no Azure credentials, calls, or charges are involved. They test
UI behavior, API status codes, validated fields, persistence, conflicts, proposal
acceptance, and error recovery without asserting exact LLM wording. Test-only
behavior is never mounted in the production application.

Configuration is in [playwright.config.ts](playwright.config.ts):

- Screenshots are captured only on failure.
- One retry is enabled, with a trace recorded on that first retry.
- An HTML report is generated at `playwright-report/index.html`.
- Failure screenshots and traces are saved under `test-results/`.

Generated reports, browser artifacts, and `node_modules/` are ignored by Git.

## Azure DevOps CI

Create an Azure DevOps YAML pipeline pointing to
[azure-pipelines.yml](azure-pipelines.yml). It uses an Ubuntu hosted agent,
Python 3.14, and Node.js 22. CI and PR triggers target `main`; change the branch
filters if your default branch differs. For Azure Repos Git, configure a build
validation branch policy for PRs; YAML `pr` triggers apply to supported external
repositories such as GitHub.

The pipeline has eleven stages and does not deploy:

| Stage | Gate |
| --- | --- |
| Build | Python compilation, locked Node installation, TypeScript checks |
| UnitTests | Pytest tool, contract, persistence, scheduler, and mocked-provider tests |
| APITests | Pytest API tests with isolated TestClient databases |
| AgentEvaluations | Agent dataset runner, separately configured decision adapter |
| SafetyEvaluations | Safety behavioral runner, required rather than silently skipped |
| PlaywrightE2E | Planner Chromium scenarios with mocked AI and an isolated server |
| RAGUnitTests | Local contracts, loader, chunker, lexical/vector/hybrid retrieval, reranker, service, tracing, and RAG API |
| RAGGoldenEvaluations | Frozen golden dataset plus lexical/vector/hybrid/reranking comparisons; original failures retained |
| RAGSafetyEvaluations | Frozen RAG safety dataset and generated safety baseline |
| RAGE2E | Required RAG study Chromium spec; missing spec or results fail the stage |
| PublishResults | JUnit publication for each suite, including after failures |

All nine test stages depend on Build and can run independently after it succeeds.
A failing evaluation does not prevent deterministic tests or E2E from running.
Every job installs its own dependencies; hosted stage workspaces are not shared.
The reusable Python steps are in [.azure-pipelines/pytest-steps.yml](.azure-pipelines/pytest-steps.yml).
They install `requirements-dev.txt` and publish pytest JUnit XML as a per-suite
pipeline artifact even when the suite fails. No `continueOnError` is used.
Deterministic tests, setup errors, missing results, and flaky E2E retries fail CI.
AI evaluation failures also fail CI, but remain separate test runs and stages.

All four RAG stages use local deterministic components and require no Azure
variables, credentials, or live AI services. RAG unit/API tests remain separate
from golden and safety quality evaluations. Each RAG stage publishes its own
JUnit run with `always()` and fails for failed tests, missing XML, or publication
errors; PublishResults republishes the downloaded suite artifacts afterward.
The golden artifact includes a current 30-case baseline summary derived from
JUnit (excluding comparison/integrity tests and never counting skipped cases as
passes). Golden and safety artifacts also include the available `ci-results/`
baseline and comparison reports, including failure diagnostics.

RAGE2E targets only `tests/e2e/rag-study.spec.ts`, not the Planner suite, and uses
the existing local server setup. That spec and the RAG browser UI are not yet
implemented, so this stage currently fails explicitly rather than claiming RAG
browser coverage. No placeholder tests or UI changes are introduced by this
pipeline update. Its `rag-playwright` artifact retains HTML, XML, first-retry
traces, and failure-only screenshots using the unchanged Playwright settings.

Configure `PLANNER_EVAL_AGENT` as a pipeline variable containing the importable
`package.module:function` decision adapter required by the evaluation runner.
When that adapter uses Azure, configure `AZURE_OPENAI_ENDPOINT`,
`AZURE_OPENAI_DEPLOYMENT`, and secret `AZURE_OPENAI_API_KEY` through pipeline
variables or an authorized variable group. Only evaluation steps receive these
values; no credentials are needed by deterministic tests or browser tests.
Do not print secrets or put their values in YAML.

Two evaluation prerequisites are currently unresolved: no production CRUD
decision adapter exists, and `tests/evals/test_safety_evals.py` has not been
implemented. Agent evaluations therefore fail without an adapter, and the safety
stage explicitly reports its missing runner and fails. Merely parsing
`safety-evals.json` is not counted as executing safety evaluations. The pipeline
does not manufacture passes or change either dataset to work around these gaps.

Playwright installs Chromium and its Linux dependencies. Its global setup owns
the temporary FastAPI server on port 8765 and cleans up its temporary database;
CI does not start or touch the normal application on port 8000.

The PublishResults stage runs with `always()`, downloads available artifacts,
and publishes JUnit to Azure DevOps **Tests**. Missing reports and failed test
results fail publication rather than being ignored. The `playwright` pipeline
artifact contains the HTML report and test outputs, including failure screenshots,
first-retry traces, and Playwright JUnit XML. Its collection and upload also use
`always()`, so partial failure evidence is retained. Hosted-run execution must
be verified in your Azure DevOps organization after creating the pipeline.

### Pipeline Variables And Secrets

Deterministic unit, API, and browser jobs require no Azure credentials and never
make live Azure calls. Configure the following variables for the separate AI
evaluation jobs in Azure DevOps:

| Pipeline Variable | Configuration |
| --- | --- |
| `PLANNER_EVAL_AGENT` | Importable `package.module:function` decision adapter |
| `AZURE_OPENAI_ENDPOINT` | HTTPS resource origin |
| `AZURE_OPENAI_DEPLOYMENT` | Structured-output-capable deployment name |
| `AZURE_OPENAI_API_KEY` | Mark as secret; obtain from your approved secret store |

Variables can be managed through the pipeline UI or an authorized variable group.
The YAML explicitly maps them into evaluation-step process environments; it never
stores or prints credential values. Do not commit credentials or add secret
values to queue parameters. An adapter that calls Azure may incur charges;
mocked-provider unit tests and browser tests remain reproducible and charge-free.