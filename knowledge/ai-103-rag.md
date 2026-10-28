# AI-103 Study: Retrieval-Augmented Generation

Scope: The project's specified RAG design; retrieval is not implemented yet.
Source: [RAG Study Agent specification](../.github/agents/rag-study-agent.agent.md).
These notes do not assert Microsoft service features or integration availability.

## Concepts

- RAG separates finding relevant knowledge from generating an answer.
- Trusted documents in the knowledge directory are the intended evidence base.
- Retrieved passages need meaningful context and traceable source metadata.
- A generated answer must stay within the evidence retrieved for that question.

## Definitions

| Term | Project Definition |
| --- | --- |
| Ingestion | Reading trusted source documents for processing |
| Chunk | A meaningful document portion prepared for retrieval |
| Indexing | Organizing processed knowledge for retrieval |
| Retrieval | Selecting content relevant to a user's question |
| Generation | Producing an answer from the retrieved evidence |
| Grounding | Restricting factual answers to supported retrieved knowledge |
| Source attribution | Identifying the source supporting a factual answer |
| No-answer behavior | Stating that the knowledge base has insufficient evidence |

## Decision Rules

1. Keep ingestion, chunking, indexing, retrieval, generation, citation handling,
   and evaluation separate and independently testable.
2. Split documents into meaningful chunks and preserve their source context.
3. Preserve source name, chunk identifier, and retrieval score for each result.
4. Use deterministic validation for exact document-processing rules.
5. Cite supporting sources for factual answers; do not fabricate sections or results.
6. When evidence is insufficient, state that limitation rather than guessing.
7. Keep the initial design provider-independent. The specification names Azure AI
   Search and Microsoft Foundry as future replacement options, not implemented services.

## Examples

- If a future retrieval result includes [planning rules](planning-rules.md) stating
  that proposals are not persisted, an answer may explain that acceptance is separate
  and attribute the statement to that document.
- A question about Microsoft service pricing has no supporting material in these
  project notes. The correct behavior is an explicit insufficient-evidence response.
- A result from this document's "Decision Rules" section should retain that heading
  and source identity, rather than present an isolated instruction without context.

## Common Mistakes

- Treating a relevant passage as support for claims it does not contain.
- Losing source names or section context during chunking.
- Treating a retrieval score as proof that an answer is grounded.
- Fabricating citations or describing planned retrieval as already operational.
- Mixing retrieval implementation with generation or assuming a specific provider.

## Interview Explanation

"The RAG design retrieves relevant passages from trusted project documents before
generating an answer. Each result preserves source metadata, and factual answers
must identify supporting evidence. If that evidence is missing, the agent says so.
This repository currently defines those requirements, not a working retriever."

## Keywords

RAG, ingestion, chunking, indexing, retrieval, generation, grounding, source
attribution, chunk identifier, retrieval score, citation, no-answer behavior