# Deterministic planning rules

- All timestamps are local, timezone-free ISO 8601 datetimes. A task stays within one calendar day.
- Intervals are half-open: 09:00-10:00 and 10:00-11:00 do not conflict.
- SQLite writes check conflicts inside an immediate transaction, including concurrent writers.
- Existing tasks, fixed or flexible, are never automatically moved or deleted.
- New explicit time ranges are placed first. Conflicting ranges are reported, not shifted.
- Flexible tasks are ordered by high/medium/low priority, then earliest deadline, then input order.
- Flexible tasks use the earliest free slot within the work window and before the deadline.
- The default buffer is 10 minutes around occupied time. Explicit fixed ranges are not shifted to enforce buffers.
- Missing durations, unsupported text, invalid times, and infeasible tasks produce warnings.
- Proposals are not persisted. Each accepted proposal goes through normal conflict validation.
- Offline interpretation uses a restricted English grammar. Optional Azure interpretation uses strict structured outputs.
- Azure extracts intent only; it is not given saved tasks and must not schedule flexible work or decide overlaps.
- Pydantic rejects invalid AI JSON, extra fields, invalid times, or model-assigned flexible slots before scheduling.
- Both interpretation modes use the same application-owned deterministic scheduler and transactional conflict checks.
- Azure credentials come only from the server environment and never go to the frontend or model prompt.