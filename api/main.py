import logging
import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Path as ApiPath, Request, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from agent.rag.service import RAGService
from agent.service import AgentService, AgentServiceError, PlannerRequest, PlanResult, create_agent_service
from api.models import Task, TaskInput
from api.rag import create_rag_router
from api.repository import ScheduleConflict, TaskNotFound, TaskRepository

ROOT = Path(__file__).resolve().parent.parent
TaskId = Annotated[int, ApiPath(gt=0)]


def create_app(
    database_path: Path | None = None, agent_service: AgentService | None = None, *,
    rag_service: RAGService | None = None,
) -> FastAPI:
    repository = TaskRepository(database_path or Path(os.getenv("PLANNER_DB_PATH", str(ROOT / "data" / "planner.sqlite3"))))
    service = agent_service if agent_service is not None else create_agent_service()
    rag = rag_service if rag_service is not None else RAGService()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        repository.initialize()
        try:
            yield
        finally:
            try:
                service.close()
            finally:
                rag.close()

    app = FastAPI(title="AI Daily Planner", lifespan=lifespan)

    @app.exception_handler(AgentServiceError)
    async def agent_error(request: Request, error: AgentServiceError) -> JSONResponse:
        return JSONResponse(status_code=error.status_code, content={"detail": str(error)})

    @app.exception_handler(TaskNotFound)
    async def not_found(request: Request, error: TaskNotFound) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": "Task not found"})

    @app.exception_handler(ScheduleConflict)
    async def conflict(request: Request, error: ScheduleConflict) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(error), "conflicts": [task.model_dump(mode="json") for task in error.tasks]})

    @app.exception_handler(sqlite3.Error)
    async def database_error(request: Request, error: sqlite3.Error) -> JSONResponse:
        logging.getLogger(__name__).exception("Database operation failed", exc_info=error)
        return JSONResponse(status_code=503, content={"detail": "Database unavailable. Please try again."})

    @app.post("/tasks", response_model=Task, status_code=201)
    def create_task(task: TaskInput) -> Task:
        return repository.save(task)

    @app.get("/tasks", response_model=list[Task])
    def list_tasks() -> list[Task]:
        return repository.list()

    @app.put("/tasks/{id}", response_model=Task)
    def update_task(id: TaskId, task: TaskInput) -> Task:
        return repository.save(task, id)

    @app.delete("/tasks/{id}", status_code=204)
    def delete_task(id: TaskId) -> Response:
        repository.delete(id)
        return Response(status_code=204)

    @app.post("/planner", response_model=PlanResult)
    def plan(request: PlannerRequest) -> PlanResult:
        return service.plan(request, repository.list())

    app.include_router(create_rag_router(rag))
    app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
    return app


app = create_app()