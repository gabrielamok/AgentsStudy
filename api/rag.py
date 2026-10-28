import logging
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Response
from pydantic import Field, JsonValue

from agent.rag.contracts import RAGHealthResponse, RAGReindexResponse, RAGRequest, RAGResponse
from agent.rag.service import RAGService


logger = logging.getLogger(__name__)


class RAGQueryRequest(RAGRequest):
    filters: dict[str, JsonValue] = Field(default_factory=dict)


def create_rag_router(service: RAGService) -> APIRouter:
    router = APIRouter(prefix="/rag", tags=["RAG"])

    @router.post("/query", response_model=RAGResponse)
    def query(
        request: RAGQueryRequest, response: Response,
        correlation_id: Annotated[str | None, Header(alias="X-Correlation-ID", max_length=128, pattern=r"^[A-Za-z0-9._-]+$")] = None,
    ) -> RAGResponse:
        identifier = service.tracer.correlation_id(correlation_id)
        response.headers["X-Correlation-ID"] = identifier
        try:
            result = RAGResponse.model_validate(service.query(request, correlation_id=identifier).model_dump())
            if not result.success:
                response.status_code = 503
            return result
        except Exception as error:
            logger.error("RAG query failed (%s)", type(error).__name__)
            raise HTTPException(
                status_code=503, detail="RAG query unavailable. Please try again.",
                headers={"X-Correlation-ID": identifier},
            ) from None

    @router.post("/reindex", response_model=RAGReindexResponse)
    def reindex() -> RAGReindexResponse:
        try:
            return RAGReindexResponse(
                success=True, indexed_chunks=service.index_documents(), error=None,
            )
        except Exception as error:
            logger.error("RAG reindex failed (%s)", type(error).__name__)
            raise HTTPException(status_code=503, detail="RAG reindex unavailable. Please try again.") from None

    @router.get("/health", response_model=RAGHealthResponse)
    def health() -> RAGHealthResponse:
        try:
            return RAGHealthResponse.model_validate(service.health().model_dump())
        except Exception as error:
            logger.error("RAG health check failed (%s)", type(error).__name__)
            raise HTTPException(status_code=503, detail="RAG service unavailable. Please try again.") from None

    return router