"""Internal FastAPI routes for per-user hybrid RAG document management."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status

from superassist.config import Settings
from superassist.rag.service import HybridRAGService


def register_rag_routes(app: FastAPI, service: HybridRAGService, settings: Settings) -> None:
    @app.get("/internal/rag/graph")
    def get_rag_graph(user_id: str = Query(...)) -> dict[str, Any]:
        return service.graph_payload(user_id)

    @app.get("/internal/rag/documents")
    def list_rag_documents(user_id: str = Query(...)) -> dict[str, Any]:
        return {
            "documents": service.list_documents(user_id),
            "supported_extensions": service.supported_extensions,
            "limits": {
                "max_file_size_mb": settings.rag_max_file_size_mb,
                "max_files_per_batch": settings.rag_max_files_per_batch,
            },
        }

    @app.post("/internal/rag/documents", status_code=status.HTTP_202_ACCEPTED)
    async def upload_rag_documents(
        user_id: str = Query(...),
        files: list[UploadFile] = File(...),
    ) -> dict[str, Any]:
        if not files:
            raise HTTPException(status_code=400, detail="At least one file is required")
        if len(files) > settings.rag_max_files_per_batch:
            raise HTTPException(
                status_code=400,
                detail=f"At most {settings.rag_max_files_per_batch} files can be uploaded at once",
            )

        documents: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        max_bytes = settings.rag_max_file_size_mb * 1024 * 1024
        for upload in files:
            filename = upload.filename or "document"
            try:
                content = await upload.read(max_bytes + 1)
                if len(content) > max_bytes:
                    raise ValueError(f"File exceeds {settings.rag_max_file_size_mb} MB limit")
                documents.append(service.add_document(user_id, filename, content))
            except ValueError as exc:
                errors.append({"name": filename, "error": str(exc)})
            finally:
                await upload.close()

        if not documents and errors:
            raise HTTPException(status_code=400, detail=errors)
        return {"documents": documents, "errors": errors}

    @app.delete("/internal/rag/documents/{document_id}", status_code=status.HTTP_202_ACCEPTED)
    def delete_rag_document(document_id: str, user_id: str = Query(...)) -> dict[str, Any]:
        try:
            return service.delete_document(user_id, document_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Document not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


__all__ = ["register_rag_routes"]
