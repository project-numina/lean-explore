"""FastAPI server for Lean Explore API."""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Dict, List

from fastapi import FastAPI, Path, Query, Request

from lean_explore import defaults
from lean_explore.local.service import Service as LocalService
from lean_explore.shared.models.api import (
    APICitationsResponse,
    APISearchResponse,
    APISearchResultItem,
)

logger = logging.getLogger(__name__)


@dataclass
class AppContext:
    """Dataclass to hold application-level context for FastAPI app."""

    search_service: LocalService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[Dict]:
    """Lifespan context manager for FastAPI app."""
    # Pre-check for essential data files before initializing LocalService
    required_files_info = {
        "Database file": defaults.DEFAULT_DB_PATH,
        "FAISS index file": defaults.DEFAULT_FAISS_INDEX_PATH,
        "FAISS ID map file": defaults.DEFAULT_FAISS_MAP_PATH,
    }
    missing_files_messages = []
    for name, path_obj in required_files_info.items():
        if not path_obj.exists():
            missing_files_messages.append(
                f"  - {name}: Expected at {path_obj.resolve()}"
            )

    if missing_files_messages:
        expected_toolchain_dir = (
            defaults.LEAN_EXPLORE_TOOLCHAINS_BASE_DIR
            / defaults.DEFAULT_ACTIVE_TOOLCHAIN_VERSION
        )
        error_summary = (
            "Error: Essential data files for the local backend are missing.\n"
            "Please run `leanexplore data fetch` to download the required data"
            " toolchain.\n"
            f"Expected data directory for active toolchain "
            f"('{defaults.DEFAULT_ACTIVE_TOOLCHAIN_VERSION}'):"
            f" {expected_toolchain_dir.resolve()}\n"
            "Details of missing files:\n"
            + "\n".join(f"  - {msg}" for msg in missing_files_messages)
        )
        logger.error(error_summary)
        sys.exit(1)
    yield {"ctx": AppContext(search_service=LocalService())}


app = FastAPI(title="Lean Explore API", lifespan=lifespan, root_path="/api/v1")


@app.get("/heartbeat")
async def heartbeat() -> Dict[str, str]:
    """Simple heartbeat endpoint to check if the server is running."""
    return {"status": "ok"}


@app.get("/search")
async def search(
    request: Request,
    q: str = Query(..., description="Search query"),
    pkg: List[str] | None = Query(None, description="Package filters"),
    limit: int = Query(50, ge=1, description="Maximum number of results to return"),
) -> APISearchResponse:
    """Search for statement groups matching the query and optional package filters."""
    search_service: LocalService = request.state.ctx.search_service

    responses: APISearchResponse = await asyncio.to_thread(
        search_service.search, q, pkg, limit
    )
    responses.results = responses.results[:limit]

    return responses


@app.get("/statement_groups/{group_id}")
async def get_by_id(
    request: Request,
    group_id: int = Path(..., description="Statement group ID"),
) -> APISearchResultItem:
    """Retrieves specific statement groups by their unique identifier(s)."""
    search_service: LocalService = request.state.ctx.search_service
    result = await asyncio.to_thread(search_service.get_by_id, group_id=group_id)

    return result


@app.get("/statement_groups/{group_id}/dependencies")
async def get_dependencies(
    request: Request,
    group_id: int = Path(..., description="Statement group ID"),
) -> APICitationsResponse:
    """Retrieves direct dependencies (citations) for specific statement group(s)."""
    search_service: LocalService = request.state.ctx.search_service
    result = await asyncio.to_thread(search_service.get_dependencies, group_id=group_id)
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
