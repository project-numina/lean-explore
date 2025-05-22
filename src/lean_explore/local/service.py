# src/lean_explore/local/service.py

"""Provides a service class for local Lean data exploration.

This module defines the Service class, which offers methods to search,
retrieve by ID, and get dependencies for statement groups using local
data assets (SQLite database, FAISS index, and embedding models).
"""

import logging
import os
import pathlib
import time
from typing import List, Optional, Dict, Any

from sqlalchemy import create_engine, select, or_
from sqlalchemy.orm import sessionmaker, Session as SQLAlchemySessionType, joinedload
from sqlalchemy.exc import SQLAlchemyError

# Assuming SentenceTransformer and faiss.Index are types, not necessarily loaded here
# if search.py handles their loading and returns typed objects.
# We'll import the loader functions themselves.
from sentence_transformers import SentenceTransformer # For type hinting if needed
import faiss # For type hinting if needed

from lean_explore import defaults
from .search import (
    perform_search,
    load_embedding_model,
    load_faiss_assets
)
from lean_explore.shared.models.db import (
    StatementGroup,
    Declaration, # Imported for relationship access via StatementGroup
    StatementGroupDependency
)
from lean_explore.shared.models.api import (
    APISearchResultItem,
    APIPrimaryDeclarationInfo,
    APISearchResponse,
    APICitationsResponse
)

logger = logging.getLogger(__name__)


class Service:
    """A service for interacting with local Lean explore data.

    This service loads necessary data assets (embedding model, FAISS index,
    database connection) upon initialization using default paths and parameters.
    It provides methods for searching statement groups, retrieving them by ID,
    and fetching their dependencies (citations).

    Attributes:
        embedding_model: The loaded sentence embedding model.
        faiss_index: The loaded FAISS index.
        text_chunk_id_map: A list mapping FAISS indices to text chunk IDs.
        engine: The SQLAlchemy engine for database connections.
        SessionLocal: The SQLAlchemy sessionmaker for creating sessions.
        default_faiss_k (int): Default number of FAISS neighbors to retrieve.
        default_pagerank_weight (float): Default weight for PageRank.
        default_text_relevance_weight (float): Default weight for text relevance.
        default_name_match_weight (float): Default weight for name matching.
        default_semantic_similarity_threshold (float): Default similarity threshold.
        default_results_limit (int): Default limit for search results.
        default_faiss_nprobe (int): Default nprobe for FAISS IVF indexes.
    """

    def __init__(self):
        """Initializes the Service by loading data assets and configurations.

        Ensures the user data directory exists. Loads the embedding model,
        FAISS index, and sets up the database engine.

        Raises:
            FileNotFoundError: If essential data files (FAISS index, map, DB)
                               are not found at their default locations.
            RuntimeError: If the embedding model fails to load or if other
                          critical initialization steps fail.
        """
        logger.info("Initializing local Service...")
        try:
            defaults.LEAN_EXPLORE_USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"User data directory ensured: {defaults.LEAN_EXPLORE_USER_DATA_DIR}")
        except OSError as e:
            logger.error(f"Could not create user data directory {defaults.LEAN_EXPLORE_USER_DATA_DIR}: {e}")
            raise RuntimeError(f"Failed to create user data directory: {e}") from e

        logger.info(f"Loading embedding model: {defaults.DEFAULT_EMBEDDING_MODEL_NAME}")
        self.embedding_model: Optional[SentenceTransformer] = load_embedding_model(defaults.DEFAULT_EMBEDDING_MODEL_NAME)
        if self.embedding_model is None:
            raise RuntimeError(f"Failed to load embedding model: {defaults.DEFAULT_EMBEDDING_MODEL_NAME}")

        logger.info(f"Loading FAISS assets: Index='{defaults.DEFAULT_FAISS_INDEX_PATH}', Map='{defaults.DEFAULT_FAISS_MAP_PATH}'")
        faiss_assets = load_faiss_assets(
            str(defaults.DEFAULT_FAISS_INDEX_PATH),
            str(defaults.DEFAULT_FAISS_MAP_PATH)
        )
        if faiss_assets[0] is None or faiss_assets[1] is None:
            # load_faiss_assets logs specific file not found errors
            raise FileNotFoundError("Failed to load FAISS index or ID map. Check logs for details and ensure files exist at specified default paths.")
        self.faiss_index: faiss.Index = faiss_assets[0]
        self.text_chunk_id_map: List[str] = faiss_assets[1]

        logger.info(f"Initializing database engine with URL: {defaults.DEFAULT_DB_URL}")
        if not defaults.DEFAULT_DB_PATH.exists() and not defaults.DEFAULT_DB_URL.endswith(":memory:"):
            # This check is for file-based SQLite. For other DBs, connection failure will occur at engine creation or first use.
             logger.error(f"Database file not found at {defaults.DEFAULT_DB_PATH}. "
                         "Please ensure it has been downloaded or created.")
             raise FileNotFoundError(f"Database file not found: {defaults.DEFAULT_DB_PATH}")

        try:
            self.engine = create_engine(defaults.DEFAULT_DB_URL)
            self.SessionLocal: sessionmaker[SQLAlchemySessionType] = sessionmaker(
                autocommit=False, autoflush=False, bind=self.engine
            )
            # Test connection
            with self.engine.connect() as conn:
                logger.info("Database connection successful.")
        except Exception as e:
            logger.error(f"Failed to initialize database engine or connection: {e}", exc_info=True)
            raise RuntimeError(f"Database initialization failed: {e}") from e

        # Store default search parameters
        self.default_faiss_k: int = defaults.DEFAULT_FAISS_K
        self.default_pagerank_weight: float = defaults.DEFAULT_PAGERANK_WEIGHT
        self.default_text_relevance_weight: float = defaults.DEFAULT_TEXT_RELEVANCE_WEIGHT
        self.default_name_match_weight: float = defaults.DEFAULT_NAME_MATCH_WEIGHT
        self.default_semantic_similarity_threshold: float = defaults.DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD
        self.default_results_limit: int = defaults.DEFAULT_RESULTS_LIMIT
        self.default_faiss_nprobe: int = defaults.DEFAULT_FAISS_NPROBE
        
        logger.info("Local Service initialized successfully.")

    def _serialize_sg_to_api_item(self, sg_orm: StatementGroup) -> APISearchResultItem:
        """Converts a StatementGroup ORM object to an APISearchResultItem Pydantic model.

        Args:
            sg_orm: The SQLAlchemy StatementGroup object.

        Returns:
            An APISearchResultItem Pydantic model instance.
        """
        primary_decl_info = APIPrimaryDeclarationInfo(
            lean_name=sg_orm.primary_declaration.lean_name if sg_orm.primary_declaration else None
        )
        return APISearchResultItem(
            id=sg_orm.id,
            primary_declaration=primary_decl_info,
            source_file=sg_orm.source_file,
            range_start_line=sg_orm.range_start_line,
            display_statement_text=sg_orm.display_statement_text,
            statement_text=sg_orm.statement_text,
            docstring=sg_orm.docstring,
            informal_description=sg_orm.informal_description,
        )

    def search(
        self,
        query: str,
        package_filters: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> APISearchResponse:
        """Performs a local search for statement groups.

        Args:
            query: The search query string.
            package_filters: An optional list of package names to filter results by.
            limit: An optional limit on the number of results to return.
                   If None, defaults.DEFAULT_RESULTS_LIMIT is used.

        Returns:
            An APISearchResponse object containing search results and metadata.
        """
        start_time = time.time()
        actual_limit = limit if limit is not None else self.default_results_limit

        if not self.embedding_model or not self.faiss_index or not self.text_chunk_id_map:
            logger.error("Search service not fully initialized (missing model or FAISS assets).")
            # This should ideally not happen if __init__ succeeded.
            raise RuntimeError("Search service assets not loaded.")

        with self.SessionLocal() as session:
            try:
                ranked_results_orm = perform_search(
                    session=session,
                    query_string=query,
                    model=self.embedding_model,
                    faiss_index=self.faiss_index,
                    text_chunk_id_map=self.text_chunk_id_map,
                    faiss_k=self.default_faiss_k,
                    pagerank_weight=self.default_pagerank_weight,
                    text_relevance_weight=self.default_text_relevance_weight,
                    name_match_weight=self.default_name_match_weight,
                    selected_packages=package_filters,
                    semantic_similarity_threshold=self.default_semantic_similarity_threshold,
                    faiss_nprobe=self.default_faiss_nprobe
                )
            except Exception as e:
                logger.error(f"Error during perform_search: {e}", exc_info=True)
                # Re-raise or handle as appropriate for the service's contract
                raise

        api_results = [
            self._serialize_sg_to_api_item(sg_obj) for sg_obj, _scores in ranked_results_orm
        ]
        
        final_results = api_results[:actual_limit]
        end_time = time.time()
        processing_time_ms = int((end_time - start_time) * 1000)

        return APISearchResponse(
            query=query,
            packages_applied=package_filters,
            results=final_results,
            count=len(final_results),
            total_candidates_considered=len(api_results), # Number before final limit
            processing_time_ms=processing_time_ms
        )

    def get_by_id(self, group_id: int) -> Optional[APISearchResultItem]:
        """Retrieves a specific statement group by its ID from local data.

        Args:
            group_id: The unique identifier of the statement group.

        Returns:
            An APISearchResultItem if found, otherwise None.
        """
        with self.SessionLocal() as session:
            try:
                stmt_group_orm = (
                    session.query(StatementGroup)
                    .options(joinedload(StatementGroup.primary_declaration))
                    .filter(StatementGroup.id == group_id)
                    .first()
                )
                if stmt_group_orm:
                    return self._serialize_sg_to_api_item(stmt_group_orm)
                return None
            except SQLAlchemyError as e:
                logger.error(f"Database error in get_by_id for group_id {group_id}: {e}", exc_info=True)
                return None # Or re-raise custom exception
            except Exception as e:
                logger.error(f"Unexpected error in get_by_id for group_id {group_id}: {e}", exc_info=True)
                return None


    def get_dependencies(self, group_id: int) -> Optional[APICitationsResponse]:
        """Retrieves citations for a specific statement group from local data.

        Citations are the statement groups that the specified group_id depends on.

        Args:
            group_id: The unique identifier of the statement group for which
                      to fetch citations.

        Returns:
            An APICitationsResponse object if the source group is found,
            otherwise None.
        """
        with self.SessionLocal() as session:
            try:
                # Check if the source statement group exists
                source_group_exists = session.query(StatementGroup.id).filter(StatementGroup.id == group_id).first()
                if not source_group_exists:
                    logger.warning(f"Source statement group ID {group_id} not found for dependency lookup.")
                    return None

                # Query for statement groups that `group_id` depends on (citations)
                cited_target_groups_orm = (
                    session.query(StatementGroup)
                    .join(StatementGroupDependency, StatementGroup.id == StatementGroupDependency.target_statement_group_id)
                    .filter(StatementGroupDependency.source_statement_group_id == group_id)
                    .options(joinedload(StatementGroup.primary_declaration))
                    .all()
                )

                citations_api_items = [
                    self._serialize_sg_to_api_item(sg_orm) for sg_orm in cited_target_groups_orm
                ]

                return APICitationsResponse(
                    source_group_id=group_id,
                    citations=citations_api_items,
                    count=len(citations_api_items)
                )
            except SQLAlchemyError as e:
                logger.error(f"Database error in get_dependencies for group_id {group_id}: {e}", exc_info=True)
                return None # Or re-raise
            except Exception as e:
                logger.error(f"Unexpected error in get_dependencies for group_id {group_id}: {e}", exc_info=True)
                return None