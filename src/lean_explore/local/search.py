# src/lean_explore/local/search.py

"""Performs semantic search and ranked retrieval of StatementGroups.

Combines semantic similarity from FAISS, pre-scaled PageRank scores, and
lexical word matching (on Lean name, docstring, and informal descriptions)
to rank StatementGroups. It loads necessary assets (embedding model,
FAISS index, ID map) using default configurations, embeds the user query,
performs FAISS search, filters based on a similarity threshold,
retrieves group details from the database, normalizes semantic similarity,
PageRank, and BM25 scores based on the current candidate set, and then
combines these normalized scores using configurable weights to produce a
final ranked list. It also logs search performance statistics to a dedicated
JSONL file.
"""

import argparse
import datetime
import json
import logging
import os
import pathlib
import re # For tokenization
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from filelock import FileLock, Timeout

# --- Dependency Imports ---
try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sqlalchemy import create_engine, or_, select
    from sqlalchemy.exc import OperationalError, SQLAlchemyError
    from sqlalchemy.orm import Session, joinedload, sessionmaker
    from nltk.stem.porter import PorterStemmer
    from rank_bm25 import BM25Plus
except ImportError as e:
    # pylint: disable=broad-exception-raised
    print(
        f"Error: Missing required libraries ({e}).\n"
        "Please install them: pip install SQLAlchemy faiss-cpu "
        "sentence-transformers numpy filelock rapidfuzz rank_bm25 nltk",
        file=sys.stderr,
    )
    sys.exit(1)

# --- Project Model & Default Config Imports ---
try:
    from lean_explore import defaults  # Using the new defaults module
    from lean_explore.shared.models.db import StatementGroup
except ImportError as e:
    # pylint: disable=broad-exception-raised
    print(
        f"Error: Could not import project modules (StatementGroup, defaults): {e}\n"
        "Ensure 'lean_explore' is installed (e.g., 'pip install -e .') "
        "and all dependencies are met.",
        file=sys.stderr,
    )
    sys.exit(1)


# --- Logging Setup ---
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- Constants ---
NEWLINE = os.linesep
EPSILON = 1e-9
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

_USER_LOGS_BASE_DIR = defaults.LEAN_EXPLORE_USER_DATA_DIR.parent / "logs"
PERFORMANCE_LOG_DIR = str(_USER_LOGS_BASE_DIR)
PERFORMANCE_LOG_FILENAME = "search_stats.jsonl"
PERFORMANCE_LOG_PATH = os.path.join(PERFORMANCE_LOG_DIR, PERFORMANCE_LOG_FILENAME)
LOCK_PATH = os.path.join(PERFORMANCE_LOG_DIR, f"{PERFORMANCE_LOG_FILENAME}.lock")


# --- Performance Logging Helper ---


def log_search_event_to_json(
    status: str,
    duration_ms: float,
    results_count: int,
    error_type: Optional[str] = None,
) -> None:
    """Logs a search event as a JSON line to a dedicated performance log file.

    Args:
        status: A string code indicating the outcome of the search.
        duration_ms: The total duration of the search processing in milliseconds.
        results_count: The number of search results returned.
        error_type: Optional. The type of error if the status indicates an error.
    """
    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "event": "search_processed",
        "status": status,
        "duration_ms": round(duration_ms, 2),
        "results_count": results_count,
    }
    if error_type:
        log_entry["error_type"] = error_type

    try:
        os.makedirs(PERFORMANCE_LOG_DIR, exist_ok=True)
    except OSError as e:
        logger.error(
            "Performance logging error: Could not create log directory %s: %s. "
            "Log entry: %s",
            PERFORMANCE_LOG_DIR,
            e,
            log_entry,
            exc_info=False,
        )
        print(
            f"FALLBACK_PERF_LOG (DIR_ERROR): {json.dumps(log_entry)}", file=sys.stderr
        )
        return

    lock = FileLock(LOCK_PATH, timeout=2)
    try:
        with lock:
            with open(PERFORMANCE_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
    except Timeout:
        logger.warning(
            "Performance logging error: Timeout acquiring lock for %s. "
            "Log entry lost: %s",
            LOCK_PATH,
            log_entry,
        )
        print(
            f"FALLBACK_PERF_LOG (LOCK_TIMEOUT): {json.dumps(log_entry)}",
            file=sys.stderr,
        )
    except Exception as e:
        logger.error(
            "Performance logging error: Failed to write to %s: %s. Log entry: %s",
            PERFORMANCE_LOG_PATH,
            e,
            log_entry,
            exc_info=False,
        )
        print(
            f"FALLBACK_PERF_LOG (WRITE_ERROR): {json.dumps(log_entry)}", file=sys.stderr
        )


# --- Asset Loading Functions ---
def load_faiss_assets(
    index_path_str: str, map_path_str: str
) -> Tuple[Optional[faiss.Index], Optional[List[str]]]:
    """Loads the FAISS index and ID map from specified file paths.

    Args:
        index_path_str: String path to the FAISS index file.
        map_path_str: String path to the JSON ID map file.

    Returns:
        A tuple (faiss.Index or None, list_of_IDs or None).
    """
    index_path = pathlib.Path(index_path_str).resolve()
    map_path = pathlib.Path(map_path_str).resolve()

    if not index_path.exists():
        logger.error("FAISS index file not found: %s", index_path)
        return None, None
    if not map_path.exists():
        logger.error("FAISS ID map file not found: %s", map_path)
        return None, None

    faiss_index_obj: Optional[faiss.Index] = None
    id_map_list: Optional[List[str]] = None

    try:
        logger.info("Loading FAISS index from %s...", index_path)
        faiss_index_obj = faiss.read_index(str(index_path))
        logger.info(
            "Loaded FAISS index with %d vectors (Metric Type: %s).",
            faiss_index_obj.ntotal,
            faiss_index_obj.metric_type,
        )
    except Exception as e:
        logger.error(
            "Failed to load FAISS index from %s: %s", index_path, e, exc_info=True
        )
        return None, id_map_list

    try:
        logger.info("Loading ID map from %s...", map_path)
        with open(map_path, encoding="utf-8") as f:
            id_map_list = json.load(f)
        if not isinstance(id_map_list, list):
            logger.error(
                "ID map file (%s) does not contain a valid JSON list.", map_path
            )
            return faiss_index_obj, None
        logger.info("Loaded ID map with %d entries.", len(id_map_list))
    except Exception as e:
        logger.error(
            "Failed to load or parse ID map file %s: %s", map_path, e, exc_info=True
        )
        return faiss_index_obj, None

    if (
        faiss_index_obj is not None
        and id_map_list is not None
        and faiss_index_obj.ntotal != len(id_map_list)
    ):
        logger.warning(
            "Mismatch: FAISS index size (%d) vs ID map size (%d). "
            "Results may be inconsistent.",
            faiss_index_obj.ntotal,
            len(id_map_list),
        )
    return faiss_index_obj, id_map_list


def load_embedding_model(model_name: str) -> Optional[SentenceTransformer]:
    """Loads the specified Sentence Transformer model.

    Args:
        model_name: The name or path of the sentence-transformer model.

    Returns:
        The loaded model, or None if loading fails.
    """
    logger.info("Loading sentence transformer model '%s'...", model_name)
    try:
        model = SentenceTransformer(model_name)
        logger.info(
            "Model '%s' loaded successfully. Max sequence length: %d.",
            model_name,
            model.max_seq_length,
        )
        return model
    except Exception as e:
        logger.error(
            "Failed to load sentence transformer model '%s': %s",
            model_name,
            e,
            exc_info=True,
        )
        return None


# --- Main Search Function ---


def perform_search(
    session: Session,
    query_string: str,
    model: SentenceTransformer,
    faiss_index: faiss.Index,
    text_chunk_id_map: List[str],
    faiss_k: int,
    pagerank_weight: float,
    text_relevance_weight: float,
    log_searches: bool,
    name_match_weight: float = defaults.DEFAULT_NAME_MATCH_WEIGHT,
    selected_packages: Optional[List[str]] = None,
    semantic_similarity_threshold: float = defaults.DEFAULT_SEM_SIM_THRESHOLD,
    faiss_nprobe: int = defaults.DEFAULT_FAISS_NPROBE,
) -> List[Tuple[StatementGroup, Dict[str, float]]]:
    """Performs semantic and lexical search, then ranks results.

    Scores (semantic similarity, PageRank, BM25) are normalized to a 0-1
    range based on the current set of candidates before being weighted and
    combined.

    Args:
        session: SQLAlchemy session for database access.
        query_string: The user's search query string.
        model: The loaded SentenceTransformer embedding model.
        faiss_index: The loaded FAISS index for text chunks.
        text_chunk_id_map: A list mapping FAISS internal indices to text chunk IDs.
        faiss_k: The number of nearest neighbors to retrieve from FAISS.
        pagerank_weight: Weight for the PageRank score.
        text_relevance_weight: Weight for the semantic similarity score.
        log_searches: If True, search performance data will be logged.
        name_match_weight: Weight for the lexical word match score (BM25).
            Defaults to `defaults.DEFAULT_NAME_MATCH_WEIGHT`.
        selected_packages: Optional list of package names to filter search by.
            Defaults to None.
        semantic_similarity_threshold: Minimum similarity for a result to be
            considered. Defaults to `defaults.DEFAULT_SEM_SIM_THRESHOLD`.
        faiss_nprobe: Number of closest cells/clusters to search for IVF-type
            FAISS indexes. Defaults to `defaults.DEFAULT_FAISS_NPROBE`.

    Returns:
        A list of tuples, sorted by `final_score`. Each tuple contains a
        `StatementGroup` object and a dictionary of its scores.
        The score dictionary includes:
        - 'final_score': The combined weighted score.
        - 'raw_similarity': Original FAISS similarity (0-1).
        - 'norm_similarity': `raw_similarity` normalized across current results.
        - 'original_pagerank_score': PageRank score from the database.
        - 'scaled_pagerank': `original_pagerank_score` normalized across current
                             results (this key is kept for compatibility, but
                             now holds the normalized PageRank).
        - 'raw_word_match_score': Original BM25 score.
        - 'norm_word_match_score': `raw_word_match_score` normalized across
                                   current results.
        - Weighted components: `weighted_norm_similarity`,
          `weighted_scaled_pagerank` (uses normalized PageRank),
          `weighted_word_match_score` (uses normalized BM25 score).

    Raises:
        Exception: If critical errors like query embedding or FAISS search fail.
    """
    overall_start_time = time.time()

    logger.info("Search request event initiated.")
    if semantic_similarity_threshold > 0.0 + EPSILON:
        logger.info(
            "Applying semantic similarity threshold: %.3f",
            semantic_similarity_threshold,
        )

    if not query_string.strip():
        logger.warning("Empty query provided. Returning no results.")
        if log_searches:
            duration_ms = (time.time() - overall_start_time) * 1000
            log_search_event_to_json(
                status="EMPTY_QUERY_SUBMITTED", duration_ms=duration_ms, results_count=0
            )
        return []

    try:
        query_embedding = model.encode([query_string.strip()], convert_to_numpy=True)[
            0
        ].astype(np.float32)
        query_embedding_reshaped = np.expand_dims(query_embedding, axis=0)
        if faiss_index.metric_type == faiss.METRIC_INNER_PRODUCT:
            logger.debug(
                "Normalizing query embedding for Inner Product (cosine) search."
            )
            faiss.normalize_L2(query_embedding_reshaped)
    except Exception as e:
        logger.error("Failed to embed query: %s", e, exc_info=True)
        if log_searches:
            duration_ms = (time.time() - overall_start_time) * 1000
            log_search_event_to_json(
                status="EMBEDDING_ERROR",
                duration_ms=duration_ms,
                results_count=0,
                error_type=type(e).__name__,
            )
        raise Exception(f"Query embedding failed: {e}") from e

    try:
        logger.debug(
            "Searching FAISS index for top %d text chunk neighbors...", faiss_k
        )
        if hasattr(faiss_index, "nprobe") and isinstance(faiss_index.nprobe, int):
            if faiss_nprobe > 0:
                faiss_index.nprobe = faiss_nprobe
                logger.debug(f"Set FAISS nprobe to: {faiss_index.nprobe}")
            else:
                logger.warning(
                    f"Configured faiss_nprobe is {faiss_nprobe}. Must be > 0. "
                    "Using FAISS default or previously set nprobe for this IVF index."
                )
        distances, indices = faiss_index.search(query_embedding_reshaped, faiss_k)
    except Exception as e:
        logger.error("FAISS search failed: %s", e, exc_info=True)
        if log_searches:
            duration_ms = (time.time() - overall_start_time) * 1000
            log_search_event_to_json(
                status="FAISS_SEARCH_ERROR",
                duration_ms=duration_ms,
                results_count=0,
                error_type=type(e).__name__,
            )
        raise Exception(f"FAISS search failed: {e}") from e

    sg_candidates_raw_similarity: Dict[int, float] = {}
    if indices.size > 0 and distances.size > 0:
        for i, faiss_internal_idx in enumerate(indices[0]):
            if faiss_internal_idx == -1:
                continue
            try:
                text_chunk_id_str = text_chunk_id_map[faiss_internal_idx]
                raw_faiss_score = distances[0][i]
                similarity_score: float

                if faiss_index.metric_type == faiss.METRIC_L2:
                    similarity_score = 1.0 / (1.0 + np.sqrt(max(0, raw_faiss_score)))
                elif faiss_index.metric_type == faiss.METRIC_INNER_PRODUCT:
                    similarity_score = raw_faiss_score
                else:
                    similarity_score = 1.0 / (1.0 + max(0, raw_faiss_score))
                    logger.warning(
                        "Unhandled FAISS metric type %d for text chunk. "
                        "Using 1/(1+score) for similarity.",
                        faiss_index.metric_type,
                    )
                similarity_score = max(0.0, min(1.0, similarity_score))

                parts = text_chunk_id_str.split("_")
                if len(parts) >= 2 and parts[0] == "sg":
                    try:
                        sg_id = int(parts[1])
                        if (
                            sg_id not in sg_candidates_raw_similarity
                            or similarity_score > sg_candidates_raw_similarity[sg_id]
                        ):
                            sg_candidates_raw_similarity[sg_id] = similarity_score
                    except ValueError:
                        logger.warning(
                            "Could not parse StatementGroup ID from chunk_id: %s",
                            text_chunk_id_str,
                        )
                else:
                    logger.warning(
                        "Malformed text_chunk_id format: %s", text_chunk_id_str
                    )
            except IndexError:
                logger.warning(
                    "FAISS internal index %d out of bounds for ID map (size %d). "
                    "Possible data inconsistency.",
                    faiss_internal_idx,
                    len(text_chunk_id_map),
                )
            except Exception as e:
                logger.warning(
                    "Error processing FAISS result for internal index %d "
                    "(chunk_id '%s'): %s",
                    faiss_internal_idx,
                    text_chunk_id_str if "text_chunk_id_str" in locals() else "N/A",
                    e,
                )

    if not sg_candidates_raw_similarity:
        logger.info(
            "No valid StatementGroup candidates found after FAISS search and parsing."
        )
        if log_searches:
            duration_ms = (time.time() - overall_start_time) * 1000
            log_search_event_to_json(
                status="NO_FAISS_CANDIDATES", duration_ms=duration_ms, results_count=0
            )
        return []
    logger.info(
        "Aggregated %d unique StatementGroup candidates from FAISS results.",
        len(sg_candidates_raw_similarity),
    )

    if semantic_similarity_threshold > 0.0 + EPSILON:
        initial_candidate_count = len(sg_candidates_raw_similarity)
        sg_candidates_raw_similarity = {
            sg_id: sim
            for sg_id, sim in sg_candidates_raw_similarity.items()
            if sim >= semantic_similarity_threshold
        }
        logger.info(
            "Post-thresholding: %d of %d candidates remaining (threshold: %.3f).",
            len(sg_candidates_raw_similarity),
            initial_candidate_count,
            semantic_similarity_threshold,
        )

        if not sg_candidates_raw_similarity:
            logger.info(
                "No StatementGroup candidates met the semantic similarity "
                "threshold of %.3f.",
                semantic_similarity_threshold,
            )
            if log_searches:
                duration_ms = (time.time() - overall_start_time) * 1000
                log_search_event_to_json(
                    status="NO_CANDIDATES_POST_THRESHOLD",
                    duration_ms=duration_ms,
                    results_count=0,
                )
            return []

    candidate_sg_ids = list(sg_candidates_raw_similarity.keys())
    sg_objects_map: Dict[int, StatementGroup] = {}
    try:
        logger.debug(
            "Fetching StatementGroup details from DB for %d IDs...",
            len(candidate_sg_ids),
        )
        stmt = select(StatementGroup).where(StatementGroup.id.in_(candidate_sg_ids))

        if selected_packages:
            logger.info("Filtering search by packages: %s", selected_packages)
            package_filters_sqla = []
            for pkg_name in selected_packages:
                if pkg_name.strip(): # Ensure package name is not empty
                    package_filters_sqla.append(
                        StatementGroup.source_file.startswith(pkg_name.strip() + "/")
                    )

            if package_filters_sqla:
                stmt = stmt.where(or_(*package_filters_sqla))

        stmt = stmt.options(joinedload(StatementGroup.primary_declaration))
        db_results = session.execute(stmt).scalars().unique().all()
        for sg_obj in db_results:
            sg_objects_map[sg_obj.id] = sg_obj

        logger.debug(
            "Fetched details for %d StatementGroups from DB that matched filters.",
            len(sg_objects_map),
        )
        final_candidate_ids_after_db_match = set(sg_objects_map.keys())
        original_faiss_candidate_ids = set(candidate_sg_ids)

        if len(final_candidate_ids_after_db_match) < len(original_faiss_candidate_ids):
            missing_from_db_or_filtered_out = (
                original_faiss_candidate_ids - final_candidate_ids_after_db_match
            )
            logger.info(
                "%d candidates from FAISS (post-threshold) were not found in DB "
                "or excluded by package filters: (e.g., %s).",
                len(missing_from_db_or_filtered_out),
                list(missing_from_db_or_filtered_out)[:5],
            )

    except SQLAlchemyError as e:
        logger.error(
            "Database query for StatementGroup details failed: %s", e, exc_info=True
        )
        if log_searches:
            duration_ms = (time.time() - overall_start_time) * 1000
            log_search_event_to_json(
                status="DB_FETCH_ERROR",
                duration_ms=duration_ms,
                results_count=0,
                error_type=type(e).__name__,
            )
        raise

    results_with_scores: List[Tuple[StatementGroup, Dict[str, float]]] = []
    # Prepare lists for normalization
    candidate_semantic_similarities: List[float] = []
    candidate_pagerank_scores: List[float] = []
    # BM25 scores will be populated later, then normalized

    processed_candidates_data: List[Dict[str, Any]] = []

    for sg_id in final_candidate_ids_after_db_match: # Use IDs that are confirmed in DB
        sg_obj = sg_objects_map[sg_id]
        raw_sem_sim = sg_candidates_raw_similarity[sg_id] # sg_id is guaranteed to be in this dict

        processed_candidates_data.append(
            {
                "sg_obj": sg_obj,
                "raw_sem_sim": raw_sem_sim,
                "original_pagerank": sg_obj.scaled_pagerank_score if sg_obj.scaled_pagerank_score is not None else 0.0
            }
        )
        candidate_semantic_similarities.append(raw_sem_sim)
        candidate_pagerank_scores.append(
            sg_obj.scaled_pagerank_score if sg_obj.scaled_pagerank_score is not None else 0.0
        )


    if not processed_candidates_data:
        logger.info(
            "No candidates remaining after matching with DB data or other "
            "processing steps."
        )
        if log_searches:
            duration_ms = (time.time() - overall_start_time) * 1000
            log_search_event_to_json(
                status="NO_CANDIDATES_POST_PROCESSING",
                duration_ms=duration_ms,
                results_count=0,
            )
        return []

    # --- BM25 Lexical Scoring ---
    stemmer = PorterStemmer()
    def _get_tokenized_list(text_to_tokenize: str) -> List[str]:
        if not text_to_tokenize:
            return []
        tokens = re.findall(r"\w+", text_to_tokenize.lower())
        return [stemmer.stem(token) for token in tokens]

    tokenized_query = _get_tokenized_list(query_string.strip())
    bm25_corpus: List[List[str]] = []
    for candidate_item_data in processed_candidates_data:
        sg_obj_for_corpus = candidate_item_data["sg_obj"]
        combined_text_for_bm25 = " ".join(
            filter(None, [
                (sg_obj_for_corpus.primary_declaration.lean_name
                 if sg_obj_for_corpus.primary_declaration else None),
                sg_obj_for_corpus.docstring,
                sg_obj_for_corpus.informal_summary,
                sg_obj_for_corpus.informal_description
            ])
        )
        bm25_corpus.append(_get_tokenized_list(combined_text_for_bm25))

    raw_bm25_scores_list: List[float] = [0.0] * len(processed_candidates_data)
    if tokenized_query and any(bm25_corpus):
        try:
            bm25_model = BM25Plus(bm25_corpus)
            raw_bm25_scores_list = bm25_model.get_scores(tokenized_query)
            # Ensure scores are float and non-negative
            raw_bm25_scores_list = [max(0.0, float(score)) for score in raw_bm25_scores_list]
        except Exception as e:
            logger.warning(
                "BM25Plus scoring failed: %s. Word match scores defaulted to 0.", e, exc_info=False
            )
            raw_bm25_scores_list = [0.0] * len(processed_candidates_data) # Ensure correct length on failure


    # --- Score Normalization Setup ---
    # Semantic Similarity
    min_sem_sim = min(candidate_semantic_similarities) if candidate_semantic_similarities else 0.0
    max_sem_sim = max(candidate_semantic_similarities) if candidate_semantic_similarities else 0.0
    range_sem_sim = max_sem_sim - min_sem_sim
    logger.debug("Raw semantic similarity range for normalization: [%.4f, %.4f]", min_sem_sim, max_sem_sim)

    # PageRank
    min_pr = min(candidate_pagerank_scores) if candidate_pagerank_scores else 0.0
    max_pr = max(candidate_pagerank_scores) if candidate_pagerank_scores else 0.0
    range_pr = max_pr - min_pr
    logger.debug("Original PageRank score range for normalization: [%.4f, %.4f]", min_pr, max_pr)

    # BM25
    min_bm25 = min(raw_bm25_scores_list) if raw_bm25_scores_list else 0.0
    max_bm25 = max(raw_bm25_scores_list) if raw_bm25_scores_list else 0.0
    range_bm25 = max_bm25 - min_bm25
    logger.debug("Raw BM25 score range for normalization: [%.4f, %.4f]", min_bm25, max_bm25)


    # --- Scoring Loop for each candidate ---
    for i, candidate_data in enumerate(processed_candidates_data):
        sg_obj = candidate_data["sg_obj"]
        current_raw_sem_sim = candidate_data["raw_sem_sim"]
        original_pagerank_score = candidate_data["original_pagerank"]
        original_bm25_score = raw_bm25_scores_list[i]

        # 1. Normalize Semantic Similarity (maintaining current file's logic style + clamping)
        norm_sem_sim = 0.5
        if candidate_semantic_similarities: # Check if list is not empty
            if range_sem_sim > EPSILON:
                norm_sem_sim = (current_raw_sem_sim - min_sem_sim) / range_sem_sim
            elif len(candidate_semantic_similarities) == 1 and candidate_semantic_similarities[0] > EPSILON:
                norm_sem_sim = 1.0
            elif len(candidate_semantic_similarities) > 0 and range_sem_sim <= EPSILON and max_sem_sim <= EPSILON: # All zeros
                 norm_sem_sim = 0.0
        else: # Should not happen if processed_candidates_data is not empty
            norm_sem_sim = 0.0
        norm_sem_sim = max(0.0, min(1.0, norm_sem_sim))

        # 2. Normalize PageRank Score
        norm_pagerank_score = 0.0
        if candidate_pagerank_scores: # Check if list is not empty
            if range_pr > EPSILON:
                norm_pagerank_score = (original_pagerank_score - min_pr) / range_pr
            elif max_pr > EPSILON: # Single item or all items have same positive score
                norm_pagerank_score = 1.0
            # else: all scores are 0, norm_pagerank_score remains 0.0
        norm_pagerank_score = max(0.0, min(1.0, norm_pagerank_score))

        # 3. Normalize BM25 Score
        norm_bm25_score = 0.0
        if raw_bm25_scores_list: # Check if list is not empty
            if range_bm25 > EPSILON:
                norm_bm25_score = (original_bm25_score - min_bm25) / range_bm25
            elif max_bm25 > EPSILON: # Single item or all items have same positive score
                norm_bm25_score = 1.0
            # else: all scores are 0, norm_bm25_score remains 0.0
        norm_bm25_score = max(0.0, min(1.0, norm_bm25_score))

        # Weighted scores using normalized components
        weighted_norm_similarity = text_relevance_weight * norm_sem_sim
        weighted_norm_pagerank = pagerank_weight * norm_pagerank_score
        weighted_norm_bm25_score = name_match_weight * norm_bm25_score

        final_score = weighted_norm_similarity + weighted_norm_pagerank + weighted_norm_bm25_score

        score_dict = {
            "final_score": final_score,
            "raw_similarity": current_raw_sem_sim,
            "norm_similarity": norm_sem_sim,
            "original_pagerank_score": original_pagerank_score,
            "scaled_pagerank": norm_pagerank_score, # Key retained for compatibility, value is normalized
            "raw_word_match_score": original_bm25_score, # Original BM25
            "norm_word_match_score": norm_bm25_score,    # Normalized BM25
            "weighted_norm_similarity": weighted_norm_similarity,
            "weighted_scaled_pagerank": weighted_norm_pagerank, # Key retained, uses normalized PR
            "weighted_word_match_score": weighted_norm_bm25_score, # Key retained, uses normalized BM25
        }
        results_with_scores.append((sg_obj, score_dict))

    results_with_scores.sort(key=lambda item: item[1]["final_score"], reverse=True)

    final_status = "SUCCESS"
    results_count = len(results_with_scores)
    if not results_with_scores and processed_candidates_data:
        final_status = "NO_RESULTS_FINAL_SCORED"
    elif not results_with_scores and not processed_candidates_data:
        # This block might need review based on earlier status logging points
        # For now, ensure a status if truly no results by this stage
        if not sg_candidates_raw_similarity: # Checks after FAISS and initial thresholding
             final_status = "NO_CANDIDATES_POST_THRESHOLD" # Or NO_FAISS_CANDIDATES if that was the case
        # else: Handled by NO_CANDIDATES_POST_PROCESSING if that was logged

    if log_searches:
        duration_ms = (time.time() - overall_start_time) * 1000
        log_search_event_to_json(
            status=final_status, duration_ms=duration_ms, results_count=results_count
        )

    return results_with_scores


# --- Output Formatting ---


def print_results(results: List[Tuple[StatementGroup, Dict[str, float]]]) -> None:
    """Formats and prints the search results to the console.

    Args:
        results: A list of tuples, each containing a StatementGroup
            object and its scores, sorted by final_score.
    """
    if not results:
        print("\nNo results found.")
        return

    print(f"\n--- Top {len(results)} Search Results (StatementGroups) ---")
    for i, (sg_obj, scores) in enumerate(results):
        primary_decl_name = (
            sg_obj.primary_declaration.lean_name
            if sg_obj.primary_declaration and sg_obj.primary_declaration.lean_name
            else "N/A"
        )
        print(
            f"\n{i + 1}. Lean Name: {primary_decl_name} (SG ID: {sg_obj.id})\n"
            f"   Final Score: {scores['final_score']:.4f} ("
            f"NormSim*W: {scores['weighted_norm_similarity']:.4f}, "
            f"NormPR*W: {scores['weighted_scaled_pagerank']:.4f}, " # Uses normalized PR
            f"NormWordMatch*W: {scores['weighted_word_match_score']:.4f})" # Uses normalized BM25
        )
        print(
            f"   Scores: [NormSim: {scores['norm_similarity']:.4f} (Raw: {scores['raw_similarity']:.4f}), "
            f"NormPR: {scores['scaled_pagerank']:.4f} (Original: {scores['original_pagerank_score']:.4f}), "
            f"NormWordMatch: {scores['norm_word_match_score']:.4f} (OriginalBM25: {scores['raw_word_match_score']:.2f})]"
        )

        lean_display = (
            sg_obj.display_statement_text or sg_obj.statement_text or "[No Lean code]"
        )
        lean_display_short = (
            (lean_display[:200] + "...") if len(lean_display) > 200 else lean_display
        )
        print(f"   Lean Code: {lean_display_short.replace(NEWLINE, ' ')}")

        desc_display = (
            sg_obj.informal_description or sg_obj.docstring or "[No description]"
        )
        desc_display_short = (
            (desc_display[:150] + "...") if len(desc_display) > 150 else desc_display
        )
        print(f"   Description: {desc_display_short.replace(NEWLINE, ' ')}")

        source_loc = sg_obj.source_file or "[No source file]"
        if source_loc.startswith("Mathlib/"):
            source_loc = source_loc[len("Mathlib/") :]
        print(f"   File: {source_loc}:{sg_obj.range_start_line}")

    print("\n---------------------------------------------------")


# --- Argument Parsing & Main Execution ---


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the search script.

    Returns:
        An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Search Lean StatementGroups using combined scoring.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("query", type=str, help="The search query string.")
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="Maximum number of final results to display. Overrides default if set.",
    )
    parser.add_argument(
        "--packages",
        metavar="PKG",
        type=str,
        nargs="*",
        default=None,
        help="Filter search results by specific package names (e.g., Mathlib Std). "
        "If not provided, searches all packages.",
    )
    return parser.parse_args()


def main():
    """Main execution function for the search script."""
    args = parse_arguments()

    logger.info(
        "Using default configurations for paths and parameters from "
        "lean_explore.defaults."
    )

    db_url = defaults.DEFAULT_DB_URL # type: ignore
    embedding_model_name = defaults.DEFAULT_EMBEDDING_MODEL_NAME # type: ignore
    resolved_idx_path = str(defaults.DEFAULT_FAISS_INDEX_PATH.resolve()) # type: ignore
    resolved_map_path = str(defaults.DEFAULT_FAISS_MAP_PATH.resolve()) # type: ignore

    faiss_k_cand = defaults.DEFAULT_FAISS_K # type: ignore
    pr_weight = defaults.DEFAULT_PAGERANK_WEIGHT # type: ignore
    sem_sim_weight = defaults.DEFAULT_TEXT_RELEVANCE_WEIGHT # type: ignore
    name_match_w = defaults.DEFAULT_NAME_MATCH_WEIGHT # type: ignore
    results_disp_limit = (
        args.limit if args.limit is not None else defaults.DEFAULT_RESULTS_LIMIT # type: ignore
    )
    semantic_sim_thresh = defaults.DEFAULT_SEM_SIM_THRESHOLD # type: ignore
    faiss_nprobe_val = defaults.DEFAULT_FAISS_NPROBE # type: ignore

    db_url_display = (
        f"...{str(defaults.DEFAULT_DB_PATH.resolve())[-30:]}" # type: ignore
        if len(str(defaults.DEFAULT_DB_PATH.resolve())) > 30 # type: ignore
        else str(defaults.DEFAULT_DB_PATH.resolve()) # type: ignore
    )
    logger.info("--- Starting Search (Direct Script Execution) ---")
    logger.info("Query: '%s'", args.query)
    logger.info("Displaying Top: %d results", results_disp_limit)
    if args.packages:
        logger.info("Filtering by user-specified packages: %s", args.packages)
    else:
        logger.info("No package filter specified, searching all packages.")
    logger.info("FAISS k (candidates): %d", faiss_k_cand)
    logger.info("FAISS nprobe (from defaults): %d", faiss_nprobe_val)
    logger.info(
        "Semantic Similarity Threshold (from defaults): %.3f", semantic_sim_thresh
    )
    logger.info(
        "Weights -> NormTextSim: %.2f, NormPR: %.2f, NormWordMatch (BM25): %.2f",
        sem_sim_weight,
        pr_weight,
        name_match_w,
    )
    logger.info("Using FAISS index: %s", resolved_idx_path)
    logger.info("Using ID map: %s", resolved_map_path)
    logger.info("Database path: %s", db_url_display)

    try:
        _USER_LOGS_BASE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning(
            f"Could not create user log directory {_USER_LOGS_BASE_DIR}: {e}"
        )

    engine = None
    try:
        s_transformer_model = load_embedding_model(embedding_model_name)
        if s_transformer_model is None:
            logger.error(
                "Sentence transformer model loading failed. Cannot proceed with search."
            )
            sys.exit(1)

        faiss_idx, id_map = load_faiss_assets(resolved_idx_path, resolved_map_path)
        if faiss_idx is None or id_map is None:
            logger.error(
                "Failed to load critical FAISS assets (index or ID map).\n"
                f"Expected at:\n  Index path: {resolved_idx_path}\n"
                f"  ID map path: {resolved_map_path}\n"
                "Please ensure these files exist or run 'leanexplore data fetch' "
                "to download the data toolchain."
            )
            sys.exit(1)

        is_file_db = db_url.startswith("sqlite:///")
        db_file_path = None
        if is_file_db:
            db_file_path_str = db_url[len("sqlite///") :]
            db_file_path = pathlib.Path(db_file_path_str)
            if not db_file_path.exists():
                logger.error(
                    f"Database file not found at the expected location: "
                    f"{db_file_path}\n"
                    "Please run 'leanexplore data fetch' to download the data "
                    "toolchain."
                )
                sys.exit(1)

        engine = create_engine(db_url, echo=False)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

        with SessionLocal() as session:
            ranked_results = perform_search(
                session=session,
                query_string=args.query,
                model=s_transformer_model,
                faiss_index=faiss_idx,
                text_chunk_id_map=id_map,
                faiss_k=faiss_k_cand,
                pagerank_weight=pr_weight,
                text_relevance_weight=sem_sim_weight,
                log_searches=True,
                name_match_weight=name_match_w,
                selected_packages=args.packages,
                semantic_similarity_threshold=semantic_sim_thresh,
                faiss_nprobe=faiss_nprobe_val,
            )

        print_results(ranked_results[:results_disp_limit])

    except FileNotFoundError as e:
        logger.error(
            f"A required file was not found: {e.filename}.\n"
            "This could be an issue with configured paths or missing data.\n"
            "If this relates to core data assets, please try running "
            "'leanexplore data fetch'."
        )
        sys.exit(1)
    except OperationalError as e_db:
        is_file_db_op_err = defaults.DEFAULT_DB_URL.startswith("sqlite:///") # type: ignore
        db_file_path_op_err = defaults.DEFAULT_DB_PATH # type: ignore
        if is_file_db_op_err and (
            "unable to open database file" in str(e_db).lower()
            or (db_file_path_op_err and not db_file_path_op_err.exists())
        ):
            p = str(db_file_path_op_err.resolve())
            logger.error(
                f"Database connection failed: {e_db}\n"
                f"The database file appears to be missing or inaccessible at: "
                f"{p if db_file_path_op_err else 'Unknown Path'}\n"
                "Please run 'leanexplore data fetch' to download or update the "
                "data toolchain."
            )
        else:
            logger.error(
                f"Database connection/operational error: {e_db}", exc_info=True
            )
        sys.exit(1)
    except SQLAlchemyError as e_sqla:
        logger.error(
            "A database error occurred during search: %s", e_sqla, exc_info=True
        )
        sys.exit(1)
    except Exception as e_general:
        logger.critical(
            "An unexpected critical error occurred during search: %s",
            e_general,
            exc_info=True,
        )
        sys.exit(1)
    finally:
        if engine:
            engine.dispose()
            logger.debug("Database engine disposed.")


if __name__ == "__main__":
    main()