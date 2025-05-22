# src/lean_explore/defaults.py

"""Provides default paths and configuration parameters for the lean_explore package.

This module centralizes default values that were previously expected from
an APP_CONFIG object. By hardcoding these, the package becomes more
self-contained and easier for users to get started with, especially
when using local search functionalities. The defined paths point to
a user-specific data directory where downloaded assets (database,
FAISS index, etc.) are expected to reside.
"""

import os
import pathlib
from typing import Final

# --- User-Specific Data Directory ---
# Define a base directory within the user's home folder to store
# downloaded data assets for lean_explore.
# Example: ~/.lean_explore/data/
# The application or a setup utility should ensure this directory exists.
USER_HOME_DIR: Final[pathlib.Path] = pathlib.Path(os.path.expanduser("~"))
LEAN_EXPLORE_USER_DATA_DIR: Final[pathlib.Path] = USER_HOME_DIR / ".lean_explore" / "data"

# --- Default Filenames ---
DEFAULT_DB_FILENAME: Final[str] = "lean_explore_data.db"
DEFAULT_FAISS_INDEX_FILENAME: Final[str] = "main_faiss.index"
DEFAULT_FAISS_MAP_FILENAME: Final[str] = "faiss_ids_map.json"

# --- Default Full Paths (to be used by the application) ---
# These paths indicate where the package will look for its data files.
# The "database manager" component will download files to these locations.

DEFAULT_DB_PATH: Final[pathlib.Path] = LEAN_EXPLORE_USER_DATA_DIR / DEFAULT_DB_FILENAME
DEFAULT_FAISS_INDEX_PATH: Final[pathlib.Path] = LEAN_EXPLORE_USER_DATA_DIR / DEFAULT_FAISS_INDEX_FILENAME
DEFAULT_FAISS_MAP_PATH: Final[pathlib.Path] = LEAN_EXPLORE_USER_DATA_DIR / DEFAULT_FAISS_MAP_FILENAME

# For SQLAlchemy, the database URL needs to be a string.
# We construct the SQLite URL string from the Path object.
DEFAULT_DB_URL: Final[str] = f"sqlite:///{DEFAULT_DB_PATH.resolve()}"


# --- Default Embedding Model ---
DEFAULT_EMBEDDING_MODEL_NAME: Final[str] = "BAAI/bge-base-en-v1.5"


# --- Default Search Parameters ---
# These values are based on the previously discussed config.yml and search.py fallbacks.

# FAISS Search Parameters
DEFAULT_FAISS_K: Final[int] = 100  # Number of nearest neighbors from FAISS
DEFAULT_FAISS_NPROBE: Final[int] = 200 # For IVF-type FAISS indexes

# Scoring and Ranking Parameters
DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD: Final[float] = 0.525
DEFAULT_PAGERANK_WEIGHT: Final[float] = 1.0
DEFAULT_TEXT_RELEVANCE_WEIGHT: Final[float] = 0.2
DEFAULT_NAME_MATCH_WEIGHT: Final[float] = 0.5

# Output Parameters
DEFAULT_RESULTS_LIMIT: Final[int] = 50 # Default number of final results to display/return


# --- Other Constants (if any emerge) ---
# Example: If your application needs other hardcoded default values,
# they can be added here.
# DEFAULT_SOME_OTHER_PARAMETER: Final[Any] = "some_value"