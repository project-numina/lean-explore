# File: scripts/update_primary_declarations.py

"""Updates primary declarations for statement groups in the database.

This script iterates through all StatementGroup entries and, for each group,
evaluates if its primary_decl_id should be changed. The new primary
declaration is chosen from the group's members if its lean_name is shorter
than the current primary's lean_name and is also a prefix of the current
primary's lean_name. If multiple such candidates exist, the one with the
shortest lean_name is selected. The StatementGroup's docstring is also
updated to match the new primary declaration's docstring.
"""

import logging
import sys
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import joinedload, scoped_session, sessionmaker

# Determine project root and add src to sys.path to import models
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from lean_explore.shared.models.db import Base, StatementGroup  # noqa: E402

# --- Configuration ---
DATABASE_FILE = PROJECT_ROOT / "data" / "lean_explore_data.db"
DATABASE_URL = f"sqlite:///{DATABASE_FILE.resolve()}"
# Adjustable batch size for committing database updates
BATCH_SIZE = 10000

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def update_primary_declarations_in_db(db_url: str):
    """Updates the primary_decl_id for StatementGroups.

    Iterates through StatementGroups, applying logic to select a new
    primary declaration based on name length and prefix relationship
    to the current primary. Updates are committed in batches.

    Args:
        db_url (str): The database connection URL.
    """
    engine = create_engine(db_url)
    Base.metadata.bind = engine
    SessionLocal = scoped_session(
        sessionmaker(autocommit=False, autoflush=False, bind=engine)
    )
    session = SessionLocal()

    updated_groups_count = 0
    processed_groups_count = 0
    pending_updates_in_batch = 0

    try:
        logger.info("Fetching all statement groups with their declarations...")
        result = session.execute(
            select(StatementGroup).options(
                joinedload(StatementGroup.primary_declaration),
                joinedload(StatementGroup.declarations),
            )
        )
        stmt_groups = result.unique().scalars().all()

        total_groups = len(stmt_groups)
        logger.info(f"Found {total_groups} statement groups to process.")

        for i, sg in enumerate(stmt_groups):
            processed_groups_count += 1
            if (i + 1) % 1000 == 0 or (i + 1) == total_groups:
                logger.info(f"Processing group {i + 1}/{total_groups} (ID: {sg.id})...")

            current_primary_decl = sg.primary_declaration
            if not current_primary_decl:
                logger.warning(
                    f"StatementGroup ID {sg.id} has no current primary declaration. "
                    "Skipping."
                )
                continue

            member_declarations = sg.declarations
            if not member_declarations:
                logger.warning(
                    f"StatementGroup ID {sg.id} has no member declarations. Skipping."
                )
                continue

            non_internal_members = [d for d in member_declarations if not d.is_internal]
            candidate_pool = (
                non_internal_members if non_internal_members else member_declarations
            )

            best_new_candidate = None

            for decl in candidate_pool:
                if decl.id == current_primary_decl.id:
                    continue

                if (
                    decl.lean_name
                    and current_primary_decl.lean_name
                    and len(decl.lean_name) < len(current_primary_decl.lean_name)
                    and current_primary_decl.lean_name.startswith(decl.lean_name)
                ):
                    if best_new_candidate is None or len(decl.lean_name) < len(
                        best_new_candidate.lean_name
                    ):
                        best_new_candidate = decl

            if best_new_candidate:
                if sg.primary_decl_id != best_new_candidate.id:
                    old_primary_name = current_primary_decl.lean_name
                    new_primary_name = best_new_candidate.lean_name
                    logger.info(
                        f"Updating StatementGroup ID {sg.id}: "
                        f"Old primary: '{old_primary_name}' "
                        f"(ID: {sg.primary_decl_id}), "
                        f"New primary: '{new_primary_name}' "
                        f"(ID: {best_new_candidate.id})"
                    )
                    sg.primary_decl_id = best_new_candidate.id
                    sg.docstring = best_new_candidate.docstring
                    session.add(sg)
                    updated_groups_count += 1
                    pending_updates_in_batch += 1

            if pending_updates_in_batch >= BATCH_SIZE:
                logger.info(
                    f"Committing batch of {pending_updates_in_batch} updates. "
                    f"Total updates so far: {updated_groups_count}."
                )
                session.commit()
                pending_updates_in_batch = 0

        if session.dirty:
            logger.info(f"Committing final {len(session.dirty)} outstanding changes...")
            session.commit()

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        session.rollback()
    finally:
        logger.info(
            f"Finished processing. Total groups processed: {processed_groups_count}. "
            f"Total groups updated: {updated_groups_count}."
        )
        session.close()


if __name__ == "__main__":
    logger.info(
        "Starting script to update primary declarations based on shorter prefix names."
    )
    update_primary_declarations_in_db(DATABASE_URL)
    logger.info("Script finished.")
