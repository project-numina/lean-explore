import argparse
import logging
from typing import Tuple

from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session, sessionmaker
from tqdm import tqdm

from lean_explore.shared.models.db import StatementGroup

logging.basicConfig(
    level="INFO",
    format="%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the database merge script.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=" Merge two versions of the database.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db-url",
        type=str,
        help="SQLAlchemy database URL of the current version of the database.",
    )
    parser.add_argument(
        "--prev-db-url",
        type=str,
        help="SQLAlchemy database URL of the previous version of the database.",
    )
    return parser.parse_args()


def create_db_session(db_url: str) -> sessionmaker[Session]:
    """Creates a new database session.

    Args:
        db_url (str): The SQLAlchemy database URL to connect to.

    Returns:
        SessionLocal: A new session local instance for database operations.
    """
    engine = create_engine(db_url, echo=False)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal


def pong_db(session_factory: sessionmaker[Session]) -> Tuple[bool, Exception | None]:
    """Checks if the database is connectable by executing a simple query.

    Args:
        session_factory (sessionmaker[Session]): A factory function for creating database sessions.
    """
    # test connectable
    try:
        with session_factory() as session:
            session.execute(text("SELECT 1"))
        return True, None
    except Exception as e:
        return False, e


def main():
    args = parse_arguments()
    db_url: str = args.db_url
    prev_db_url: str = args.prev_db_url

    db_url_display = db_url if len(db_url) < 70 else f"{db_url[:30]}...{db_url[-30:]}"
    prev_db_url_display = (
        prev_db_url
        if len(prev_db_url) < 70
        else f"{prev_db_url[:30]}...{prev_db_url[-30:]}"
    )
    logger.info("Database URL: %s", db_url_display)
    logger.info("Previous Database URL: %s", prev_db_url_display)

    current_session_factory = create_db_session(db_url)
    prev_session_factory = create_db_session(prev_db_url)

    connect_ok, ex = pong_db(current_session_factory)
    if not connect_ok:
        logger.error(f"Could not connect to the database {db_url_display}: {ex}")
        exit(1)

    connect_ok, ex = pong_db(prev_session_factory)
    if not connect_ok:
        logger.error(f"Could not connect to the database {prev_db_url_display}: {ex}")
        exit(1)

    stmt_groups = select(StatementGroup)
    with prev_session_factory() as prev_session:
        prev_groups = prev_session.execute(stmt_groups).scalars().all()
        logger.info(f"Found {len(prev_groups)} groups in the previous database.")

    prev_group_to_desc = {
        # _group.text_hash: _group.informal_description

        f"{_group.text_hash}_{_group.docstring}_{_group.source_file}": _group.informal_description

        # f"{_group.text_hash}_{_group.docstring}_{_group.source_file}_"
        # f"{_group.range_start_line}_{_group.range_start_col}_"
        # f"{_group.range_end_line}_{_group.range_end_col}": _group.informal_description
        for _group in prev_groups
    }

    with current_session_factory() as crt_session:
        crt_groups = crt_session.execute(stmt_groups).scalars().all()
        logger.info(f"Found {len(crt_groups)} groups in the current database.")

    crt_group_to_id = {
        # _group.text_hash: _group.id

        f"{_group.text_hash}_{_group.docstring}_{_group.source_file}": _group.id

        # f"{_group.text_hash}_{_group.docstring}_{_group.source_file}_"
        # f"{_group.range_start_line}_{_group.range_start_col}_"
        # f"{_group.range_end_line}_{_group.range_end_col}": _group.id
        for _group in crt_groups
    }

    overlap_keys = set(prev_group_to_desc.keys()) & set(crt_group_to_id.keys())
    logger.info(f"Found {len(overlap_keys)} overlapping groups.")

    with current_session_factory() as crt_session:
        for _key in tqdm(overlap_keys):
            new_group = StatementGroup(
                id=crt_group_to_id[_key], informal_description=prev_group_to_desc[_key]
            )
            crt_session.merge(new_group)
            crt_session.commit()


if __name__ == "__main__":
    main()
