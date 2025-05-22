# src/lean_explore/mcp/server.py

"""Main script to run the Lean Explore MCP (Model Context Protocol) Server.

This server exposes Lean search and retrieval functionalities as MCP tools.
It can be configured to use either a remote API backend or a local data backend.

The server listens for MCP messages (JSON-RPC 2.0) over stdio.

Command-line arguments:
  --backend {'api', 'local'} : Specifies the backend to use. (required)
  --api-key TEXT             : The API key, required if --backend is 'api'.
  --log-level TEXT           : Sets the logging output level (e.g., INFO, WARNING, DEBUG).
"""

import argparse
import logging
import sys

# Import the FastMCP app instance from mcp.app
# This instance has the lifespan manager configured.
from lean_explore.mcp.app import mcp_app, BackendServiceType

# Import backend clients/services
from lean_explore.api.client import Client as APIClient
from lean_explore.local.service import Service as LocalService

# Import tools to ensure they are registered with the mcp_app
# The '# noqa' comments silence linters about unused imports, as the import
# itself triggers tool registration via decorators in tools.py.
from lean_explore.mcp import tools  # noqa: F401 pylint: disable=unused-import

# Initial basicConfig for the module. This will be overridden in main() if --log-level is passed.
logging.basicConfig(
    level=logging.INFO, # Default level if script is not run through main() or if main() fails early
    format="%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stderr  # MCP servers should log to stderr
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the MCP server.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
                            Attributes include 'backend', 'api_key', and 'log_level'.
    """
    parser = argparse.ArgumentParser(
        description="Lean Explore MCP Server. Provides Lean search tools via MCP."
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["api", "local"],
        required=True,
        help="Specifies the backend to use: 'api' for remote API, 'local' for local data."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the remote API backend. Required if --backend is 'api'."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="ERROR",
        help="Set the logging output level (default: INFO)."
    )
    return parser.parse_args()


def main():
    """Main function to initialize and run the MCP server."""
    args = parse_arguments()

    # Configure logging level based on command-line argument
    # This will override the module-level basicConfig thanks to force=True (Python 3.8+)
    log_level_name = args.log_level.upper()
    numeric_level = getattr(logging, log_level_name, logging.INFO) # Default to INFO if getattr fails
    if not isinstance(numeric_level, int): # Should not happen with argparse choices
        numeric_level = logging.INFO
        # Log this potential issue using a temporary basic config if force=True wasn't available
        # or if we want to ensure this message appears regardless of the user's chosen level's severity.
        # For now, we assume getattr will work due to choices.

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
        force=True # Allows re-configuration by replacing handlers
    )

    logger.info(f"Starting Lean Explore MCP Server with backend: {args.backend}")
    if numeric_level > logging.INFO: # If log level is WARNING or higher, some INFO logs won't show
        # To ensure the "Starting..." message is seen even if default is higher,
        # one might log it before reconfiguring, or log it at a higher level.
        # For now, it will respect the user's chosen log level.
        pass


    backend_service_instance: BackendServiceType = None

    if args.backend == "local":
        try:
            backend_service_instance = LocalService()
            logger.info("Local backend service initialized successfully.")
        except FileNotFoundError as e:
            logger.error(f"Failed to initialize LocalService due to missing data file: {e}")
            logger.error(
                "Please ensure local data (database, FAISS index) has been downloaded "
                "and is accessible at the default paths."
            )
            sys.exit(1)
        except RuntimeError as e:
            logger.error(f"Failed to initialize LocalService: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"An unexpected error occurred while initializing LocalService: {e}", exc_info=True)
            sys.exit(1)

    elif args.backend == "api":
        if not args.api_key:
            logger.error("--api-key is required when using the 'api' backend.")
            sys.exit(1)
        try:
            backend_service_instance = APIClient(api_key=args.api_key)
            logger.info("API client backend initialized successfully.")
        except Exception as e:
            logger.error(f"An unexpected error occurred while initializing APIClient: {e}", exc_info=True)
            sys.exit(1)
    else:
        # argparse choices should prevent this, but as a safeguard:
        logger.error(f"Invalid backend choice: {args.backend}. Must be 'api' or 'local'.")
        sys.exit(1)

    if backend_service_instance is None:
        # This case should ideally be caught by earlier checks.
        logger.critical("Backend service instance was not created. Exiting.")
        sys.exit(1)

    # Make the initialized backend service available to the FastMCP app's lifespan context
    # The app_lifespan function in mcp.app will pick this up.
    mcp_app._lean_explore_backend_service = backend_service_instance
    logger.info(f"Backend service ({args.backend}) attached to MCP app state.")

    try:
        logger.info("Running MCP server with stdio transport...")
        # The tools are registered when lean_explore.mcp.tools was imported.
        mcp_app.run(transport='stdio')
    except Exception as e:
        logger.critical(f"MCP server exited with an unexpected error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("MCP server has shut down.")


if __name__ == "__main__":
    main()