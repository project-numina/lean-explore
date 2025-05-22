# src/lean_explore/cli/config_utils.py

"""Utilities for managing CLI user configurations, such as API keys.

This module provides functions to save and load user-specific settings,
such as API keys for Lean Explore and OpenAI, from a configuration
file stored in the user's home directory. It handles file creation,
parsing, and sets secure permissions for files containing sensitive
information.
"""

import logging
import os
import pathlib
import toml
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Define the application's configuration directory and file name
_APP_CONFIG_DIR_NAME: str = "leanexplore"
_CONFIG_FILENAME: str = "config.toml"

# Define keys for Lean Explore API section
_LEAN_EXPLORE_API_SECTION_NAME: str = "lean_explore_api" # Renamed for clarity
_LEAN_EXPLORE_API_KEY_NAME: str = "key"

# Define keys for OpenAI API section
_OPENAI_API_SECTION_NAME: str = "openai"
_OPENAI_API_KEY_NAME: str = "api_key" # Using a distinct key name for clarity


def get_config_file_path() -> pathlib.Path:
    """Constructs and returns the absolute path to the configuration file.

    The path is typically ~/.config/leanexplore/config.toml.

    Returns:
        pathlib.Path: The absolute path to the configuration file.
    """
    config_dir = pathlib.Path(os.path.expanduser("~")) / ".config" / _APP_CONFIG_DIR_NAME
    return config_dir / _CONFIG_FILENAME


def _ensure_config_dir_exists() -> None:
    """Ensures that the configuration directory exists.

    Creates the directory if it's not already present.

    Raises:
        OSError: If the directory cannot be created due to permission issues
                 or other OS-level errors.
    """
    config_file_path = get_config_file_path()
    config_dir = config_file_path.parent
    try:
        config_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create configuration directory {config_dir}: {e}")
        raise


def _load_config_data(config_file_path: pathlib.Path) -> Dict[str, Any]:
    """Loads configuration data from a TOML file.

    Args:
        config_file_path: Path to the configuration file.

    Returns:
        A dictionary containing the configuration data. Returns an empty
        dictionary if the file does not exist or is corrupted.
    """
    config_data: Dict[str, Any] = {}
    if config_file_path.exists() and config_file_path.is_file():
        try:
            with open(config_file_path, "r", encoding="utf-8") as f:
                config_data = toml.load(f)
        except toml.TomlDecodeError:
            logger.warning(f"Configuration file {config_file_path} is corrupted. Treating as empty.")
            # Potentially back up corrupted file before returning empty
        except Exception as e:
            logger.error(f"Error reading existing config file {config_file_path}: {e}", exc_info=True)
            # Decide if to proceed with empty or raise further
    return config_data


def _save_config_data(config_file_path: pathlib.Path, config_data: Dict[str, Any]) -> bool:
    """Saves configuration data to a TOML file with secure permissions.

    Args:
        config_file_path: Path to the configuration file.
        config_data: Dictionary containing the configuration data to save.

    Returns:
        True if saving was successful, False otherwise.
    """
    try:
        with open(config_file_path, "w", encoding="utf-8") as f:
            toml.dump(config_data, f)
        os.chmod(config_file_path, 0o600) # Set user read/write only
        return True
    except OSError as e:
        logger.error(f"OS error saving configuration to {config_file_path}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error saving configuration to {config_file_path}: {e}", exc_info=True)
    return False


# --- Lean Explore API Key Management ---

def save_api_key(api_key: str) -> bool:
    """Saves the Lean Explore API key to the user's configuration file.

    Args:
        api_key: The Lean Explore API key string to save.

    Returns:
        bool: True if the API key was saved successfully, False otherwise.
    """
    if not api_key or not isinstance(api_key, str):
        logger.error("Attempted to save an invalid or empty Lean Explore API key.")
        return False

    config_file_path = get_config_file_path()
    try:
        _ensure_config_dir_exists()
        config_data = _load_config_data(config_file_path)

        if _LEAN_EXPLORE_API_SECTION_NAME not in config_data or \
           not isinstance(config_data[_LEAN_EXPLORE_API_SECTION_NAME], dict):
            config_data[_LEAN_EXPLORE_API_SECTION_NAME] = {}
        
        config_data[_LEAN_EXPLORE_API_SECTION_NAME][_LEAN_EXPLORE_API_KEY_NAME] = api_key

        if _save_config_data(config_file_path, config_data):
            logger.info(f"Lean Explore API key saved to {config_file_path}")
            return True
    except Exception as e: # Catch any exception from _ensure_config_dir_exists or broad issues
        logger.error(f"General error during Lean Explore API key saving process: {e}", exc_info=True)
    return False


def load_api_key() -> Optional[str]:
    """Loads the Lean Explore API key from the user's configuration file.

    Returns:
        Optional[str]: The Lean Explore API key string if found and valid, otherwise None.
    """
    config_file_path = get_config_file_path()
    if not config_file_path.exists() or not config_file_path.is_file():
        logger.debug(f"Configuration file not found at {config_file_path} for Lean Explore API key.")
        return None

    try:
        config_data = _load_config_data(config_file_path)
        api_key = config_data.get(_LEAN_EXPLORE_API_SECTION_NAME, {}).get(_LEAN_EXPLORE_API_KEY_NAME)
        
        if api_key and isinstance(api_key, str):
            logger.debug(f"Lean Explore API key loaded successfully from {config_file_path}")
            return api_key
        elif api_key: # Found but not a string
            logger.warning(f"Lean Explore API key found in {config_file_path} but is not a valid string.")
        else: # Not found under the expected keys
            logger.debug(
                f"Lean Explore API key not found under section "
                f"'{_LEAN_EXPLORE_API_SECTION_NAME}', key '{_LEAN_EXPLORE_API_KEY_NAME}' "
                f"in {config_file_path}"
            )
    except Exception as e: # Catch any other unexpected errors during loading
        logger.error(f"Unexpected error loading Lean Explore API key from {config_file_path}: {e}", exc_info=True)
    return None


def delete_api_key() -> bool:
    """Deletes the Lean Explore API key from the user's configuration file.

    Returns:
        bool: True if the API key was successfully removed or if it did not exist;
              False if an error occurred.
    """
    config_file_path = get_config_file_path()
    if not config_file_path.exists():
        logger.info("No Lean Explore API key to delete: configuration file does not exist.")
        return True

    try:
        config_data = _load_config_data(config_file_path)
        api_section = config_data.get(_LEAN_EXPLORE_API_SECTION_NAME)

        if api_section and isinstance(api_section, dict) and _LEAN_EXPLORE_API_KEY_NAME in api_section:
            del api_section[_LEAN_EXPLORE_API_KEY_NAME]
            logger.info("Lean Explore API key removed from configuration data.")
            
            if not api_section: # If the section is now empty
                del config_data[_LEAN_EXPLORE_API_SECTION_NAME]
                logger.info(f"Empty '{_LEAN_EXPLORE_API_SECTION_NAME}' section removed.")

            if _save_config_data(config_file_path, config_data):
                logger.info(f"Lean Explore API key deleted from {config_file_path}")
                return True
            return False # Save failed
        else:
            logger.info(f"Lean Explore API key not found in {config_file_path}, no deletion performed.")
            return True # Key wasn't there, so considered successful
            
    except Exception as e:
        logger.error(f"Unexpected error deleting Lean Explore API key from {config_file_path}: {e}", exc_info=True)
    return False


# --- OpenAI API Key Management ---

def save_openai_api_key(api_key: str) -> bool:
    """Saves the OpenAI API key to the user's configuration file.

    The API key is stored in the same TOML formatted file as other configurations,
    under a distinct section. File permissions are set securely.

    Args:
        api_key: The OpenAI API key string to save.

    Returns:
        bool: True if the API key was saved successfully, False otherwise.
    """
    if not api_key or not isinstance(api_key, str):
        logger.error("Attempted to save an invalid or empty OpenAI API key.")
        return False

    config_file_path = get_config_file_path()
    try:
        _ensure_config_dir_exists()
        config_data = _load_config_data(config_file_path)

        if _OPENAI_API_SECTION_NAME not in config_data or \
           not isinstance(config_data[_OPENAI_API_SECTION_NAME], dict):
            config_data[_OPENAI_API_SECTION_NAME] = {}
        
        config_data[_OPENAI_API_SECTION_NAME][_OPENAI_API_KEY_NAME] = api_key

        if _save_config_data(config_file_path, config_data):
            logger.info(f"OpenAI API key saved to {config_file_path}")
            return True
    except Exception as e:
        logger.error(f"General error during OpenAI API key saving process: {e}", exc_info=True)
    return False


def load_openai_api_key() -> Optional[str]:
    """Loads the OpenAI API key from the user's configuration file.

    Returns:
        Optional[str]: The OpenAI API key string if found and valid, otherwise None.
    """
    config_file_path = get_config_file_path()
    if not config_file_path.exists() or not config_file_path.is_file():
        logger.debug(f"Configuration file not found at {config_file_path} for OpenAI API key.")
        return None

    try:
        config_data = _load_config_data(config_file_path)
        api_key = config_data.get(_OPENAI_API_SECTION_NAME, {}).get(_OPENAI_API_KEY_NAME)
        
        if api_key and isinstance(api_key, str):
            logger.debug(f"OpenAI API key loaded successfully from {config_file_path}")
            return api_key
        elif api_key: # Found but not a string
            logger.warning(f"OpenAI API key found in {config_file_path} but is not a valid string.")
        else: # Not found under the expected keys
            logger.debug(
                f"OpenAI API key not found under section "
                f"'{_OPENAI_API_SECTION_NAME}', key '{_OPENAI_API_KEY_NAME}' "
                f"in {config_file_path}"
            )
    except Exception as e:
        logger.error(f"Unexpected error loading OpenAI API key from {config_file_path}: {e}", exc_info=True)
    return None


def delete_openai_api_key() -> bool:
    """Deletes the OpenAI API key from the user's configuration file.

    Returns:
        bool: True if the API key was successfully removed or if it did not exist;
              False if an error occurred.
    """
    config_file_path = get_config_file_path()
    if not config_file_path.exists():
        logger.info("No OpenAI API key to delete: configuration file does not exist.")
        return True

    try:
        config_data = _load_config_data(config_file_path)
        api_section = config_data.get(_OPENAI_API_SECTION_NAME)

        if api_section and isinstance(api_section, dict) and _OPENAI_API_KEY_NAME in api_section:
            del api_section[_OPENAI_API_KEY_NAME]
            logger.info("OpenAI API key removed from configuration data.")
            
            if not api_section: # If the section is now empty
                del config_data[_OPENAI_API_SECTION_NAME]
                logger.info(f"Empty '{_OPENAI_API_SECTION_NAME}' section removed.")

            if _save_config_data(config_file_path, config_data):
                logger.info(f"OpenAI API key deleted from {config_file_path}")
                return True
            return False # Save failed
        else:
            logger.info(f"OpenAI API key not found in {config_file_path}, no deletion performed.")
            return True # Key wasn't there, so considered successful
            
    except Exception as e:
        logger.error(f"Unexpected error deleting OpenAI API key from {config_file_path}: {e}", exc_info=True)
    return False