"""Utility functions for File manipulation"""

import logging
import os

from rich.console import Console

logger = logging.getLogger(__name__)


def load_video_data(file_path: str, console: Console) -> str:
    """Reads video data from the specified file."""
    try:
        with open(file_path, encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        console.print(f"[bold red]Error: {file_path} file not found[/bold red]")
        raise


def save_code_to_file(code: str, filename: str = "video.py") -> str:
    """
    Saves the generated code to a Python file.

    Args:
        code: String containing the Python code to save
        filename: Name of the file to save to (defaults to video.py)

    Returns:
        str: Path to the saved file if successful, empty string if failed
    """
    try:
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(filename, "w", encoding="utf-8") as f:
            f.write(code)
        return filename
    except Exception as e:
        logger.exception(e)
        return ""
