"""Utility functions for File manipulation"""
from rich.console import Console

def load_video_data(file_path: str, console: Console) -> str:
    """Reads video data from the specified file."""
    try:
        with open(file_path, 'r', encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        console.print(f"[bold red]Error: {file_path} file not found[/bold red]")
        raise