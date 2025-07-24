"""Utility functions for preserving workflow artifacts and debugging information."""
import os
import shutil
from typing import Optional
from rich.console import Console


class ArtifactManager:
    """Manages preservation of workflow artifacts for debugging and analysis."""

    def __init__(self, output_dir: str, console: Console):
        self.output_dir = output_dir
        self.console = console
        self.steps_dir = os.path.join(output_dir, "steps")
        os.makedirs(self.steps_dir, exist_ok=True)
    
    def _write_file(self, directory: str, filename: str, content: Optional[str]) -> None:
        """Write content to a file if content is provided."""
        if content:
            with open(os.path.join(directory, filename), "w", encoding="utf-8") as f:
                f.write(content)
    
    def save_step_artifacts(
        self, 
        step_name: str, 
        code: Optional[str] = None, 
        prompt: Optional[str] = None, 
        logs: Optional[str] = None, 
        frames_dir: Optional[str] = None,
        review_text: Optional[str] = None,
        reasoning: Optional[str] = None
    ) -> str:
        """Save all artifacts for a workflow step."""
        step_dir = os.path.join(self.steps_dir, step_name)
        os.makedirs(step_dir, exist_ok=True)
        
        file_mappings = {
            "code.py": code,
            "prompt.txt": prompt,
            "logs.txt": logs,
            "review.md": review_text,
            "reasoning.txt": reasoning
        }
        
        # Save all files 
        for filename, content in file_mappings.items():
            self._write_file(step_dir, filename, content)
        
        # Copy frames if they exist
        if frames_dir and os.path.exists(frames_dir):
            frames_dest = os.path.join(step_dir, "frames")
            if os.path.exists(frames_dest):
                shutil.rmtree(frames_dest)
            shutil.copytree(frames_dir, frames_dest)
        
        return step_dir
    
    def get_step_frames_path(self, step_name: str) -> str:
        """Get the path where frames should be saved for a step."""
        step_dir = os.path.join(self.steps_dir, step_name)
        frames_dir = os.path.join(step_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        return frames_dir
