"""Utility functions for preserving workflow artifacts and debugging information."""

import os

from rich.console import Console


class ArtifactManager:
    """Manages preservation of workflow artifacts"""

    def __init__(self, output_dir: str, console: Console):
        self.output_dir = output_dir
        self.console = console
        self.steps_dir = os.path.join(output_dir, "steps")
        os.makedirs(self.steps_dir, exist_ok=True)

    def _write_file(self, directory: str, filename: str, content: str | None) -> None:
        """Write content to a file if content is provided."""
        if content:
            with open(os.path.join(directory, filename), "w", encoding="utf-8") as f:
                f.write(content)

    def save_step_artifacts(
        self,
        step_name: str,
        code: str | None = None,
        prompt: str | None = None,
        logs: str | None = None,
        review_text: str | None = None,
        reasoning: str | None = None,
    ) -> str:
        """Save all artifacts for a workflow step."""
        step_dir = os.path.join(self.steps_dir, step_name)
        os.makedirs(step_dir, exist_ok=True)

        file_mappings = {
            "code.py": code,
            "prompt.txt": prompt,
            "logs.txt": logs,
            "review.md": review_text,
            "reasoning.txt": reasoning,
        }

        # save all
        for filename, content in file_mappings.items():
            self._write_file(step_dir, filename, content)

        return step_dir

    def get_step_frames_path(self, step_name: str) -> str:
        """Get the path where frames should be saved for a step."""
        step_dir = os.path.join(self.steps_dir, step_name)
        frames_dir = os.path.join(step_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        return frames_dir

