"""Utility functions for preserving workflow artifacts and debugging information."""

import json
import os
from datetime import datetime

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

    def save_final_summary(
        self,
        manim_model: str,
        review_model: str,
        video_data: str,
        total_cost: float,
        workflow_duration_seconds: float,
        llm_time_seconds: float,
        final_success: bool,
    ) -> None:
        """Save a comprehensive final summary JSON with all key metrics."""
        summary = {
            "models": {
                "manim_model": manim_model,
                "review_model": review_model,
            },
            "input": {
                "video_data": video_data,
            },
            "cost": {
                "total_usd": total_cost,
            },
            "timing": {
                "total_workflow_time_seconds": workflow_duration_seconds,
                "llm_request_time_seconds": llm_time_seconds,
                "rendering_and_other_time_seconds": workflow_duration_seconds - llm_time_seconds,
            },
            "status": {
                "final_success": final_success,
            },
            "timestamp": datetime.now().isoformat(),
        }

        summary_file = os.path.join(self.output_dir, "final_summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self.console.print(f"[bold cyan]Final summary saved to: {summary_file}[/bold cyan]")
