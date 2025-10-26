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
        review_cycles: int,
        total_executions: int,
        successful_executions: int,
        initial_success: bool,
        duration_human: str,
        token_usage_steps: list,
        total_prompt_tokens: int,
        total_completion_tokens: int,
        total_tokens: int,
        video_path: str | None = None,
        args: dict | None = None,
    ) -> None:
        """Save a comprehensive final summary JSON with all key metrics."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "models": {
                "manim_model": manim_model,
                "review_model": review_model,
            },
            "input": {
                "video_data": video_data.strip().replace("\n", ""),
                "args": args,
            },
            "execution_stats": {
                "review_cycles_completed": review_cycles,
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "initial_success": initial_success,
                "final_success": final_success,
            },
            "timing": {
                "total_workflow_time_seconds": workflow_duration_seconds,
                "total_workflow_time_human": duration_human,
                "llm_request_time_seconds": llm_time_seconds,
                "rendering_and_other_time_seconds": workflow_duration_seconds - llm_time_seconds,
            },
            "usage": {
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
                "total_cost_usd": total_cost,
                "steps": token_usage_steps,
            },
            "output": {
                "video_path": video_path,
            },
        }

        summary_file = os.path.join(self.output_dir, "workflow_summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self.console.print(f"[bold cyan]Workflow summary saved to: {summary_file}[/bold cyan]")
