import json
import os
from datetime import datetime
from rich.console import Console
from rich.table import Table


class TokenUsageTracker:
    """Tracks token usage and costs across workflow steps."""

    def __init__(self):
        self.token_usage_tracking = {"steps": [], "total_tokens": 0, "total_cost": 0.0}

    def add_step(self, step_name: str, model: str, usage_info: dict) -> None:
        """Add a step's usage information to tracking."""
        step_info = {
            "step": step_name,
            "model": model,
            **usage_info,
        }
        self.token_usage_tracking["steps"].append(step_info)
        self.token_usage_tracking["total_tokens"] += usage_info.get("total_tokens", 0)
        self.token_usage_tracking["total_cost"] += usage_info.get("cost", 0.0)

    def get_tracking_data(self) -> dict:
        """Get the complete tracking data."""
        return self.token_usage_tracking


def display_usage_summary(console: Console, token_usage_tracking: dict):
    """Display a summary table of token usage and costs."""
    table = Table(title="Token Usage & Cost Summary")
    table.add_column("Step", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Prompt Tokens", justify="right", style="blue")
    table.add_column("Completion Tokens", justify="right", style="blue")
    table.add_column("Total Tokens", justify="right", style="blue")
    table.add_column("Cost (USD)", justify="right", style="red")

    total_prompt_tokens = sum(
        step.get("prompt_tokens", 0) for step in token_usage_tracking["steps"]
    )
    total_completion_tokens = sum(
        step.get("completion_tokens", 0) for step in token_usage_tracking["steps"]
    )

    for step in token_usage_tracking["steps"]:
        table.add_row(
            step["step"],
            step["model"],
            str(step.get("prompt_tokens", 0)),
            str(step.get("completion_tokens", 0)),
            str(step.get("total_tokens", 0)),
            f"${step.get('cost', 0.0):.6f}",
        )

    table.add_section()
    table.add_row(
        "[bold]TOTAL",
        "",
        f"[bold]{total_prompt_tokens}",
        f"[bold]{total_completion_tokens}",
        f"[bold]{token_usage_tracking['total_tokens']}",
        f"[bold]${token_usage_tracking['total_cost']:.6f}",
    )

    console.print(table)


def save_usage_report(output_dir: str, token_usage_tracking: dict, console: Console):
    """Save detailed usage report to a JSON file."""
    total_prompt_tokens = sum(
        step.get("prompt_tokens", 0) for step in token_usage_tracking["steps"]
    )
    total_completion_tokens = sum(
        step.get("completion_tokens", 0) for step in token_usage_tracking["steps"]
    )

    report = {
        "summary": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": token_usage_tracking["total_tokens"],
            "total_cost": token_usage_tracking["total_cost"],
            "timestamp": datetime.now().isoformat(),
        },
        "steps": token_usage_tracking["steps"],
    }

    report_file = os.path.join(output_dir, "token_usage_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    console.print(f"[bold cyan]Token usage report saved to: {report_file}[/bold cyan]")


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60
        return f"{hours}h {remaining_minutes}m {remaining_seconds:.1f}s"


def save_workflow_metadata(output_dir: str, metadata: dict, console: Console):
    """Save workflow metadata including timing and execution statistics."""
    full_metadata = {
        "workflow_timing": {
            "duration_seconds": metadata["duration_seconds"],
            "duration_human": format_duration(metadata["duration_seconds"]),
            "timestamp": datetime.now().isoformat(),
        },
        "execution_stats": {
            "review_cycles_completed": metadata["review_cycles"],
            "total_executions": metadata["total_executions"],
            "successful_executions": metadata["successful_executions"],
            "initial_success": metadata["initial_success"],
            "final_success": metadata["final_success"],
        },
        "input_data": {"video_data": metadata["video_data"]},
    }

    metadata_file = os.path.join(output_dir, "workflow_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(full_metadata, f, indent=2)

    console.print(f"[bold cyan]Workflow metadata saved to: {metadata_file}[/bold cyan]")
