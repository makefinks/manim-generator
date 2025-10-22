from rich.console import Console
from rich.table import Table


class TokenUsageTracker:
    """Tracks token usage and costs across workflow steps."""

    def __init__(self):
        self.token_usage_tracking = {
            "steps": [],
            "total_tokens": 0,
            "total_cost": 0.0,
            "total_llm_time": 0.0,
        }

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
        self.token_usage_tracking["total_llm_time"] += usage_info.get("llm_time", 0.0)

    def get_tracking_data(self) -> dict:
        """Get the complete tracking data."""
        return self.token_usage_tracking


def get_usage_totals(token_usage_tracking: dict) -> tuple[int, int]:
    """Calculate total prompt and completion tokens from usage tracking data."""
    total_prompt_tokens = sum(
        step.get("prompt_tokens", 0) for step in token_usage_tracking["steps"]
    )
    total_completion_tokens = sum(
        step.get("completion_tokens", 0) for step in token_usage_tracking["steps"]
    )
    return total_prompt_tokens, total_completion_tokens


def display_usage_summary(console: Console, token_usage_tracking: dict):
    """Display a summary table of token usage and costs."""
    table = Table(title="Token Usage & Cost Summary")
    table.add_column("Step", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Prompt Tokens", justify="right", style="blue")
    table.add_column("Completion Tokens", justify="right", style="blue")
    table.add_column("Total Tokens", justify="right", style="blue")
    table.add_column("Cost (USD)", justify="right", style="red")

    total_prompt_tokens, total_completion_tokens = get_usage_totals(token_usage_tracking)

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
