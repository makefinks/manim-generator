import json
import os
from datetime import datetime
from rich.console import Console
from rich.table import Table


def display_usage_summary(console: Console, token_usage_tracking: dict):
    """Display a summary table of token usage and costs."""
    table = Table(title="Token Usage & Cost Summary")
    table.add_column("Step", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Prompt Tokens", justify="right", style="blue")
    table.add_column("Completion Tokens", justify="right", style="blue") 
    table.add_column("Total Tokens", justify="right", style="blue")
    table.add_column("Cost (USD)", justify="right", style="red")
    
    # Calculate totals
    total_prompt_tokens = sum(step.get("prompt_tokens", 0) for step in token_usage_tracking["steps"])
    total_completion_tokens = sum(step.get("completion_tokens", 0) for step in token_usage_tracking["steps"])
    
    for step in token_usage_tracking["steps"]:
        table.add_row(
            step["step"],
            step["model"],
            str(step.get("prompt_tokens", 0)),
            str(step.get("completion_tokens", 0)),
            str(step.get("total_tokens", 0)),
            f"${step.get('cost', 0.0):.6f}"
        )
    
    table.add_section()
    table.add_row(
        "[bold]TOTAL",
        "",
        f"[bold]{total_prompt_tokens}",
        f"[bold]{total_completion_tokens}",
        f"[bold]{token_usage_tracking['total_tokens']}",
        f"[bold]${token_usage_tracking['total_cost']:.6f}"
    )
    
    console.print(table)


def save_usage_report(output_dir: str, token_usage_tracking: dict, console: Console):
    """Save detailed usage report to a JSON file."""
    # Calculate separate totals
    total_prompt_tokens = sum(step.get("prompt_tokens", 0) for step in token_usage_tracking["steps"])
    total_completion_tokens = sum(step.get("completion_tokens", 0) for step in token_usage_tracking["steps"])
    
    report = {
        "summary": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": token_usage_tracking["total_tokens"],
            "total_cost": token_usage_tracking["total_cost"],
            "timestamp": datetime.now().isoformat()
        },
        "steps": token_usage_tracking["steps"]
    }
    
    report_file = os.path.join(output_dir, "token_usage_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    console.print(f"[bold cyan]Token usage report saved to: {report_file}[/bold cyan]")
