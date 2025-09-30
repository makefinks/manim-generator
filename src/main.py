import time
from rich.console import Console
from rich.panel import Panel

from src.utils.config import Config
from src.workflow import ManimWorkflow
from src.utils.file import load_video_data
from src.utils.usage import (
    display_usage_summary,
    save_usage_report,
    save_workflow_metadata,
    format_duration,
)


def main():
    start_time = time.time()
    console = Console()

    config_manager = Config()
    config, video_data_arg, video_data_file = config_manager.parse_arguments()
    
    headless = config.get("headless", False)

    if video_data_arg:
        video_data = video_data_arg
    else:
        video_data = load_video_data(video_data_file, console)

    workflow = ManimWorkflow(config, console)

    current_code, main_messages = workflow.generate_initial_code(video_data)
    success, last_frames, combined_logs = workflow.execute_code(current_code, "Initial")
    workflow.initial_success = success
    working_code = current_code if success else None

    if not working_code and not headless:
        console.print(
            Panel(
                "[bold red]Initial code failed to execute properly. Starting review cycles to fix issues.[/bold red]",
                border_style="red",
            )
        )

    current_code, new_working_code, combined_logs = workflow.review_and_update_code(
        current_code, combined_logs, last_frames, video_data
    )
    working_code = new_working_code if new_working_code else working_code

    end_time = time.time()
    workflow_duration = end_time - start_time

    if not headless:
        console.rule("[bold cyan]Workflow Summary", style="cyan")
        console.print(
            f"[bold cyan]Total workflow time:[/bold cyan] {format_duration(workflow_duration)}"
        )
        console.print(f"[cyan]Review cycles completed:[/cyan] {workflow.cycles_completed}")
        console.print(f"[cyan]Total executions:[/cyan] {workflow.execution_count}")
        console.print(
            f"[cyan]Successful executions:[/cyan] {workflow.successful_executions}"
        )
        console.print(
            f"[cyan]Initial success:[/cyan] {'✓' if workflow.initial_success else '✗'}"
        )
        console.print(
            f"[cyan]Final working code:[/cyan] {'✓' if working_code is not None else '✗'}"
        )

        console.rule("[bold cyan]Token Usage & Cost Summary", style="cyan")
        token_usage_tracking = workflow.usage_tracker.get_tracking_data()
        display_usage_summary(console, token_usage_tracking)
        save_usage_report(config["output_dir"], token_usage_tracking, console)
    else:
        token_usage_tracking = workflow.usage_tracker.get_tracking_data()
        save_usage_report(config["output_dir"], token_usage_tracking, console)
        console.print(f"\n[bold green]✓ Workflow complete in {format_duration(workflow_duration)}[/bold green]")
        console.print(f"[green]Output saved to: {config['output_dir']}/video.py[/green]")

    save_workflow_metadata(
        config["output_dir"],
        {
            "duration_seconds": workflow_duration,
            "review_cycles": workflow.cycles_completed,
            "total_executions": workflow.execution_count,
            "successful_executions": workflow.successful_executions,
            "initial_success": workflow.initial_success,
            "final_success": working_code is not None,
            "video_data": video_data,
        },
        console,
    )

    workflow.finalize_output(working_code, current_code, combined_logs)


if __name__ == "__main__":
    main()
