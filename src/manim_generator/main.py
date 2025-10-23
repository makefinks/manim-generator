import time

from rich.console import Console
from rich.panel import Panel

from manim_generator.utils.config import Config
from manim_generator.utils.file import load_video_data
from manim_generator.utils.usage import (
    display_usage_summary,
    format_duration,
    get_usage_totals,
)
from manim_generator.workflow import ManimWorkflow


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
    success, last_frames, combined_logs, successful_scenes = workflow.execute_code(
        current_code, "Initial"
    )
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
        current_code, combined_logs, last_frames, video_data, successful_scenes
    )
    working_code = new_working_code if new_working_code else working_code

    end_time = time.time()
    workflow_duration = end_time - start_time

    video_path = workflow.finalize_output(working_code, current_code, combined_logs)

    if not headless:
        console.rule("[bold cyan]Workflow Summary", style="cyan")
        console.print(
            f"[bold cyan]Total workflow time:[/bold cyan] {format_duration(workflow_duration)}"
        )
        console.print(f"[cyan]Review cycles completed:[/cyan] {workflow.cycles_completed}")
        console.print(f"[cyan]Total executions:[/cyan] {workflow.execution_count}")
        console.print(f"[cyan]Successful executions:[/cyan] {workflow.successful_executions}")
        console.print(f"[cyan]Initial success:[/cyan] {'✓' if workflow.initial_success else '✗'}")
        console.print(
            f"[cyan]Final working code:[/cyan] {'✓' if working_code is not None else '✗'}"
        )
        if video_path:
            console.print(f"[cyan]Final video:[/cyan] {video_path}")

        console.rule("[bold cyan]Token Usage & Cost Summary", style="cyan")
        token_usage_tracking = workflow.usage_tracker.get_tracking_data()
        display_usage_summary(console, token_usage_tracking)
    else:
        console.print(
            f"\n[bold green]✓ Workflow complete in {format_duration(workflow_duration)}[/bold green]"
        )
        console.print(f"[green]Output saved to: {config['output_dir']}/video.py[/green]")

    token_usage_tracking = workflow.usage_tracker.get_tracking_data()
    total_prompt_tokens, total_completion_tokens = get_usage_totals(token_usage_tracking)

    workflow.artifact_manager.save_final_summary(
        manim_model=config["manim_model"],
        review_model=config["review_model"],
        video_data=video_data,
        total_cost=token_usage_tracking["total_cost"],
        workflow_duration_seconds=workflow_duration,
        llm_time_seconds=token_usage_tracking["total_llm_time"],
        final_success=working_code is not None,
        review_cycles=workflow.cycles_completed,
        total_executions=workflow.execution_count,
        successful_executions=workflow.successful_executions,
        initial_success=workflow.initial_success,
        duration_human=format_duration(workflow_duration),
        token_usage_steps=token_usage_tracking["steps"],
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_tokens=token_usage_tracking["total_tokens"],
        video_path=video_path,
        args=config,
    )


if __name__ == "__main__":
    main()
