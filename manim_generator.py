from rich.console import Console
from rich.panel import Panel

from config import Config
from manim_workflow import ManimWorkflow
from utils.file import load_video_data
from utils.usage import display_usage_summary, save_usage_report


def main():
    """Main entry point for the Manim generator application."""
    console = Console()
    
    # Parse configuration
    config_manager = Config()
    config, video_data_arg, video_data_file = config_manager.parse_arguments()
    
    # Get video data from argument or file
    if video_data_arg:
        video_data = video_data_arg
    else:
        video_data = load_video_data(video_data_file, console)
    
    # Initialize workflow
    workflow = ManimWorkflow(config, console)
    
    # Generate initial code
    current_code, main_messages = workflow.generate_initial_code(video_data)
    
    # Execute initial code
    success, last_frames, combined_logs = workflow.execute_code(current_code, "Initial")
    working_code = current_code if success else None
    
    if not working_code:
        console.print(
            Panel(
                "[bold red]Initial code failed to execute properly. Starting review cycles to fix issues.[/bold red]",
                border_style="red",
            )
        )
    
    # Execute review cycles - even if initial code failed, the review process might fix it
    current_code, new_working_code, combined_logs = workflow.review_and_update_code(
        current_code, main_messages, combined_logs, last_frames, video_data
    )
    working_code = new_working_code if new_working_code else working_code
    
    # Display and save usage summary
    console.rule("[bold cyan]Token Usage & Cost Summary", style="cyan")
    token_usage_tracking = workflow.usage_tracker.get_tracking_data()
    display_usage_summary(console, token_usage_tracking)
    save_usage_report(config["output_dir"], token_usage_tracking, console)
    
    # Final output handling
    workflow.finalize_output(working_code, current_code, combined_logs)


if __name__ == "__main__":
    main()
