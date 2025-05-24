from datetime import datetime
import argparse
import os
from rich.console import Console
from rich.prompt import Confirm
from rich.panel import Panel
from rich.markdown import Markdown
from litellm import supports_vision

from utils.code import (
    extract_scene_class_names,
    run_manim_multiscene,
    parse_code_block,
    save_code_to_file,
)
from utils.console import get_response_with_status, print_code_with_syntax
from utils.text import (
    convert_frames_to_message_format,
    format_previous_reviews,
    format_prompt,
)
from utils.video import render_and_concat
from utils.file import load_video_data
from utils.usage import display_usage_summary, save_usage_report

# Default configuration
DEFAULT_CONFIG = {
    "manim_model": "gemini/gemini-2.5-pro-exp-03-25",
    "review_model": "gemini/gemini-2.5-pro-exp-03-25",
    "review_cycles": 5,
    "output_dir": f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    "manim_logs": False,
    "streaming": False,
    "temperature": 0.4,
}

console = Console()

# Global tracking for token usage
token_usage_tracking = {"steps": [], "total_tokens": 0, "total_cost": 0.0}

# litellm._turn_on_debug()


def parse_arguments():
    """Parse command line arguments and return the configuration."""
    parser = argparse.ArgumentParser(description="Generate Manim animations using AI")

    # Video data input
    parser.add_argument(
        "--video_data", type=str, help="Description of the video to generate"
    )
    parser.add_argument(
        "--video_data_file",
        type=str,
        default="video_data.txt",
        help="Path to file containing video description",
    )

    # Model configuration
    parser.add_argument(
        "--manim_model",
        type=str,
        default=DEFAULT_CONFIG["manim_model"],
        help="Model to use for generating Manim code",
    )
    parser.add_argument(
        "--review_model",
        type=str,
        default=DEFAULT_CONFIG["review_model"],
        help="Model to use for reviewing code",
    )

    # Process configuration
    parser.add_argument(
        "--review_cycles",
        type=int,
        default=DEFAULT_CONFIG["review_cycles"],
        help="Number of review cycles to perform",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_CONFIG["output_dir"],
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--manim_logs",
        action="store_true",
        default=DEFAULT_CONFIG["manim_logs"],
        help="Show Manim execution logs",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Enable streaming responses from the model",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_CONFIG["temperature"],
        help="Temperature for the LLM Model",
    )

    parser.add_argument(
        "--force_vision",
        action="store_true",
        default=False,
        help="Adds images to the review process, regardless if LiteLLM reports vision is not supported. (Check API provider)",
    )

    # Reasoning tokens configuration
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        choices=["low", "medium", "high"],
        help="Reasoning effort level for OpenAI-style models (low/medium/high)",
    )
    parser.add_argument(
        "--reasoning_max_tokens",
        type=int,
        help="Maximum tokens for reasoning (Anthropic-style)",
    )
    parser.add_argument(
        "--reasoning_exclude",
        action="store_true",
        default=False,
        help="Exclude reasoning tokens from response (model still uses reasoning internally)",
    )

    args = parser.parse_args()

    # Validate reasoning arguments - ensure only one method is used
    if args.reasoning_effort and args.reasoning_max_tokens:
        console.print(
            "[bold red]Error: Cannot use both --reasoning_effort and --reasoning_max_tokens at the same time.[/bold red]"
        )
        exit(1)

    # Check if the model supports vision/images
    vision_enabled = supports_vision(model=args.review_model) or args.force_vision
    console.print(
        f"[bold {'green' if vision_enabled else 'yellow'}]Vision support: {'Enabled' if vision_enabled else 'Disabled'}[/bold {'green' if vision_enabled else 'yellow'}]"
    )

    # Build reasoning config
    reasoning_config = {}
    if args.reasoning_effort:
        reasoning_config["effort"] = args.reasoning_effort
    if args.reasoning_max_tokens:
        reasoning_config["max_tokens"] = args.reasoning_max_tokens
    if args.reasoning_exclude:
        reasoning_config["exclude"] = True

    # Build config from arguments
    config = {
        "manim_model": args.manim_model,
        "review_model": args.review_model,
        "review_cycles": args.review_cycles,
        "output_dir": args.output_dir,
        "manim_logs": args.manim_logs,
        "streaming": args.streaming,
        "temperature": args.temperature,
        "vision_enabled": vision_enabled,
        "reasoning": reasoning_config if reasoning_config else None,
    }

    # Create output directory if it doesn't exist
    os.makedirs(config["output_dir"], exist_ok=True)

    return config, args.video_data, args.video_data_file


def generate_initial_code(video_data: str, config: dict) -> tuple[str, list]:
    """Generates the initial Manim code based on video data.

    Takes the video data and creates the initial manim code using the code model.

    Args:
        video_data (str): Description of the video/animation to be generated
        config (dict): Configuration dictionary

    Returns:
        tuple[str, list]: A tuple containing:
            - The generated Manim code as a string
            - The conversation history with the model as a list of messages
    """

    console.rule("[bold green]Initial Manim Code Generation", style="green")
    main_messages = [
        {
            "role": "system",
            "content": format_prompt("init_prompt", {"video_data": video_data}),
        }
    ]
    response, usage_info, reasoning_content = get_response_with_status(
        config["manim_model"],
        main_messages,
        config["temperature"],
        config["streaming"],
        f"[bold green]Generating initial code \\[{config['manim_model']}\]",
        console,
        reasoning=config["reasoning"],
    )

    # Track usage
    step_info = {
        "step": "Initial Code Generation",
        "model": config["manim_model"],
        **usage_info,
    }
    token_usage_tracking["steps"].append(step_info)
    token_usage_tracking["total_tokens"] += usage_info.get("total_tokens", 0)
    token_usage_tracking["total_cost"] += usage_info.get("cost", 0.0)

    console.clear()
    
    # Display reasoning content if available and not streaming
    if reasoning_content and not config["streaming"]:
        console.print(
            Panel(
                reasoning_content,
                title="[yellow]Model Reasoning[/yellow]",
                border_style="yellow",
            )
        )
    
    code = parse_code_block(response)
    print_code_with_syntax(code, console, "Generated Initial Manim Code")
    return code, main_messages


def review_and_update_code(
    current_code: str,
    main_messages: list,
    combined_logs: str,
    last_frames: list,
    config: dict,
    video_data: str,
) -> tuple[str, str | None, str]:
    """Performs review cycles and updates the Manim code based on model feedback.

    Args:
        current_code (str): The current Manim code to be reviewed and updated
        main_messages (list): The conversation history with the code generation model
        combined_logs (str): Logs from previous Manim script executions
        last_frames (list): List of rendered frames from previous executions
        config (dict): Configuration dictionary

    Returns:
        tuple[str, str | None, str]: A tuple containing:
            - The final version of the code after all review cycles
            - The last working version of the code (None if no version executed successfully)
            - The combined logs from all Manim executions
    """
    working_code = None
    previous_reviews = []

    for cycle in range(config["review_cycles"]):
        console.rule(f"[bold blue]Review Cycle {cycle + 1}", style="blue")

        frames_formatted = (
            convert_frames_to_message_format(last_frames)
            if last_frames and config["vision_enabled"]
            else []
        )

        console.print(f"[blue] Adding {len(frames_formatted)} images to the review")
        review_content = format_prompt(
            "review_prompt",
            {
                "previous_reviews": format_previous_reviews(previous_reviews),
                "video_code": current_code,
                "execution_logs": combined_logs,
            },
        )
        review_message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": review_content},
                ]
                + frames_formatted,
            }
        ]

        review, review_usage_info, review_reasoning = get_response_with_status(
            config["review_model"],
            review_message,
            config["temperature"],
            config["streaming"],
            status=f"[bold blue]Generating Review \\[{config['review_model']}\\]",
            console=console,
            reasoning=config["reasoning"],
        )

        # Track review usage
        review_step_info = {
            "step": f"Review Cycle {cycle + 1}",
            "model": config["review_model"],
            **review_usage_info,
        }
        token_usage_tracking["steps"].append(review_step_info)
        token_usage_tracking["total_tokens"] += review_usage_info.get("total_tokens", 0)
        token_usage_tracking["total_cost"] += review_usage_info.get("cost", 0.0)
        previous_reviews.append(review)
        
        # Display reasoning content if available and not streaming
        if review_reasoning and not config["streaming"]:
            console.print(
                Panel(
                    review_reasoning,
                    title="[yellow]Review Model Reasoning[/yellow]",
                    border_style="yellow",
                )
            )
        
        console.print(
            Panel(
                Markdown(review),
                title="[blue]Review Feedback[/blue]",
                border_style="blue",
            )
        )

        # Create focused messages for code revision (without full conversation history)
        revision_messages = [
            {
                "role": "system",
                "content": format_prompt("init_prompt", {"video_data": video_data}),
            },
            {
                "role": "user",
                "content": f"Here is the current code:\n\n```python\n{current_code}\n```\n\nHere is some feedback on your code:\n\n<review>\n{review}\n</review>\n\nPlease implement the suggestions and respond with the whole script. Do not leave anything out.",
            },
        ]

        console.rule(f"[bold green]Generating Code Revision {cycle + 1}", style="green")
        revised_response, revision_usage_info, revision_reasoning = get_response_with_status(
            config["manim_model"],
            revision_messages,
            config["temperature"],
            config["streaming"],
            f"[bold green]Generating code revision \\[{config['manim_model']}]",
            console,
            reasoning=config["reasoning"],
        )

        # Display reasoning content if available and not streaming
        if revision_reasoning and not config["streaming"]:
            console.print(
                Panel(
                    revision_reasoning,
                    title="[yellow]Model Reasoning[/yellow]",
                    border_style="yellow",
                )
            )

        # Track revision usage
        revision_step_info = {
            "step": f"Code Revision {cycle + 1}",
            "model": config["manim_model"],
            **revision_usage_info,
        }
        token_usage_tracking["steps"].append(revision_step_info)
        token_usage_tracking["total_tokens"] += revision_usage_info.get(
            "total_tokens", 0
        )
        token_usage_tracking["total_cost"] += revision_usage_info.get("cost", 0.0)
        current_code = parse_code_block(revised_response)
        print_code_with_syntax(
            current_code, console, f"Revised Code - Cycle {cycle + 1}"
        )

        console.rule(
            f"[bold green]Running Manim Script - Revision {cycle + 1}", style="green"
        )
        success, last_frames, combined_logs = run_manim_multiscene(
            current_code, console, config["output_dir"]
        )

        status_color = "green" if success else "red"

        scene_names = extract_scene_class_names(current_code)
        scenes_rendered = f"{len(last_frames)} of {len(scene_names) if isinstance(scene_names, list) else '? (Syntax error)'}"
        console.print(
            f"[bold {status_color}]Execution Status: {'Success' if success else 'Failed'}[/bold {status_color}]"
        )
        console.print(
            f"[bold {status_color}]Scenes Rendered: {scenes_rendered}[/bold {status_color}]"
        )

        # update working_code if this iteration was successful
        if success:
            working_code = current_code

        if config["manim_logs"] or not success:
            # Always show logs if there was an error, regardless of manim_logs setting
            log_title = (
                "[green]Execution Logs[/green]"
                if success
                else "[red]Execution Errors[/red]"
            )
            log_style = "green" if success else "red"
            console.print(
                Panel(
                    combined_logs,
                    title=log_title,
                    border_style=log_style,
                )
            )

    return current_code, working_code, combined_logs


def main():
    config, video_data_arg, video_data_file = parse_arguments()

    # Check and register models for cost calculation
    from utils.llm import check_and_register_models
    models_to_check = [config["manim_model"], config["review_model"]]
    check_and_register_models(models_to_check, console)

    # Get video data from argument or file
    if video_data_arg:
        video_data = video_data_arg
    else:
        video_data = load_video_data(video_data_file, console)

    current_code, main_messages = generate_initial_code(video_data, config)

    console.rule("[bold green]Running Initial Manim Script", style="green")
    success, last_frames, combined_logs = run_manim_multiscene(
        current_code, console, config["output_dir"]
    )

    status_color = "green" if success else "red"

    extract_output = extract_scene_class_names(current_code)
    all_scenes_count = (
        "N/A" if isinstance(extract_output, Exception) else len(extract_output)
    )
    scenes_rendered = f"{len(last_frames)} of {all_scenes_count}"
    console.print(
        f"[bold {status_color}]Execution Status: {'Success' if success else 'Failed'}[/bold {status_color}]"
    )
    console.print(
        f"[bold {status_color}]Scenes Rendered: {scenes_rendered}[/bold {status_color}]"
    )

    if config["manim_logs"]:
        console.print(
            Panel(
                combined_logs,
                title="[green]Execution Logs[/green]",
                border_style="green",
            )
        )

    working_code = current_code if success else None

    if not working_code:
        console.print(
            Panel(
                "[bold red]Initial code failed to execute properly. Starting review cycles to fix issues.[/bold red]",
                border_style="red",
            )
        )

    # Execute review cycles - even if initial code failed, the review process might fix it
    current_code, new_working_code, combined_logs = review_and_update_code(
        current_code, main_messages, combined_logs, last_frames, config, video_data
    )
    working_code = new_working_code if new_working_code else working_code

    # Display and save usage summary
    console.rule("[bold cyan]Token Usage & Cost Summary", style="cyan")
    display_usage_summary(console, token_usage_tracking)
    save_usage_report(config["output_dir"], token_usage_tracking, console)

    # Final output
    if working_code:
        console.rule("[bold green]Final Result", style="green")
        print_code_with_syntax(working_code, console, "Final Manim Code")
        saved_file = save_code_to_file(
            working_code, filename=f"{config['output_dir']}/video.py"
        )
        console.print(f"[bold green]Code saved to: {saved_file}[/bold green]")

        console.rule("[bold blue]Rendering Options", style="blue")
        if Confirm.ask(
            "[bold blue]Would you like to render the final video?[/bold blue]"
        ):
            console.rule("[bold blue]Rendering Final Video", style="blue")
            render_and_concat(saved_file, config["output_dir"], "final_video.mp4")
    else:
        console.rule(
            "[bold red]Final Result - With Errors (Not Executable)", style="red"
        )
        print_code_with_syntax(current_code, console, "Final Manim Code (with errors)")
        console.print(
            Panel(
                combined_logs, title="[red]Execution Errors[/red]", border_style="red"
            )
        )


if __name__ == "__main__":
    main()
