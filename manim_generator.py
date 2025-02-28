from datetime import datetime
import argparse
import os
from rich.console import Console
from rich.syntax import Syntax
from rich.prompt import Confirm
from litellm import supports_vision

from utils.code import (
    extract_scene_class_names,
    run_manim_multiscene,
    parse_code_block,
    save_code_to_file,
)
from utils.console import get_response_with_status
from utils.text import convert_frames_to_message_format, format_previous_reviews, format_prompt
from utils.video import render_and_concat
from utils.file import load_video_data


# Default configuration
DEFAULT_CONFIG = {
    "manim_model": "openrouter/anthropic/claude-3.7-sonnet",
    "review_model": "openrouter/anthropic/claude-3.7-sonnet",
    "review_cycles": 3,
    "output_dir": f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    "manim_logs": False,
    "streaming": False,
}

console = Console()

def parse_arguments():
    """Parse command line arguments and return the configuration."""
    parser = argparse.ArgumentParser(description="Generate Manim animations using AI")
    
    # Video data input
    parser.add_argument("--video_data", type=str, help="Description of the video to generate")
    parser.add_argument("--video_data_file", type=str, default="video_data.txt", 
                        help="Path to file containing video description")
    
    # Model configuration
    parser.add_argument("--manim_model", type=str, default=DEFAULT_CONFIG["manim_model"],
                        help="Model to use for generating Manim code")
    parser.add_argument("--review_model", type=str, default=DEFAULT_CONFIG["review_model"],
                        help="Model to use for reviewing code")
    
    # Process configuration
    parser.add_argument("--review_cycles", type=int, default=DEFAULT_CONFIG["review_cycles"],
                        help="Number of review cycles to perform")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG["output_dir"],
                        help="Directory to save outputs")
    parser.add_argument("--manim_logs", action="store_true", default=DEFAULT_CONFIG["manim_logs"],
                        help="Show Manim execution logs")
    parser.add_argument("--streaming", action="store_true", 
                        help="Enable streaming responses from the model")
    
    args = parser.parse_args()
    
    # Build config from arguments
    config = {
        "manim_model": args.manim_model,
        "review_model": args.review_model,
        "review_cycles": args.review_cycles,
        "output_dir": args.output_dir,
        "manim_logs": args.manim_logs,
        "streaming": args.streaming,
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
    console.rule("Initial Manim Code Generation")
    main_messages = [{
        "role": "system",
        "content": format_prompt("init_prompt", {"video_data": video_data}),
    }]
    response = get_response_with_status(config["manim_model"], main_messages, 0.7, config["streaming"], "[bold green] Generating initial code", console)
    console.clear()
    console.print("[bold green]Generated initial Manim code[/bold green]")
    code = parse_code_block(response)
    console.print(Syntax(code, "python", theme="monokai", line_numbers=True))
    return code, main_messages


def review_and_update_code(current_code: str, main_messages: list, combined_logs: str, last_frames: list, config: dict) -> tuple[str, str | None, str]:
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
    vision_enabled = supports_vision(model=config["review_model"])

    for cycle in range(config["review_cycles"]):
        console.rule(f"Review Cycle {cycle + 1}", style="bold blue")
        review_content = format_prompt(
            "review_prompt",
            {
                "previous_reviews": format_previous_reviews(previous_reviews),
                "video_code": current_code,
                "execution_logs": combined_logs
            },
        )
        review_message = [{
            "role": "system",
            "content": [
                {"type": "text", "text": review_content},
            ] + (convert_frames_to_message_format(last_frames) if last_frames and vision_enabled else [])
        }]
        review = get_response_with_status(config["review_model"], review_message, 0.7, config["streaming"], status="[bold blue] Generating Review", console=console)
        previous_reviews.append(review)
        console.print(f"[yellow]Received review feedback:[/yellow]\n{review}")

        # Append feedback for code revision
        main_messages.append({
            "role": "user",
            "content": f"Here is some feedback on your code. \n\n<review{review}</review>\n\nPlease implement the suggestions and respond with the whole script."
        })

        console.rule(f"Generating Manim Code Revision {cycle + 1}")
        revised_response = get_response_with_status(config["manim_model"], main_messages, 0.7, config["streaming"], "[bold green] generating code revision", console)
        main_messages.append({"role": "assistant", "content": revised_response})
        current_code = parse_code_block(revised_response)
        console.print(f"[green]Revised code generated for cycle {cycle + 1}[/green]")
        console.print(Syntax(current_code, "python", theme="monokai", line_numbers=True))
        console.rule(f"Running manim script - Revision {cycle + 1}")

        success, last_frames, combined_logs = run_manim_multiscene(current_code, console, config["output_dir"])
        console.print(f"Success: {success}\n{len(last_frames)} of {len(extract_scene_class_names(current_code))} scenes rendered successfully")
        
        # update working_code if this iteration was successful
        if success:
            working_code = current_code
        
        if config["manim_logs"]:
            console.print(f"Output:\n{combined_logs}")

    return current_code, working_code, combined_logs


def main():
    config, video_data_arg, video_data_file = parse_arguments()
    
    # Get video data from argument or file
    if video_data_arg:
        video_data = video_data_arg
    else:
        video_data = load_video_data(video_data_file, console)
    
    current_code, main_messages = generate_initial_code(video_data, config)
    
    console.rule("Running initial manim script")
    success, last_frames, combined_logs = run_manim_multiscene(current_code, console, config["output_dir"])
    console.print(f"Success: {success}\n{len(last_frames)} of {len(extract_scene_class_names(current_code))} scenes rendered successfully")
    if config["manim_logs"]:
        console.print(f"Output:\n{combined_logs}")
    working_code = current_code if success else None

    if not working_code:
        console.print("[bold red]Initial code failed to execute properly.[/bold red]")
    
    # Execute review cycles
    current_code, new_working_code, combined_logs = review_and_update_code(current_code, main_messages, combined_logs, last_frames, config)
    working_code = new_working_code if new_working_code else working_code

    # Final output
    if working_code:
        console.rule("Final Result", style="bold green")
        console.print("\n[bold underline]Final Manim Code:[/bold underline]\n")
        console.print(working_code)
        saved_file = save_code_to_file(working_code, filename=f"{config['output_dir']}/video.py")

        console.rule("Rendering", style="bold blue")
        if Confirm.ask("[bold yellow]Would you like to render the final video?[/bold yellow]"):
            console.rule("Rendering Final Video", style="bold blue")
            render_and_concat(saved_file, config["output_dir"], "final_video.mp4")
    else:
        console.rule("Final Result - With Errors (Not Executable)", style="bold green")
        console.print("\n[bold underline]Final Manim Code with errors :[/bold underline]\n")
        console.print(Syntax(current_code, "python", theme="monokai", line_numbers=True))
        console.print("\n[bold underline]Output :[/bold underline]\n")
        console.print(combined_logs)


if __name__ == "__main__":
    main()