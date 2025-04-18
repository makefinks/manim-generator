from datetime import datetime
import argparse
import os
from rich.console import Console
from rich.syntax import Syntax
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
from utils.text import convert_frames_to_message_format, format_previous_reviews, format_prompt
from utils.video import render_and_concat
from utils.file import load_video_data


# Default configuration
DEFAULT_CONFIG = {
    "manim_model": "gemini/gemini-2.5-pro-exp-03-25",
    "review_model": "gemini/gemini-2.5-pro-exp-03-25",
    "review_cycles": 3,
    "output_dir": f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    "manim_logs": False,
    "streaming": False,
    "temperature": 0.4
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
    parser.add_argument("--temperature", type=float, default=DEFAULT_CONFIG["temperature"], help="Temperature for the LLM Model")

    args = parser.parse_args()
    
    # Build config from arguments
    config = {
        "manim_model": args.manim_model,
        "review_model": args.review_model,
        "review_cycles": args.review_cycles,
        "output_dir": args.output_dir,
        "manim_logs": args.manim_logs,
        "streaming": args.streaming,
        "temperature": args.temperature
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
    main_messages = [{
        "role": "system",
        "content": format_prompt("init_prompt", {"video_data": video_data}),
    }]
    response = get_response_with_status(config["manim_model"], main_messages, config["temperature"], config["streaming"], f"[bold green]Generating initial code \\[{config['manim_model']}\]", console)
    console.clear()
    code = parse_code_block(response)
    print_code_with_syntax(code, console, "Generated Initial Manim Code")
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
    
    # Check if the model supports vision/images
    vision_enabled = supports_vision(model=config["review_model"])
    console.print(f"[bold {'green' if vision_enabled else 'yellow'}]Vision support: {'Enabled' if vision_enabled else 'Disabled'}[/bold {'green' if vision_enabled else 'yellow'}]")

    for cycle in range(config["review_cycles"]):
        console.rule(f"[bold blue]Review Cycle {cycle + 1}", style="blue")
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
        review = get_response_with_status(config["review_model"], review_message, config["temperature"], config["streaming"], status=f"[bold blue]Generating Review \\[{config['review_model']}\\]", console=console)
        previous_reviews.append(review)
        console.print(Panel(Markdown(review), title="[blue]Review Feedback[/blue]", border_style="blue"))

        # Append feedback for code revision
        main_messages.append({
            "role": "user",
            "content": f"Here is some feedback on your code. \n\n<review{review}</review>\n\nPlease implement the suggestions and respond with the whole script."
        })

        console.rule(f"[bold green]Generating Code Revision {cycle + 1}", style="green")
        revised_response = get_response_with_status(config["manim_model"], main_messages, config["temperature"], config["streaming"], f"[bold green]Generating code revision \\[{config['manim_model']}]", console)
        main_messages.append({"role": "assistant", "content": revised_response})
        current_code = parse_code_block(revised_response)
        print_code_with_syntax(current_code, console, f"Revised Code - Cycle {cycle + 1}")
        
        console.rule(f"[bold green]Running Manim Script - Revision {cycle + 1}", style="green")
        success, last_frames, combined_logs = run_manim_multiscene(current_code, console, config["output_dir"])
        
        status_color = "green" if success else "red"
        scenes_rendered = f"{len(last_frames)} of {len(extract_scene_class_names(current_code))}"
        console.print(f"[bold {status_color}]Execution Status: {'Success' if success else 'Failed'}[/bold {status_color}]")
        console.print(f"[bold {status_color}]Scenes Rendered: {scenes_rendered}[/bold {status_color}]")
        
        # update working_code if this iteration was successful
        if success:
            working_code = current_code
        
        if config["manim_logs"]:
            console.print(Panel(combined_logs, title="[green]Execution Logs[/green]", border_style="green"))

    return current_code, working_code, combined_logs


def main():
    config, video_data_arg, video_data_file = parse_arguments()
    
    # Get video data from argument or file
    if video_data_arg:
        video_data = video_data_arg
    else:
        video_data = load_video_data(video_data_file, console)
    
    current_code, main_messages = generate_initial_code(video_data, config)
    
    console.rule("[bold green]Running Initial Manim Script", style="green")
    success, last_frames, combined_logs = run_manim_multiscene(current_code, console, config["output_dir"])
    
    status_color = "green" if success else "red"
    scenes_rendered = f"{len(last_frames)} of {len(extract_scene_class_names(current_code))}"
    console.print(f"[bold {status_color}]Execution Status: {'Success' if success else 'Failed'}[/bold {status_color}]")
    console.print(f"[bold {status_color}]Scenes Rendered: {scenes_rendered}[/bold {status_color}]")
    
    if config["manim_logs"]:
        console.print(Panel(combined_logs, title="[green]Execution Logs[/green]", border_style="green"))
    
    working_code = current_code if success else None

    if not working_code:
        console.print(Panel("[bold red]Initial code failed to execute properly.[/bold red]", border_style="red"))
    
    # Execute review cycles
    current_code, new_working_code, combined_logs = review_and_update_code(current_code, main_messages, combined_logs, last_frames, config)
    working_code = new_working_code if new_working_code else working_code

    # Final output
    if working_code:
        console.rule("[bold green]Final Result", style="green")
        print_code_with_syntax(working_code, console, "Final Manim Code")
        saved_file = save_code_to_file(working_code, filename=f"{config['output_dir']}/video.py")
        console.print(f"[bold green]Code saved to: {saved_file}[/bold green]")

        console.rule("[bold blue]Rendering Options", style="blue")
        if Confirm.ask("[bold blue]Would you like to render the final video?[/bold blue]"):
            console.rule("[bold blue]Rendering Final Video", style="blue")
            render_and_concat(saved_file, config["output_dir"], "final_video.mp4")
    else:
        console.rule("[bold red]Final Result - With Errors (Not Executable)", style="red")
        print_code_with_syntax(current_code, console, "Final Manim Code (with errors)")
        console.print(Panel(combined_logs, title="[red]Execution Errors[/red]", border_style="red"))


if __name__ == "__main__":
    main()
