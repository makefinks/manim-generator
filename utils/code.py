"""Utility functions for code execution / interaction"""

import base64
import logging
import os
import re
import subprocess
from rich.console import Console

logger = logging.getLogger(__file__)

def save_code_to_file(code: str, filename: str = "video.py") -> str:
    """
    Saves the generated code to a Python file.

    Args:
        code: String containing the Python code to save
        filename: Name of the file to save to (defaults to video.py)

    Returns:
        str: Path to the saved file if successful, empty string if failed
    """
    try:
        # Create directories if they don't exist
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
            
        with open(filename, "w", encoding="utf-8") as f:
            f.write(code)
        return filename
    except Exception as e:
        logger.exception(e)
        return ""
    
def run_manim_multiscene(code: str,  console: Console, output_media_dir: str = "output") -> tuple[bool, list[str], str]:
    """
    Saves the code to a file, extracts scene names, and runs each scene individually with the
    --save_last_frame flag. For each scene, if the last frame is produced successfully, the image
    is read and encoded as a Base64 string.
    
    Args:
        code: String containing the Manim Python code to execute
        console: Rich Console instance for status updates and output
        output_media_dir: Directory to store rendered media files (defaults to "output")
    
    Returns a tuple containing:
      - a boolean success flag (True only if all scenes rendered successfully and files were found),
      - a list of Base64 encoded image strings as data URLs,
      - and a combined log string.
    """
    # Save the code to a temporary file
    filename = save_code_to_file(code, filename=f"{output_media_dir}/video.py")
    
    # Extract scene names from the code
    scene_names = extract_scene_class_names(code)
    
    combined_logs = ""
    overall_success = True

    # Run each scene
    for scene in scene_names:
        with console.status(f"[bold blue]Rendering scene {scene}..."):
            command = ["manim", "-ql", "--save_last_frame", "--media_dir", output_media_dir, filename, scene, ]
            process = subprocess.run(
                command,
                text=True,
                capture_output=True,
                env=os.environ.copy()
            )
            
            # Log the output for each scene
            stdout = process.stdout.replace('\\n\\n', '')
            stderr = process.stderr.replace('\\n\\n', '')
            log_entry = (
                f"Scene: {scene}\n"
                f"STDOUT:\n{stdout}\n"
                f"STDERR:\n{stderr}\n\n"
            )
            combined_logs += log_entry
            
            if process.returncode != 0:
                overall_success = False
                console.print(f"[red]Error running scene {scene}[/red]")

    # After all scenes are run, walk media directory for images
    frames_base64 = []
    base_path = os.path.join(output_media_dir, "images", os.path.splitext(os.path.basename(filename))[0])
    
    if os.path.exists(base_path):
        with console.status("[bold blue]Processing rendered frames..."):
            for root, dirs, files in os.walk(base_path):
                for file in files:
                    if file.endswith(".png"):
                        frame_path = os.path.join(root, file)
                        try:
                            with open(frame_path, "rb") as image_file:
                                image_data = image_file.read()
                            # Encode the image data in Base64 and format as data URL
                            image_base64 = base64.b64encode(image_data).decode('utf-8')
                            data_url = f"data:image/png;base64,{image_base64}"
                            frames_base64.append(data_url)
                            # Delete the image file after encoding
                            os.remove(frame_path)
                        except Exception as e:
                            overall_success = False
                            console.print(f"[red]Error reading/encoding frame {frame_path}: {e}[/red]")
    else:
        overall_success = False
        console.print(f"[red]Media directory not found at {base_path}[/red]")
    
    return overall_success, frames_base64, combined_logs

def extract_scene_class_names(code: str) -> list[str]:
    """
    Extracts all class names that inherit from a Manim Scene.
    Assumes that the classes are defined like: `class ClassName(...Scene):`
    """
    pattern = re.compile(r'class\s+(\w+)\(.*Scene.*\):')
    return pattern.findall(code)


def parse_code_block(text: str) -> str | None:
    """
    Returns the first Python code block from text (None if none found).
    Handles optional 'python' specifier and trims whitespace.
    """

    match = re.search(
        r"```(?:python)?\s*\n(.*?)```",  # Matches optional 'python' and code
        text,
        re.DOTALL,  # Allows . to match newlines
    )
    return match.group(1).strip() if match else text
