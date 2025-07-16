"""Utility functions for code execution / interaction"""
import base64
import logging
import os
import re
import shutil
import subprocess
from rich.console import Console
import ast
import cv2
import numpy as np

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
    
def run_manim_multiscene(code: str,  console: Console, output_media_dir: str = "output", step_name: str = None, artifact_manager = None) -> tuple[bool, list[str], str]:
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

    # if we encounter an exception parsing the scene class names (e.g syntax error) we return the exception message
    if isinstance(scene_names, Exception):
        error_msg = f"Code parsing failed: {str(scene_names)}\n\nGenerated code has syntax errors and cannot be executed."
        console.print(f"[red]Code parsing error: {str(scene_names)}[/red]")
        return False, [], error_msg

    combined_logs = ""
    overall_success = True

    # Run each scene
    for scene in scene_names:
        with console.status(f"[bold blue]Rendering scene {scene}..."):
            command = ["manim", "-ql", "--media_dir", output_media_dir, filename, scene, ]
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

    # After all scenes are run, extract best frames from scene videos
    frames_base64 = []
    video_base_path = os.path.join(output_media_dir, "videos", "video", "480p15")
    
    if os.path.exists(video_base_path):
        with console.status("[bold blue]Extracting best frames from scene videos..."):
            for scene in scene_names:
                scene_video_path = os.path.join(video_base_path, f"{scene}.mp4")
                if os.path.exists(scene_video_path):
                    try:
                        # Extract the frame with highest pixel density
                        best_frame = extract_frame_with_highest_density(scene_video_path)
                        if best_frame is not None:
                            # Encode the frame as Base64
                            _, buffer = cv2.imencode('.png', best_frame)
                            image_base64 = base64.b64encode(buffer).decode('utf-8')
                            data_url = f"data:image/png;base64,{image_base64}"
                            frames_base64.append(data_url)
                    except Exception as e:
                        overall_success = False
                        console.print(f"[red]Error extracting frame from {scene_video_path}: {e}[/red]")
                else:
                    overall_success = False
                    console.print(f"[red]Video file not found for scene {scene} at {scene_video_path}[/red]")
        
        # Save extracted frames to step directory if step_name and artifact_manager are provided
        if step_name and artifact_manager and frames_base64:
            step_frames_dir = artifact_manager.get_step_frames_path(step_name)
            os.makedirs(step_frames_dir, exist_ok=True)
            
            # Save each extracted frame
            for i, data_url in enumerate(frames_base64):
                # Extract base64 data from data URL
                base64_data = data_url.split(',')[1]
                image_data = base64.b64decode(base64_data)
                
                frame_filename = f"best_frame_{i+1}.png"
                frame_path = os.path.join(step_frames_dir, frame_filename)
                
                with open(frame_path, 'wb') as f:
                    f.write(image_data)
        
        # Clean up video files after extracting frames to prevent old videos 
        # from previous iterations affecting scene counting
        with console.status("[bold blue]Cleaning up video files..."):
            for scene in scene_names:
                scene_video_path = os.path.join(video_base_path, f"{scene}.mp4")
                if os.path.exists(scene_video_path):
                    try:
                        os.remove(scene_video_path)
                    except Exception as e:
                        console.print(f"[yellow]Warning: Could not delete {scene_video_path}: {e}[/yellow]")
    else:
        overall_success = False
        console.print(f"[red]Video directory not found at {video_base_path}[/red]")
    
    return overall_success, frames_base64, combined_logs


def extract_scene_class_names(code: str) -> list[str] | Exception:
    try:
        tree = ast.parse(code)
    except Exception as e:
        # Return the syntax error with more context
        return SyntaxError(f"Syntax error in code: {str(e)}")
        
    scene_names: list[str] = []
    
    try:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    # gets scenes that inherit from 'Scene'
                    base_id = (
                        base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
                    )
                    if base_id.endswith("Scene"):
                        scene_names.append(node.name)
                        break
    except Exception as e:
        return e
    return scene_names

def calculate_scene_success_rate(frames: list, scene_names: list[str] | Exception) -> tuple[float, int, int]:
    """
    Calculate the success rate of scene rendering.
    
    Args:
        frames: List of rendered frame data
        scene_names: List of scene class names or Exception if parsing failed
        
    Returns:
        tuple: (success_rate, scenes_rendered, total_scenes)
    """
    if isinstance(scene_names, Exception):
        return 0.0, 0, 0
    
    total_scenes = len(scene_names)
    scenes_rendered = len(frames)
    
    if total_scenes == 0:
        return 0.0, 0, 0
    
    success_rate = (scenes_rendered / total_scenes) * 100
    return success_rate, scenes_rendered, total_scenes

def extract_frame_with_highest_density(video_path: str, max_frames: int = 30) -> np.ndarray | None:
    """
    Extract the frame with the highest non-black pixel density from a video.
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to sample for performance
    
    Returns:
        numpy.ndarray: The frame with highest pixel density, or None if error
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return None
        
        # Sample frames evenly throughout the video
        frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
        
        best_frame = None
        best_density = 0
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Calculate non-black pixel density
            # Consider a pixel non-black if any channel > threshold
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            non_black_pixels = np.sum(gray > 30)  # Threshold for "black"
            total_pixels = gray.shape[0] * gray.shape[1]
            density = non_black_pixels / total_pixels
            
            if density > best_density:
                best_density = density
                best_frame = frame.copy()
        
        cap.release()
        return best_frame
        
    except Exception as e:
        logger.exception(f"Error extracting frame from {video_path}: {e}")
        return None


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



