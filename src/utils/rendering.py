"""Utility functions for code execution / interaction.

This module executes generated Manim code scene-by-scene and extracts a
representative frame from each rendered scene video for potential use with
vision-capable review models.
"""

import base64
import logging
import os
import re
import subprocess
from rich.console import Console
import cv2
import numpy as np

from src.utils.file import save_code_to_file
from src.utils.parsing import extract_scene_class_names

logger = logging.getLogger(__name__)


def run_manim_multiscene(
    code: str,
    console: Console,
    output_media_dir: str = "output",
    step_name: str | None = None,
    artifact_manager=None,
    frame_extraction_mode: str = "fixed_count",
    frame_count: int = 3,
    progress_manager=None,
) -> tuple[bool, list[str], str]:
    """
    Saves the code to a file, extracts scene names, and runs each scene individually.
    After rendering, extracts representative frames from each scene's video using
    the specified extraction mode and encodes them as Base64 data URLs for use
    with vision-capable models.

    Args:
        code: String containing the Manim Python code to execute
        console: Rich Console instance for status updates and output
        output_media_dir: Directory to store rendered media files (defaults to "output")
        step_name: Optional step name for artifact management
        artifact_manager: Optional artifact manager instance
        frame_extraction_mode: "highest_density" for single best frame, "fixed_count" for multiple frames
        frame_count: Number of frames to extract in fixed_count mode
        progress_manager: Optional progress manager for headless mode

    Returns a tuple containing:
      - a boolean success flag (True only if all scenes rendered successfully and files were found),
      - a list of Base64 encoded image strings as data URLs,
      - and a combined log string.
    """
    # save code to temp file
    filename = save_code_to_file(code, filename=f"{output_media_dir}/video.py")

    # extract scene names
    scene_names = extract_scene_class_names(code)

    # catch syntaxerrors
    if isinstance(scene_names, Exception):
        error_msg = f"Code parsing failed: {str(scene_names)}\n\nGenerated code has syntax errors and cannot be executed."
        if not (progress_manager and progress_manager.is_headless):
            console.print(f"[red]Code parsing error: {str(scene_names)}[/red]")
        elif progress_manager:
            progress_manager.show_error("Parsing Error", step_name or "Code Execution", str(scene_names))
        return False, [], error_msg

    combined_logs = ""
    overall_success = True

    # Run each scene
    for idx, scene in enumerate(scene_names, 1):
        # Update progress in headless mode
        if progress_manager and progress_manager.is_headless:
            progress_manager.update_rendering(idx, len(scene_names))
        
        # Update progress in headless mode
        if progress_manager and progress_manager.is_headless:
            progress_manager.update_rendering(idx, len(scene_names))
        
        # Execute the rendering command
        command = [
            "manim",
            "-ql",  # low quality for speed; produces 480p15 folder
            "--media_dir",
            output_media_dir,
            filename,
            scene,
        ]
        
        if progress_manager and progress_manager.is_headless:
            # In headless mode, run without Rich status display
            process = subprocess.run(
                command, text=True, capture_output=True, env=os.environ.copy()
            )
        else:
            # Standard mode with Rich status display
            with console.status(f"[bold blue]Rendering scene {scene}..."):
                process = subprocess.run(
                    command, text=True, capture_output=True, env=os.environ.copy()
                )

            # get output for each scene
            stdout = process.stdout
            stderr = process.stderr
            log_entry = (
                f"<{scene}>\n"
                f"\t<STDOUT>\n"
                f"\t\t{stdout}\n"
                f"\t</STDOUT>\n"
                f"\t<STDERR>\n"
                f"\t\t{stderr}\n"
                f"\t</STDERR>\n"
                f"</{scene}>\n\n"
            )
            combined_logs += log_entry

        if process.returncode != 0:
            overall_success = False
            if not (progress_manager and progress_manager.is_headless):
                console.print(
                    f"[red]Rendering scene {scene} failed with exit code {process.returncode}[/red]"
                )
            elif progress_manager:
                progress_manager.show_error(
                    "Rendering Failed", 
                    f"Scene {scene}", 
                    f"Exit code {process.returncode}"
                )

    # Determine videos directory for the rendered files
    # According to Manim docs, structure: <media_dir>/videos/<script_basename>/<quality_folder>/<Scene>.mp4
    script_basename = os.path.splitext(os.path.basename(filename))[0]
    quality_folder = "480p15"  # matches -ql argument

    video_base_path = os.path.join(
        output_media_dir, "videos", script_basename, quality_folder
    )

    # List of tuples: (scene_name, data_url)
    frames: list[tuple[str, str]] = []

    if os.path.exists(video_base_path):
        for scene in scene_names:
            scene_video_path = os.path.join(video_base_path, f"{scene}.mp4")
            if os.path.exists(scene_video_path):
                try:
                    extracted_frames = extract_frames_from_video(
                        scene_video_path, frame_extraction_mode, frame_count
                    )
                    if extracted_frames:
                        # encode all frames to base64
                        for idx, frame in enumerate(extracted_frames):
                            success, buffer = cv2.imencode(".png", frame)
                            if success:
                                image_base64 = base64.b64encode(
                                    buffer.tobytes()
                                ).decode("utf-8")
                                data_url = f"data:image/png;base64,{image_base64}"
                                frame_name = (
                                    f"{scene}_{idx + 1}"
                                    if len(extracted_frames) > 1
                                    else scene
                                )
                                frames.append((frame_name, data_url))
                            else:
                                overall_success = False
                                if not (progress_manager and progress_manager.is_headless):
                                    console.print(
                                        f"[yellow]Failed to encode frame {idx + 1} for {scene_video_path}[/yellow]"
                                    )
                    else:
                        overall_success = False
                        if not (progress_manager and progress_manager.is_headless):
                            console.print(
                                f"[yellow]No suitable frames extracted from {scene_video_path}[/yellow]"
                            )
                except Exception as e:
                    overall_success = False
                    if not (progress_manager and progress_manager.is_headless):
                        console.print(
                            f"[red]Error extracting frame from {scene_video_path}: {e}[/red]"
                        )
            else:
                overall_success = False
                if not (progress_manager and progress_manager.is_headless):
                    console.print(
                        f"[red]Video file not found for scene {scene} at {scene_video_path}[/red]"
                    )

        # save artifacts (e.g extracted frames) using scene names
        if step_name and artifact_manager and frames:
            step_frames_dir = artifact_manager.get_step_frames_path(step_name)
            os.makedirs(step_frames_dir, exist_ok=True)

            for idx, (scene_name, data_url) in enumerate(frames, start=1):
                base64_data = data_url.split(",")[1]
                image_data = base64.b64decode(base64_data)
                safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", scene_name)
                frame_filename = f"{idx:02d}_{safe_name}.png"
                frame_path = os.path.join(step_frames_dir, frame_filename)

                with open(frame_path, "wb") as f:
                    f.write(image_data)

        # Clean up video files after extracting frames to prevent old videos
        # from previous iterations affecting scene counting
        if progress_manager and progress_manager.is_headless:
            # In headless mode, no status display
            for scene in scene_names:
                scene_video_path = os.path.join(video_base_path, f"{scene}.mp4")
                if os.path.exists(scene_video_path):
                    try:
                        os.remove(scene_video_path)
                    except Exception as e:
                        if not progress_manager.is_headless:
                            console.print(
                                f"[yellow]Warning: Could not delete {scene_video_path}: {e}[/yellow]"
                            )
        else:
            with console.status("[bold blue]Cleaning up video files..."):
                for scene in scene_names:
                    scene_video_path = os.path.join(video_base_path, f"{scene}.mp4")
                    if os.path.exists(scene_video_path):
                        try:
                            os.remove(scene_video_path)
                        except Exception as e:
                            console.print(
                                f"[yellow]Warning: Could not delete {scene_video_path}: {e}[/yellow]"
                            )
    else:
        overall_success = False
        if not (progress_manager and progress_manager.is_headless):
            console.print(f"[red]Video directory not found at {video_base_path}[/red]")

    return overall_success, [data_url for _, data_url in frames], combined_logs


def calculate_scene_success_rate(
    frames: list,
    scene_names: list[str] | Exception,
    frames_per_scene: int,
    extraction_mode: str,
) -> tuple[float, float, int]:
    """
    Calculate the success rate of scene rendering.

    Args:
        frames: List of rendered frame data
        scene_names: List of scene class names or Exception if parsing failed
        frames_per_scene: Expected frames per scene when in fixed-count mode
        extraction_mode: Selected frame extraction strategy

    Returns:
        tuple: (success_rate, scenes_rendered, total_scenes)
    """
    if isinstance(scene_names, Exception):
        return 0.0, 0, 0

    total_scenes = len(scene_names)

    if total_scenes == 0:
        return 0.0, 0, 0

    if extraction_mode == "fixed_count" and frames_per_scene > 0:
        scenes_rendered = len(frames) / frames_per_scene
    else:
        # Highest-density (and other single-frame modes) return one frame per scene
        scenes_rendered = len(frames)

    scenes_rendered = min(scenes_rendered, total_scenes)

    success_rate = (scenes_rendered / total_scenes) * 100
    return success_rate, scenes_rendered, total_scenes


def extract_frames_from_video(
    video_path: str,
    mode: str = "fixed_count",
    frame_count: int = 3,
    max_frames: int = 30,
) -> list[np.ndarray] | None:
    """
    Extract frames from a video using different strategies.

    Args:
        video_path: Path to the video file
        mode: "highest_density" for single best frame, "fixed_count" for multiple frames
        frame_count: Number of frames to extract in fixed_count mode
        max_frames: Maximum number of frames to sample for performance in highest_density mode

    Returns:
        list of numpy.ndarray: The extracted frames, or None if error
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return None

        if mode == "highest_density":
            # evenly sample frames
            frame_indices = np.linspace(
                0, total_frames - 1, min(max_frames, total_frames), dtype=int
            )

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
            return [best_frame] if best_frame is not None else None

        elif mode == "fixed_count":
            # extract fixed count
            frame_indices = np.linspace(
                0, total_frames - 1, min(frame_count, total_frames), dtype=int
            )

            extracted_frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if ret:
                    extracted_frames.append(frame.copy())

            cap.release()
            return extracted_frames if extracted_frames else None

        else:
            cap.release()
            return None

    except Exception as e:
        logger.exception(f"Error extracting frames from {video_path}: {e}")
        return None
