"""Utility functions for video generation and manipulation with Manim."""

import os
import subprocess
import logging
from utils.code import extract_scene_class_names

logger = logging.getLogger(__name__)

def render_and_concat(script_file: str, output_media_dir: str, final_output: str) -> None:
    """
    Runs a Manim script as a subprocess, then concatenates the rendered scene videos
    (in the order they appear in the script) into one final video using ffmpeg.

    Parameters:
      script_file (str): Path to the Manim Python script (e.g. "video.py")
      output_media_dir (str): The media directory specified to Manim (e.g. "output")
      final_output (str): The filename for the concatenated final video (e.g. "final_video.mp4")
    """

    # Run Manim as a subprocess with real-time output
    manim_command = [
        "manim", "-qh", script_file, "--write_all", "--media_dir", output_media_dir
    ]
    process = subprocess.Popen(
        manim_command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
        bufsize=1,
        universal_newlines=True
    )
    
    # Print output in real-time
    while True:
        output = process.stdout.readline() 
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
            logger.info(output.strip())
    
    if process.returncode != 0:
        logger.error("Error during Manim rendering")
        return
    else:
        logger.info("Manim rendering completed successfully.")
    
    # Extract scene names in order from the Python script using regex.
    with open(script_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    scene_names = extract_scene_class_names(content)
    
    logger.info("Found scene names in order: %s", scene_names)
    
    # Build the path to the rendered videos.
    # Manim will save videos in: output/videos/<script_basename>/<quality_folder>/
    script_basename = os.path.splitext(os.path.basename(script_file))[0]

    # The quality folder is "1080p60" since we used -pqh
    quality_folder = "1080p60"
    videos_dir = os.path.join(output_media_dir, "videos", script_basename, quality_folder)
    if not os.path.exists(videos_dir):
        logger.error("Rendered videos folder not found: %s", videos_dir)
        return
    
    # Create a temporary file for ffmpeg's concat list in the output directory
    concat_list_path = os.path.join(output_media_dir, "ffmpeg_concat_list.txt")
    with open(concat_list_path, "w", encoding="utf-8") as file_list:
        for scene in scene_names:
            video_path = os.path.join(videos_dir, f"{scene}.mp4")
            if not os.path.exists(video_path):
                logger.warning("Expected video file for scene '%s' not found at %s", scene, video_path)
            abs_path = os.path.abspath(video_path)
            # ffmpeg requires lines in the form: file '/path/to/file'
            file_list.write(f"file '{abs_path}'\n")
    
    # Save final output to the output directory
    final_output_path = os.path.join(output_media_dir, final_output)
    
    # Call ffmpeg to concatenate the videos with real-time output
    ffmpeg_command = [
        "ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_list_path,
        "-c", "copy", final_output_path
    ]
    logger.info("Concatenating videos with ffmpeg: %s", " ".join(ffmpeg_command))
    
    ffmpeg_proc = subprocess.Popen(
        ffmpeg_command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True
    )
    
    # Print ffmpeg output in real-time
    while True:
        output = ffmpeg_proc.stdout.readline()
        if output == '' and ffmpeg_proc.poll() is not None:
            break
        if output:
            print(output.strip())
            logger.info(output.strip())
    
    if ffmpeg_proc.returncode != 0:
        logger.error("Error during ffmpeg concatenation")
    else:
        logger.info("Final concatenated video created at: %s", final_output_path)
    
    # Clean up temporary concat file.
    os.remove(concat_list_path)

    # Play the final video
    play_command = []
    if os.name == 'nt':  # Windows
        # Use the full path and shell=True for Windows
        final_output_path = os.path.abspath(final_output_path)
        try:
            subprocess.run(['cmd', '/c', 'start', '', final_output_path], shell=True)
            logger.info("Playing video with default media player")
        except subprocess.CalledProcessError as e:
            logger.error("Failed to play video: %s", str(e))
    elif os.name == 'posix':  # Linux/Mac
        # For Linux, use any available xdg-open from PATH instead of hardcoded path
        if os.uname().sysname == 'Linux':  # More reliable Linux detection
            # Create absolute path for the video file
            abs_path = os.path.abspath(final_output_path)
            try:
                # Use shell=True for better path and environment handling on Linux
                subprocess.run(['xdg-open', abs_path], check=True, env=os.environ.copy())
                logger.info("Playing video with xdg-open")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                logger.error("Failed to play video with xdg-open: %s", str(e))
                try:
                    # Fallback to a common video player if available
                    for player in ['vlc', 'mpv', 'ffplay', 'mplayer']:
                        try:
                            subprocess.run(['which', player], check=True, stdout=subprocess.PIPE)
                            subprocess.run([player, abs_path], check=False)
                            logger.info(f"Playing video with {player}")
                            break
                        except subprocess.CalledProcessError:
                            continue
                except Exception as e:
                    logger.error("Failed to play video with fallback players: %s", str(e))
        else:  # Mac
            play_command = ['open', final_output_path]
            try:
                subprocess.run(play_command, check=True)
                logger.info("Playing video with default media player")
            except subprocess.CalledProcessError as e:
                logger.error("Failed to play video: %s", str(e))
    else:
        logger.error("Could not determine appropriate video player command for this system")
