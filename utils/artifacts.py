"""Utility functions for preserving workflow artifacts and debugging information."""
import os
import shutil
from typing import List, Optional
from rich.console import Console


class ArtifactManager:
    """Manages preservation of workflow artifacts for debugging and analysis."""
    
    def __init__(self, output_dir: str, console: Console):
        self.output_dir = output_dir
        self.console = console
        self.steps_dir = os.path.join(output_dir, "steps")
        os.makedirs(self.steps_dir, exist_ok=True)
    
    def save_step_artifacts(
        self, 
        step_name: str, 
        code: str | None = None, 
        prompt: str | None = None, 
        logs: str | None = None, 
        frames_dir: str | None = None,
        review_text: str | None  = None,
        reasoning: str | None  = None
    ) -> str:
        """Save all artifacts for a workflow step."""
        step_dir = os.path.join(self.steps_dir, step_name)
        os.makedirs(step_dir, exist_ok=True)
        
        # Save code
        if code:
            with open(os.path.join(step_dir, "code.py"), "w", encoding="utf-8") as f:
                f.write(code)
        
        # Save prompt
        if prompt:
            with open(os.path.join(step_dir, "prompt.txt"), "w", encoding="utf-8") as f:
                f.write(prompt)
        
        # Save logs
        if logs:
            with open(os.path.join(step_dir, "logs.txt"), "w", encoding="utf-8") as f:
                f.write(logs)
        
        # Save review text
        if review_text:
            with open(os.path.join(step_dir, "review.md"), "w", encoding="utf-8") as f:
                f.write(review_text)
        
        # Save reasoning
        if reasoning:
            with open(os.path.join(step_dir, "reasoning.txt"), "w", encoding="utf-8") as f:
                f.write(reasoning)
        
        # Copy frames if they exist
        if frames_dir and os.path.exists(frames_dir):
            frames_dest = os.path.join(step_dir, "frames")
            if os.path.exists(frames_dest):
                shutil.rmtree(frames_dest)
            shutil.copytree(frames_dir, frames_dest)
        
        return step_dir
    
    def get_step_frames_path(self, step_name: str) -> str:
        """Get the path where frames should be saved for a step."""
        step_dir = os.path.join(self.steps_dir, step_name)
        frames_dir = os.path.join(step_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        return frames_dir
