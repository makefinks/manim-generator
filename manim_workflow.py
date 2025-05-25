from rich.console import Console
from rich.prompt import Confirm
from rich.panel import Panel
from rich.markdown import Markdown

from utils.code import (
    extract_scene_class_names,
    run_manim_multiscene,
    parse_code_block,
    save_code_to_file,
    calculate_scene_success_rate,
)
from utils.console import get_response_with_status, print_code_with_syntax
from utils.text import (
    convert_frames_to_message_format,
    format_previous_reviews,
    format_prompt,
)
from utils.video import render_and_concat
from utils.llm import check_and_register_models


class TokenUsageTracker:
    """Tracks token usage and costs across workflow steps."""
    
    def __init__(self):
        self.token_usage_tracking = {
            "steps": [],
            "total_tokens": 0,
            "total_cost": 0.0
        }
    
    def add_step(self, step_name: str, model: str, usage_info: dict) -> None:
        """Add a step's usage information to tracking."""
        step_info = {
            "step": step_name,
            "model": model,
            **usage_info,
        }
        self.token_usage_tracking["steps"].append(step_info)
        self.token_usage_tracking["total_tokens"] += usage_info.get("total_tokens", 0)
        self.token_usage_tracking["total_cost"] += usage_info.get("cost", 0.0)
    
    def get_tracking_data(self) -> dict:
        """Get the complete tracking data."""
        return self.token_usage_tracking


class ManimWorkflow:
    """Manages the Manim code generation and review workflow."""
    
    def __init__(self, config: dict, console: Console):
        self.config = config
        self.console = console
        self.usage_tracker = TokenUsageTracker()
        
        # Check and register models for cost calculation
        models_to_check = [config["manim_model"], config["review_model"]]
        check_and_register_models(models_to_check, console)
    
    def generate_initial_code(self, video_data: str) -> tuple[str, list]:
        """Generate the initial Manim code based on video data.
        
        Args:
            video_data: Description of the video/animation to be generated
            
        Returns:
            tuple: (generated_code, conversation_history)
        """
        self.console.rule("[bold green]Initial Manim Code Generation", style="green")
        
        main_messages = [{
            "role": "system",
            "content": format_prompt("init_prompt", {"video_data": video_data}),
        }]
        
        response, usage_info, reasoning_content = get_response_with_status(
            self.config["manim_model"],
            main_messages,
            self.config["temperature"],
            self.config["streaming"],
            f"[bold green]Generating initial code \\[{self.config['manim_model']}\\]",
            self.console,
            reasoning=self.config["reasoning"],
            provider=self.config["provider"],
        )
        
        # Track usage
        self.usage_tracker.add_step("Initial Code Generation", self.config["manim_model"], usage_info)
        
        self.console.clear()
        
        # Display reasoning content if available and not streaming
        if reasoning_content and not self.config["streaming"]:
            self.console.print(
                Panel(
                    reasoning_content,
                    title="[yellow]Model Reasoning[/yellow]",
                    border_style="yellow",
                )
            )
        
        code = parse_code_block(response)
        print_code_with_syntax(code, self.console, "Generated Initial Manim Code")
        
        return code, main_messages
    
    def execute_code(self, code: str, step_name: str = "Execution") -> tuple[bool, list, str]:
        """Execute Manim code and return results.
        
        Args:
            code: The Manim code to execute
            step_name: Name of the execution step for logging
            
        Returns:
            tuple: (success, frames, logs)
        """
        self.console.rule(f"[bold green]Running Manim Script - {step_name}", style="green")
        
        success, frames, logs = run_manim_multiscene(
            code, self.console, self.config["output_dir"]
        )
        
        self._display_execution_status(success, frames, code, logs)
        
        return success, frames, logs
    
    def _display_execution_status(self, success: bool, frames: list, code: str, logs: str) -> None:
        """Display execution status information."""
        status_color = "green" if success else "red"
        
        scene_names = extract_scene_class_names(code)
        scenes_rendered = f"{len(frames)} of {len(scene_names) if isinstance(scene_names, list) else '? (Syntax error)'}"
        
        self.console.print(
            f"[bold {status_color}]Execution Status: {'Success' if success else 'Failed'}[/bold {status_color}]"
        )
        self.console.print(
            f"[bold {status_color}]Scenes Rendered: {scenes_rendered}[/bold {status_color}]"
        )
        
        if self.config["manim_logs"] or not success:
            log_title = "[green]Execution Logs[/green]" if success else "[red]Execution Errors[/red]"
            log_style = "green" if success else "red"
            self.console.print(
                Panel(
                    logs,
                    title=log_title,
                    border_style=log_style,
                )
            )
    
    def review_and_update_code(
        self,
        current_code: str,
        main_messages: list,
        combined_logs: str,
        last_frames: list,
        video_data: str,
    ) -> tuple[str, str | None, str]:
        """Perform review cycles and update the Manim code.
        
        Args:
            current_code: The current Manim code
            main_messages: Conversation history
            combined_logs: Logs from executions
            last_frames: Rendered frames from last execution
            video_data: Original video description
            
        Returns:
            tuple: (final_code, last_working_code, final_logs)
        """
        working_code = None
        previous_reviews = []
        
        for cycle in range(self.config["review_cycles"]):
            self.console.rule(f"[bold blue]Review Cycle {cycle + 1}", style="blue")
            
            # Generate review
            review, review_reasoning = self._generate_review(
                current_code, combined_logs, last_frames, previous_reviews, cycle + 1
            )
            
            previous_reviews.append(review)
            
            # Display review feedback
            if review_reasoning and not self.config["streaming"]:
                self.console.print(
                    Panel(
                        review_reasoning,
                        title="[yellow]Review Model Reasoning[/yellow]",
                        border_style="yellow",
                    )
                )
            
            self.console.print(
                Panel(
                    Markdown(review),
                    title="[blue]Review Feedback[/blue]",
                    border_style="blue",
                )
            )
            
            # Generate code revision
            current_code = self._generate_code_revision(
                current_code, review, video_data, cycle + 1
            )
            
            # Execute revised code
            success, last_frames, combined_logs = self.execute_code(
                current_code, f"Revision {cycle + 1}"
            )
            
            if success:
                working_code = current_code
        
        return current_code, working_code, combined_logs
    
    def _generate_review(
        self,
        code: str,
        logs: str,
        frames: list,
        previous_reviews: list,
        cycle_num: int
    ) -> tuple[str, str]:
        """Generate a review of the current code."""
        frames_formatted = (
            convert_frames_to_message_format(frames)
            if frames and self.config["vision_enabled"]
            else []
        )
        
        self.console.print(f"[blue] Adding {len(frames_formatted)} images to the review")
        
        # Calculate scene success rate to determine review prompt type
        scene_names = extract_scene_class_names(code)
        success_rate, scenes_rendered, total_scenes = calculate_scene_success_rate(frames, scene_names)
        
        # Use enhanced review prompt if success rate >= threshold
        use_enhanced_prompt = success_rate >= self.config["success_threshold"]
        prompt_name = "review_prompt_enhanced" if use_enhanced_prompt else "review_prompt"
        
        if use_enhanced_prompt:
            self.console.print(f"[green]High success rate ({success_rate:.1f}%) - Using enhanced visual review prompt")
            review_content = format_prompt(
                prompt_name,
                {
                    "previous_reviews": format_previous_reviews(previous_reviews),
                    "video_code": code,
                    "execution_logs": logs,
                    "success_rate": success_rate,
                    "scenes_rendered": scenes_rendered,
                    "total_scenes": total_scenes,
                },
            )
        else:
            self.console.print(f"[yellow]Success rate ({success_rate:.1f}%) - Using standard technical review prompt")
            review_content = format_prompt(
                prompt_name,
                {
                    "previous_reviews": format_previous_reviews(previous_reviews),
                    "video_code": code,
                    "execution_logs": logs,
                },
            )
        
        review_message = [{
            "role": "user",
            "content": [
                {"type": "text", "text": review_content},
            ] + frames_formatted,
        }]
        
        response, usage_info, reasoning_content = get_response_with_status(
            self.config["review_model"],
            review_message,
            self.config["temperature"],
            self.config["streaming"],
            status=f"[bold blue]Generating {'Enhanced Visual' if use_enhanced_prompt else 'Technical'} Review \\[{self.config['review_model']}\\]",
            console=self.console,
            reasoning=self.config["reasoning"],
            provider=self.config["provider"],
        )
        
        # Track review usage
        self.usage_tracker.add_step(f"Review Cycle {cycle_num}", self.config["review_model"], usage_info)
        
        return response, reasoning_content
    
    def _generate_code_revision(
        self, current_code: str, review: str, video_data: str, cycle_num: int
    ) -> str:
        """Generate a revised version of the code based on review feedback."""
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
        
        self.console.rule(f"[bold green]Generating Code Revision {cycle_num}", style="green")
        
        revised_response, usage_info, reasoning_content = get_response_with_status(
            self.config["manim_model"],
            revision_messages,
            self.config["temperature"],
            self.config["streaming"],
            f"[bold green]Generating code revision \\[{self.config['manim_model']}]",
            self.console,
            reasoning=self.config["reasoning"],
            provider=self.config["provider"],
        )
        
        # Display reasoning content if available and not streaming
        if reasoning_content and not self.config["streaming"]:
            self.console.print(
                Panel(
                    reasoning_content,
                    title="[yellow]Model Reasoning[/yellow]",
                    border_style="yellow",
                )
            )
        
        # Track revision usage
        self.usage_tracker.add_step(f"Code Revision {cycle_num}", self.config["manim_model"], usage_info)
        
        revised_code = parse_code_block(revised_response)
        print_code_with_syntax(
            revised_code, self.console, f"Revised Code - Cycle {cycle_num}"
        )
        
        return revised_code
    
    def finalize_output(self, working_code: str | None, final_code: str, logs: str) -> None:
        """Handle final output, saving, and rendering."""
        if working_code:
            self.console.rule("[bold green]Final Result", style="green")
            print_code_with_syntax(working_code, self.console, "Final Manim Code")
            
            saved_file = save_code_to_file(
                working_code, filename=f"{self.config['output_dir']}/video.py"
            )
            self.console.print(f"[bold green]Code saved to: {saved_file}[/bold green]")
            
            self.console.rule("[bold blue]Rendering Options", style="blue")
            if Confirm.ask("[bold blue]Would you like to render the final video?[/bold blue]"):
                self.console.rule("[bold blue]Rendering Final Video", style="blue")
                render_and_concat(saved_file, self.config["output_dir"], "final_video.mp4")
        else:
            self.console.rule(
                "[bold red]Final Result - With Errors (Not Executable)", style="red"
            )
            print_code_with_syntax(final_code, self.console, "Final Manim Code (with errors)")
            self.console.print(
                Panel(
                    logs, title="[red]Execution Errors[/red]", border_style="red"
                )
            )
