from rich.console import Console
from rich.prompt import Confirm
from rich.panel import Panel
from rich.markdown import Markdown

from src.artifacts import ArtifactManager
from src.utils.usage import TokenUsageTracker
from src.utils.progress import ProgressManager
from src.utils.rendering import (
    extract_scene_class_names,
    run_manim_multiscene,
    calculate_scene_success_rate,
)
from src.utils.file import save_code_to_file
from src.console import get_response_with_status, print_code_with_syntax, get_response_with_progress_aware_status
from src.utils.progress import ProgressManager, CycleInfo
from src.utils.prompt import (
    convert_frames_to_message_format,
    format_previous_reviews,
    format_prompt,
)
from src.utils.video import render_and_concat
from src.utils.llm import check_and_register_models
from src.utils.parsing import parse_code_block


class ManimWorkflow:
    """Manages the Manim code generation and review workflow."""

    def __init__(self, config: dict, console: Console, progress_manager=None):
        self.config = config
        self.console = console
        self.progress_manager = progress_manager
        self.usage_tracker = TokenUsageTracker()
        self.artifact_manager = ArtifactManager(config["output_dir"], console)
        self.cycles_completed = 0
        self.execution_count = 0
        self.successful_executions = 0
        self.initial_success = False

        models_to_check = [config["manim_model"], config["review_model"]]
        check_and_register_models(models_to_check, console)

    def generate_initial_code(self, video_data: str) -> tuple[str, list]:
        """Generate the initial Manim code based on video data.

        Args:
            video_data: Description of the video/animation to be generated

        Returns:
            tuple: (generated_code, conversation_history)
        """
        # Update progress if in headless mode
        if self.progress_manager:
            self.progress_manager.update_stage("Initial Code Generation")
            self.progress_manager.update_step("Preparing prompt...")
        
        # Skip verbose output in headless mode
        if not (self.progress_manager and self.progress_manager.is_headless):
            self.console.rule("[bold green]Initial Manim Code Generation", style="green")

        main_messages = [
            {
                "role": "system",
                "content": format_prompt("init_prompt", {"video_data": video_data}),
            }
        ]

        # Update progress before API call
        if self.progress_manager:
            self.progress_manager.update_step("Generating code...")

        response, usage_info, reasoning_content = get_response_with_progress_aware_status(
            self.config["manim_model"],
            main_messages,
            self.config["temperature"],
            self.config["streaming"],
            f"[bold green]Generating initial code \\[{self.config['manim_model']}\\]",
            self.console,
            self.progress_manager,
            reasoning=self.config["reasoning"],
            provider=self.config["provider"],
        )

        self.usage_tracker.add_step(
            "Initial Code Generation", self.config["manim_model"], usage_info
        )
        
        if not (self.progress_manager and self.progress_manager.is_headless):
            self.console.clear()

        # Update progress after completion
        if self.progress_manager:
            self.progress_manager.update_step("Processing generated code...")

        # display reasoning content if available and not streaming
        if reasoning_content and not self.config["streaming"] and not (self.progress_manager and self.progress_manager.is_headless):
            self.console.print(
                Panel(
                    reasoning_content,
                    title="[yellow]Model Reasoning[/yellow]",
                    border_style="yellow",
                )
            )

        code = parse_code_block(response)

        if not self.config["streaming"] and not (self.progress_manager and self.progress_manager.is_headless):
            print_code_with_syntax(code, self.console, "Generated Initial Manim Code")

        prompt_content = format_prompt("init_prompt", {"video_data": video_data})
        self.artifact_manager.save_step_artifacts(
            "initial", code=code, prompt=prompt_content, reasoning=reasoning_content
        )

        # Final progress update
        if self.progress_manager:
            self.progress_manager.update_step("Initial code generated ✓")

        return code, main_messages

    def execute_code(
        self, code: str, step_name: str = "Execution"
    ) -> tuple[bool, list, str]:
        """Execute Manim code and return results.

        Args:
            code: The Manim code to execute
            step_name: Name of the execution step for logging

        Returns:
            tuple: (success, frames, logs)
        """
        # Update progress if in headless mode
        if self.progress_manager:
            self.progress_manager.update_stage(f"Code Execution - {step_name}")
            self.progress_manager.update_step("Starting execution...")
        
        # Skip verbose output in headless mode
        if not (self.progress_manager and self.progress_manager.is_headless):
            self.console.rule(
                f"[bold green]Running Manim Script - {step_name}", style="green"
            )

        success, frames, logs = run_manim_multiscene(
            code,
            self.console,
            self.config["output_dir"],
            step_name.lower().replace(" ", "_"),
            self.artifact_manager,
            self.config.get("frame_extraction_mode", "fixed_count"),
            self.config.get("frame_count", 3),
            self.progress_manager,
        )

        self._display_execution_status(success, frames, code, logs)

        if step_name != "Initial":
            self.execution_count += 1
            if success:
                self.successful_executions += 1

        self.artifact_manager.save_step_artifacts(
            step_name.lower().replace(" ", "_"), code=code, logs=logs
        )

        # Update progress after execution
        if self.progress_manager:
            status = "completed successfully ✓" if success else "completed with errors ⚠"
            self.progress_manager.update_step(f"Execution {status}")

        return success, frames, logs

    def _display_execution_status(
        self, success: bool, frames: list, code: str, logs: str
    ) -> None:
        """Display execution status information."""
        # Skip detailed output in headless mode
        if self.progress_manager and self.progress_manager.is_headless:
            return
            
        status_color = "green" if success else "red"

        scene_names = extract_scene_class_names(code)
        scenes_rendered = f"{int(len(frames) / self.config['frame_count'])} of {len(scene_names) if isinstance(scene_names, list) else '? (Syntax error)'}"

        self.console.print(
            f"[bold {status_color}]Execution Status: {'Success' if success else 'Failed'}[/bold {status_color}]"
        )
        self.console.print(
            f"[bold {status_color}]Scenes Rendered: {scenes_rendered}[/bold {status_color}]"
        )

        if self.config["manim_logs"]:
            log_title = (
                "[green]Execution Logs[/green]"
                if success
                else "[red]Execution Errors[/red]"
            )
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
        combined_logs: str,
        last_frames: list,
        video_data: str,
    ) -> tuple[str, str | None, str]:
        """Perform review cycles and update the Manim code.

        Args:
            current_code: The current Manim code
            combined_logs: Logs from executions
            last_frames: Rendered frames from last execution
            video_data: Original video description

        Returns:
            tuple: (final_code, last_working_code, final_logs)
        """
        working_code = None
        previous_reviews = []

        for cycle in range(self.config["review_cycles"]):
            # Update progress with cycle information
            if self.progress_manager:
                cycle_info = CycleInfo(
                    current_cycle=cycle + 1,
                    total_cycles=self.config["review_cycles"],
                    cycle_type="review"
                )
                self.progress_manager.set_cycle_info(cycle_info)
                self.progress_manager.update_stage(f"Review Cycle {cycle + 1}")
                self.progress_manager.update_step("Generating review...")
            
            # Skip verbose output in headless mode
            if not (self.progress_manager and self.progress_manager.is_headless):
                self.console.rule(f"[bold blue]Review Cycle {cycle + 1}", style="blue")

            review, review_reasoning = self._generate_review(
                current_code, combined_logs, last_frames, previous_reviews, cycle + 1
            )
            previous_reviews.append(review)

            if review_reasoning and not self.config["streaming"] and not (self.progress_manager and self.progress_manager.is_headless):
                self.console.print(
                    Panel(
                        review_reasoning,
                        title="[yellow]Review Model Reasoning[/yellow]",
                        border_style="yellow",
                    )
                )

            if not (self.progress_manager and self.progress_manager.is_headless):
                self.console.print(
                    Panel(
                        Markdown(review),
                        title="[blue]Review Feedback[/blue]",
                        border_style="blue",
                    )
                )

            # Update progress for code revision
            if self.progress_manager:
                self.progress_manager.update_step("Generating code revision...")

            current_code = self._generate_code_revision(
                current_code, review, video_data, cycle + 1, last_frames
            )

            success, last_frames, combined_logs = self.execute_code(
                current_code, f"Revision {cycle + 1}"
            )
            if success:
                working_code = current_code

            self.cycles_completed = cycle + 1

        # Clear cycle info when done
        if self.progress_manager:
            self.progress_manager.set_cycle_info(None)

        return current_code, working_code, combined_logs

    def _generate_review(
        self, code: str, logs: str, frames: list, previous_reviews: list, cycle_num: int
    ) -> tuple[str, str | None]:
        """Generate a review of the current code."""
        frames_formatted = (
            convert_frames_to_message_format(frames)
            if frames and self.config["vision_enabled"]
            else []
        )

        if frames_formatted and not (self.progress_manager and self.progress_manager.is_headless):
            self.console.print(
                f"[blue] Adding {len(frames_formatted)} images to the review"
            )

        # success rate determines review prompt
        scene_names = extract_scene_class_names(code)
        success_rate, scenes_rendered, total_scenes = calculate_scene_success_rate(
            frames,
            scene_names,
            self.config["frame_count"],
            self.config["frame_extraction_mode"],
        )

        # check if we can use visual enhance review prompt
        use_enhanced_prompt = success_rate >= self.config["success_threshold"]
        prompt_name = (
            "review_prompt_enhanced" if use_enhanced_prompt else "review_prompt"
        )

        if use_enhanced_prompt:
            if not (self.progress_manager and self.progress_manager.is_headless):
                self.console.print(
                    f"[green]High success rate ({success_rate:.1f}%) - Using enhanced visual review prompt"
                )
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
            if not (self.progress_manager and self.progress_manager.is_headless):
                self.console.print(
                    f"[yellow]Success rate ({success_rate:.1f}%) - Using standard technical review prompt"
                )
            review_content = format_prompt(
                prompt_name,
                {
                    "previous_reviews": format_previous_reviews(previous_reviews),
                    "video_code": code,
                    "execution_logs": logs,
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

        response, usage_info, reasoning_content = get_response_with_progress_aware_status(
            self.config["review_model"],
            review_message,
            self.config["temperature"],
            self.config["streaming"],
            status=f"[bold blue]Generating {'Enhanced Visual' if use_enhanced_prompt else 'Technical'} Review \\[{self.config['review_model']}\\]",
            console=self.console,
            progress_manager=self.progress_manager,
            reasoning=self.config["reasoning"],
            provider=self.config["provider"],
        )

        self.usage_tracker.add_step(
            f"Review Cycle {cycle_num}", self.config["review_model"], usage_info
        )
        self.artifact_manager.save_step_artifacts(
            f"review_{cycle_num}",
            prompt=review_content,
            review_text=response,
            reasoning=reasoning_content,
        )

        return response, reasoning_content

    def _generate_code_revision(
        self,
        current_code: str,
        review: str,
        video_data: str,
        cycle_num: int,
        frames: list | None = None,
    ) -> str:
        """Generate a revised version of the code based on review feedback."""

        # we also pass the frames to the manim model as context
        frames_formatted = (
            convert_frames_to_message_format(frames)
            if frames and self.config["vision_enabled"]
            else []
        )

        revision_prompt = f"Here is the current code:\n\n```python\n{current_code}\n```\n\nHere is some feedback on your code:\n\n<review>\n{review}\n</review>\n\nPlease implement the suggestions and respond with the whole script. Do not leave anything out."

        # include frames
        if frames_formatted:
            if not (self.progress_manager and self.progress_manager.is_headless):
                self.console.print(
                    f"[green]Adding {len(frames_formatted)} images to code revision"
                )
            user_content = [
                {"type": "text", "text": revision_prompt}
            ] + frames_formatted
        else:
            user_content = revision_prompt

        revision_messages = [
            {
                "role": "system",
                "content": format_prompt("init_prompt", {"video_data": video_data}),
            },
            {"role": "user", "content": user_content},
        ]

        # Skip verbose output in headless mode
        if not (self.progress_manager and self.progress_manager.is_headless):
            self.console.rule(
                f"[bold green]Generating Code Revision {cycle_num}", style="green"
            )

        revised_response, usage_info, reasoning_content = get_response_with_progress_aware_status(
            self.config["manim_model"],
            revision_messages,
            self.config["temperature"],
            self.config["streaming"],
            f"[bold green]Generating code revision \\[{self.config['manim_model']}]",
            self.console,
            self.progress_manager,
            reasoning=self.config["reasoning"],
            provider=self.config["provider"],
        )

        if reasoning_content and not self.config["streaming"] and not (self.progress_manager and self.progress_manager.is_headless):
            self.console.print(
                Panel(
                    reasoning_content,
                    title="[yellow]Model Reasoning[/yellow]",
                    border_style="yellow",
                )
            )
        self.usage_tracker.add_step(
            f"Code Revision {cycle_num}", self.config["manim_model"], usage_info
        )

        revised_code = parse_code_block(revised_response)

        # only display code block if not streaming and not in headless mode
        if not self.config["streaming"] and not (self.progress_manager and self.progress_manager.is_headless):
            print_code_with_syntax(
                revised_code, self.console, f"Revised Code - Cycle {cycle_num}"
            )

        self.artifact_manager.save_step_artifacts(
            f"revision_{cycle_num}",
            code=revised_code,
            prompt=revision_prompt,
            reasoning=reasoning_content,
        )

        return revised_code

    def finalize_output(
        self, working_code: str | None, final_code: str, logs: str
    ) -> None:
        """Handle final output, saving, and rendering."""
        
        # Update progress for finalization
        if self.progress_manager:
            self.progress_manager.update_stage("Finalizing Output")
            self.progress_manager.update_step("Preparing final results...")

        if working_code:
            # In headless mode, automatically save without asking
            if self.progress_manager and self.progress_manager.is_headless:
                if self.progress_manager:
                    self.progress_manager.update_step("Saving final code...")
                
                saved_file = save_code_to_file(
                    working_code, filename=f"{self.config['output_dir']}/video.py"
                )
                
                self.artifact_manager.save_step_artifacts(
                    "final", code=working_code
                )
            else:
                # Normal interactive mode
                if Confirm.ask("View final working code?"):
                    self.console.rule("[bold green]Final Result", style="green")

                    print_code_with_syntax(working_code, self.console, "Final Manim Code")

                    if self.progress_manager:
                        self.progress_manager.update_step("Saving final code...")

                    saved_file = save_code_to_file(
                        working_code, filename=f"{self.config['output_dir']}/video.py"
                    )
                    self.console.print(
                        f"[bold green]Code saved to: {saved_file}[/bold green]"
                    )

                    self.console.rule("[bold blue]Rendering Options", style="blue")
                    if Confirm.ask(
                        "[bold blue]Would you like to render the final video?[/bold blue]"
                    ):
                        if self.progress_manager:
                            self.progress_manager.update_step("Rendering final video...")
                            
                        self.console.rule("[bold blue]Rendering Final Video", style="blue")
                        render_and_concat(
                            saved_file, self.config["output_dir"], "final_video.mp4"
                        )

                        self.artifact_manager.save_step_artifacts(
                            "final", code=working_code
                        )
        else:
            if not (self.progress_manager and self.progress_manager.is_headless):
                if Confirm.ask("View final non-working code?"):
                    self.console.rule(
                        "[bold red]Final Result - With Errors (Not Executable)", style="red"
                    )
                    print_code_with_syntax(
                        final_code, self.console, "Final Manim Code (with errors)"
                    )
                    self.console.print(
                        Panel(logs, title="[red]Execution Errors[/red]", border_style="red")
                    )

        # Final progress update
        if self.progress_manager:
            self.progress_manager.update_step("Finalization complete ✓")
