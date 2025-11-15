import argparse
import os
from datetime import datetime

from litellm.utils import supports_vision
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

# Default configuration
DEFAULT_CONFIG = {
    "manim_model": "openrouter/x-ai/grok-code-fast-1",
    "review_model": "openrouter/x-ai/grok-code-fast-1",
    "review_cycles": 4,
    "manim_logs": False,
    "streaming": False,
    "temperature": 0.4,
    "success_threshold": 100,
    "output_dir": None,
    "frame_extraction_mode": "highest_density",
    "frame_count": 3,
    "scene_timeout": 120,
}


class Config:
    """Configuration manager for manim generator."""

    def __init__(self):
        self.console = Console()

    def parse_arguments(self) -> tuple[dict, str | None, str]:
        """Parse command line arguments and return the configuration."""
        parser = self._create_parser()
        args = parser.parse_args()

        self._validate_reasoning_arguments(args)
        config = self._build_config(args)

        # Create output directory if it doesn't exist
        os.makedirs(config["output_dir"], exist_ok=True)

        return config, args.video_data, args.video_data_file

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create and configure the argument parser."""
        parser = argparse.ArgumentParser(description="Generate Manim animations using AI")

        # Video data input
        parser.add_argument("--video-data", type=str, help="Description of the video to generate")
        parser.add_argument(
            "--video-data-file",
            type=str,
            default="video_data.txt",
            help="Path to file containing video description",
        )

        # Model configuration
        parser.add_argument(
            "--manim-model",
            type=str,
            default=DEFAULT_CONFIG["manim_model"],
            help="Model to use for generating Manim code",
        )
        parser.add_argument(
            "--review-model",
            type=str,
            default=DEFAULT_CONFIG["review_model"],
            help="Model to use for reviewing code",
        )

        # Process configuration
        parser.add_argument(
            "--review-cycles",
            type=int,
            default=DEFAULT_CONFIG["review_cycles"],
            help="Number of review cycles to perform",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default=None,
            help=("Directory to save outputs (overrides the auto-generated folder name)"),
        )
        parser.add_argument(
            "--manim-logs",
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
            "--no-temperature",
            action="store_true",
            default=False,
            help="Skip temperature parameter in LLM requests.",
        )

        parser.add_argument(
            "--force-vision",
            action="store_true",
            default=False,
            help="Adds images to the review process, regardless if LiteLLM reports vision is not supported. (Check API provider)",
        )
        parser.add_argument(
            "--provider",
            type=str,
            help="Specific provider to use for OpenRouter requests (e.g., 'anthropic', 'openai')",
        )

        # Reasoning tokens configuration
        parser.add_argument(
            "--reasoning-effort",
            type=str,
            choices=["none", "minimal", "low", "medium", "high"],
            help=(
                "Reasoning effort level for OpenAI-style models "
                "(none/minimal/low/medium/high). 'miminimal' specific to GPT-5 and 'none' specific to GPT-5.1"
            ),
        )
        parser.add_argument(
            "--reasoning-max-tokens",
            type=int,
            help="Maximum tokens for reasoning (Anthropic-style)",
        )
        parser.add_argument(
            "--reasoning-exclude",
            action="store_true",
            default=False,
            help="Exclude reasoning tokens from response (model still uses reasoning internally)",
        )
        parser.add_argument(
            "--success-threshold",
            type=float,
            default=DEFAULT_CONFIG["success_threshold"],
            help="Percentage of scenes that must render successfully to trigger enhanced visual review mode (focuses on creative improvements instead of technical fixes)",
        )
        parser.add_argument(
            "--frame-extraction-mode",
            type=str,
            default=DEFAULT_CONFIG["frame_extraction_mode"],
            choices=["highest_density", "fixed_count"],
            help="Frame extraction mode: highest_density (single best frame) or fixed_count (multiple frames)",
        )
        parser.add_argument(
            "--frame-count",
            type=int,
            default=DEFAULT_CONFIG["frame_count"],
            help="Number of frames to extract when using fixed_count mode",
        )
        parser.add_argument(
            "--scene-timeout",
            type=int,
            default=DEFAULT_CONFIG["scene_timeout"],
            help="Maximum seconds allowed for a single scene render (set to 0 to disable)",
        )
        parser.add_argument(
            "--headless",
            action="store_true",
            default=False,
            help="Suppress most output and show only a single progress bar",
        )

        return parser

    def _validate_reasoning_arguments(self, args) -> None:
        """Validate reasoning arguments to ensure only one method is used."""
        if args.reasoning_effort and args.reasoning_max_tokens:
            self.console.print(
                "[bold red]Error: Cannot use both --reasoning_effort and --reasoning_max_tokens at the same time.[/bold red]"
            )
            exit(1)

    def _build_config(self, args) -> dict:
        """Build configuration dictionary from parsed arguments."""

        video_data = args.video_data
        if not video_data and args.video_data_file:
            try:
                with open(args.video_data_file) as f:
                    video_data = f.read().strip()
            except FileNotFoundError:
                self.console.print(
                    f"[bold red]Error: Video data file '{args.video_data_file}' not found.[/bold red]"
                )
                exit(1)
            except Exception as e:
                self.console.print(f"[bold red]Error reading video data file: {e}[/bold red]")
                exit(1)

        output_dir = args.output_dir
        short_file_desc = "output"  # Default value
        if not output_dir and video_data:
            words = video_data.split()[:4]
            if words:
                short_file_desc = (
                    "_".join(words).replace("\n", "_").replace("\r", "_").replace(" ", "_")
                )

        if not output_dir:
            model_name = args.manim_model.replace("openrouter/", "").replace("/", "_")
            output_dir = (
                f"output/{model_name}_{short_file_desc}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        # Check if both models support vision/images
        main_vision_support = supports_vision(model=args.manim_model) or args.force_vision
        review_vision_support = supports_vision(model=args.review_model) or args.force_vision
        vision_enabled = main_vision_support and review_vision_support

        # Build reasoning config
        reasoning_config = {}
        if args.reasoning_effort:
            reasoning_config["effort"] = args.reasoning_effort
        if args.reasoning_max_tokens:
            reasoning_config["max_tokens"] = args.reasoning_max_tokens
        if args.reasoning_exclude:
            reasoning_config["exclude"] = True

        if not args.headless:
            self.console.print(
                self._build_settings_table(
                    args=args,
                    output_dir=output_dir,
                    main_vision_support=main_vision_support,
                    review_vision_support=review_vision_support,
                    vision_enabled=vision_enabled,
                    reasoning_config=reasoning_config,
                )
            )
            if not self._confirm_settings():
                self.console.print(
                    "[bold yellow]Configuration not confirmed; aborting run.[/bold yellow]"
                )
                exit(0)

        # Build config from arguments
        return {
            "manim_model": args.manim_model,
            "review_model": args.review_model,
            "review_cycles": args.review_cycles,
            "output_dir": output_dir,
            "manim_logs": args.manim_logs,
            "streaming": args.streaming,
            "temperature": args.temperature,
            "no_temperature": args.no_temperature,
            "vision_enabled": vision_enabled,
            "reasoning": reasoning_config if reasoning_config else None,
            "provider": args.provider,
            "success_threshold": args.success_threshold,
            "frame_extraction_mode": args.frame_extraction_mode,
            "frame_count": args.frame_count,
            "headless": args.headless,
            "scene_timeout": None if args.scene_timeout == 0 else args.scene_timeout,
        }

    def _build_settings_table(
        self,
        *,
        args: argparse.Namespace,
        output_dir: str,
        main_vision_support: bool,
        review_vision_support: bool,
        vision_enabled: bool,
        reasoning_config: dict,
    ) -> Table:
        """Build the Rich table that displays launch settings."""

        table = Table(title="Launch Settings")
        table.add_column("Setting", style="cyan", no_wrap=True)
        table.add_column("Value", style="bold", overflow="fold")

        video_source = (
            f"Inline (--video-data): {args.video_data[:50]}"
            if args.video_data
            else args.video_data_file
        )
        temperature_value = (
            "[yellow]Disabled (--no-temperature)[/yellow]"
            if args.no_temperature
            else str(args.temperature)
        )
        scene_timeout = (
            "[yellow]Disabled[/yellow]" if args.scene_timeout == 0 else f"{args.scene_timeout}s"
        )
        reasoning_summary = self._format_reasoning_summary(reasoning_config)

        table.add_row("Output Directory", output_dir)
        table.add_row("Video Input", video_source or "video_data.txt")
        table.add_row("Manim Model", args.manim_model)
        table.add_row("Review Model", args.review_model)
        table.add_row("Review Cycles", str(args.review_cycles))
        table.add_row("Temperature", temperature_value)
        table.add_row("Streaming", self._format_bool(args.streaming))
        table.add_row("Show Manim Logs", self._format_bool(args.manim_logs))
        table.add_row("Enhance Prompt Success Threshold", f"{args.success_threshold:g}%")
        table.add_row("Frame Mode", args.frame_extraction_mode)
        table.add_row(
            "Frame Count",
            str(args.frame_count) if args.frame_extraction_mode == "fixed_count" else "1",
        )
        table.add_row("Scene Rendering Timeout", scene_timeout)
        table.add_row("Reasoning", reasoning_summary)
        table.add_row("Provider", args.provider or "Auto")
        table.add_row("Force Vision", self._format_bool(args.force_vision))
        table.add_row(
            "Vision (Main Model)",
            self._format_bool(
                main_vision_support,
                true_label="Enabled",
                false_label="Disabled",
                false_color="yellow",
            ),
        )
        table.add_row(
            "Vision (Review Model)",
            self._format_bool(
                review_vision_support,
                true_label="Enabled",
                false_label="Disabled",
                false_color="yellow",
            ),
        )
        table.add_row(
            "Vision Enabled",
            self._format_bool(
                vision_enabled,
                true_label="Yes",
                false_label="No",
                false_color="yellow",
            ),
        )

        return table

    def _format_bool(
        self,
        value: bool,
        *,
        true_label: str = "Yes",
        false_label: str = "No",
        false_color: str = "red",
    ) -> str:
        """Return a colorized string for boolean values."""
        color = "green" if value else false_color
        label = true_label if value else false_label
        return f"[{color}]{label}[/{color}]"

    def _confirm_settings(self) -> bool:
        """Prompt the user to confirm the displayed settings."""
        response = Prompt.ask("Proceed with these settings?", choices=["y", "n"], default="y")
        return response.lower() == "y"

    def _format_reasoning_summary(self, reasoning_config: dict) -> str:
        """Create a concise summary of the reasoning configuration."""
        if not reasoning_config:
            return "Disabled"

        summary_parts: list[str] = []
        if effort := reasoning_config.get("effort"):
            summary_parts.append(f"effort={effort}")
        if max_tokens := reasoning_config.get("max_tokens"):
            summary_parts.append(f"max_tokens={max_tokens}")

        return ", ".join(summary_parts)
