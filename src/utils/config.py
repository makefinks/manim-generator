from datetime import datetime
import argparse
import os
from rich.console import Console
from rich.table import Table
from litellm import supports_vision, completion

# Default configuration
DEFAULT_CONFIG = {
    "manim_model": "openrouter/anthropic/claude-sonnet-4",
    "review_model": "openrouter/anthropic/claude-sonnet-4",
    "review_cycles": 5,
    "manim_logs": False,
    "streaming": False,
    "temperature": 0.4,
    "success_threshold": 100,
    "output_dir": None,
    "frame_extraction_mode": "fixed_count",
    "frame_count": 3,
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
        parser = argparse.ArgumentParser(
            description="Generate Manim animations using AI"
        )

        # Video data input
        parser.add_argument(
            "--video_data", type=str, help="Description of the video to generate"
        )
        parser.add_argument(
            "--video_data_file",
            type=str,
            default="video_data.txt",
            help="Path to file containing video description",
        )

        # Model configuration
        parser.add_argument(
            "--manim_model",
            type=str,
            default=DEFAULT_CONFIG["manim_model"],
            help="Model to use for generating Manim code",
        )
        parser.add_argument(
            "--review_model",
            type=str,
            default=DEFAULT_CONFIG["review_model"],
            help="Model to use for reviewing code",
        )

        # Process configuration
        parser.add_argument(
            "--review_cycles",
            type=int,
            default=DEFAULT_CONFIG["review_cycles"],
            help="Number of review cycles to perform",
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            default=None,
            help=(
                "Directory to save outputs (overrides the auto-generated folder name)"
            ),
        )
        parser.add_argument(
            "--manim_logs",
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
            "--force_vision",
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
            "--reasoning_effort",
            type=str,
            choices=["minimal", "low", "medium", "high"],
            help="Reasoning effort level for OpenAI-style models (minimal/low/medium/high). Note: Minimal is only to be used with GPT-5",
        )
        parser.add_argument(
            "--reasoning_max_tokens",
            type=int,
            help="Maximum tokens for reasoning (Anthropic-style)",
        )
        parser.add_argument(
            "--reasoning_exclude",
            action="store_true",
            default=False,
            help="Exclude reasoning tokens from response (model still uses reasoning internally)",
        )
        parser.add_argument(
            "--success_threshold",
            type=float,
            default=DEFAULT_CONFIG["success_threshold"],
            help="Percentage of scenes that must render successfully to trigger enhanced visual review mode (focuses on creative improvements instead of technical fixes)",
        )
        parser.add_argument(
            "--frame_extraction_mode",
            type=str,
            default=DEFAULT_CONFIG["frame_extraction_mode"],
            choices=["highest_density", "fixed_count"],
            help="Frame extraction mode: highest_density (single best frame) or fixed_count (multiple frames)",
        )
        parser.add_argument(
            "--frame_count",
            type=int,
            default=DEFAULT_CONFIG["frame_count"],
            help="Number of frames to extract when using fixed_count mode",
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
                with open(args.video_data_file, "r") as f:
                    video_data = f.read().strip()
            except FileNotFoundError:
                self.console.print(
                    f"[bold red]Error: Video data file '{args.video_data_file}' not found.[/bold red]"
                )
                exit(1)
            except Exception as e:
                self.console.print(
                    f"[bold red]Error reading video data file: {e}[/bold red]"
                )
                exit(1)

        output_dir = args.output_dir
        short_file_desc = "manim_animation"
        if not output_dir and video_data:
            try:
                response = completion(
                    model=args.manim_model,
                    reasoning_effort="low",
                    messages=[
                        {
                            "content": (
                                "Generate a max 4 word file descriptor for this content, "
                                "no suffix, underscores instead of spaces. Answer with "
                                f"nothing else!: \n {video_data}"
                            ),
                            "role": "user",
                        }
                    ],
                )
                short_file_desc = response.choices[0].message.content
            except Exception as e:
                print(e)
                self.console.print(
                    f"[bold yellow]Warning: Could not generate file descriptor: {e}. Using default.[/bold yellow]"
                )

        if not output_dir:
            output_dir = f"{short_file_desc}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print("File descriptor: " + short_file_desc)
        # Check if both models support vision/images
        main_vision_support = (
            supports_vision(model=args.manim_model) or args.force_vision
        )
        review_vision_support = (
            supports_vision(model=args.review_model) or args.force_vision
        )
        vision_enabled = main_vision_support and review_vision_support

        # Display vision support as a table
        table = Table(title="Vision Support Check")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="bold")
        table.add_row(
            "Main Model",
            f"[{'green' if main_vision_support else 'yellow'}]{'Enabled' if main_vision_support else 'Disabled'}[/{'green' if main_vision_support else 'yellow'}]",
        )
        table.add_row(
            "Review Model",
            f"[{'green' if review_vision_support else 'yellow'}]{'Enabled' if review_vision_support else 'Disabled'}[/{'green' if review_vision_support else 'yellow'}]",
        )
        table.add_row(
            "Combined",
            f"[{'green' if vision_enabled else 'yellow'}]{'Enabled' if vision_enabled else 'Disabled'}[/{'green' if vision_enabled else 'yellow'}]",
        )
        self.console.print(table)

        # Build reasoning config
        reasoning_config = {}
        if args.reasoning_effort:
            reasoning_config["effort"] = args.reasoning_effort
        if args.reasoning_max_tokens:
            reasoning_config["max_tokens"] = args.reasoning_max_tokens
        if args.reasoning_exclude:
            reasoning_config["exclude"] = True

        # Build config from arguments
        return {
            "manim_model": args.manim_model,
            "review_model": args.review_model,
            "review_cycles": args.review_cycles,
            "output_dir": output_dir,
            "manim_logs": args.manim_logs,
            "streaming": args.streaming,
            "temperature": args.temperature,
            "vision_enabled": vision_enabled,
            "reasoning": reasoning_config if reasoning_config else None,
            "provider": args.provider,
            "success_threshold": args.success_threshold,
            "frame_extraction_mode": args.frame_extraction_mode,
            "frame_count": args.frame_count,
        }
