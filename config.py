from datetime import datetime
import argparse
import os
from rich.console import Console
from litellm import supports_vision

# Default configuration
DEFAULT_CONFIG = {
    "manim_model": "anthropic/claude-sonnet-4",
    "review_model": "anthropic/claude-sonnet-4",
    "review_cycles": 5,
    "output_dir": f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    "manim_logs": False,
    "streaming": False,
    "temperature": 0.4,
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
            default=DEFAULT_CONFIG["output_dir"],
            help="Directory to save outputs",
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
            choices=["low", "medium", "high"],
            help="Reasoning effort level for OpenAI-style models (low/medium/high)",
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
        # Check if the model supports vision/images
        vision_enabled = supports_vision(model=args.review_model) or args.force_vision
        self.console.print(
            f"[bold {'green' if vision_enabled else 'yellow'}]Vision support: {'Enabled' if vision_enabled else 'Disabled'}[/bold {'green' if vision_enabled else 'yellow'}]"
        )

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
            "output_dir": args.output_dir,
            "manim_logs": args.manim_logs,
            "streaming": args.streaming,
            "temperature": args.temperature,
            "vision_enabled": vision_enabled,
            "reasoning": reasoning_config if reasoning_config else None,
            "provider": args.provider,
        }
