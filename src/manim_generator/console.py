import time

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.syntax import Syntax

from manim_generator.utils.llm import get_completion_with_retry, get_streaming_completion_with_retry


class HeadlessProgressManager:
    """Manages a single progress bar for headless mode."""

    def __init__(self, console: Console, total_cycles: int):
        self.console = console
        self.total_cycles = total_cycles
        self.current_cycle = 0
        self.execution_count = 0
        self.successful_executions = 0
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        )
        self.task_id = None
        self.progress_started = False

    def start(self):
        """Start the progress display."""
        if not self.progress_started:
            self.progress.start()
            self.task_id = self.progress.add_task(
                "Initializing...", total=self._calculate_total_steps()
            )
            self.progress_started = True

    def _calculate_total_steps(self) -> int:
        """Calculate total number of steps in the workflow."""
        return 2 + (self.total_cycles * 3)

    def update(self, phase: str, extra_info: str = ""):
        """Update the progress bar with current phase."""
        if not self.progress_started:
            self.start()

        if self.task_id is None:
            return

        description = f"{phase}"
        if extra_info:
            description += f" | {extra_info}"

        if self.execution_count > 0:
            description += (
                f" | Executions: {self.execution_count} ({self.successful_executions} successful)"
            )

        current_step = self._get_current_step(phase)
        self.progress.update(self.task_id, description=description, completed=current_step)

    def _get_current_step(self, phase: str) -> int:
        """Calculate current step number based on phase."""
        if "Initial Code Generation" in phase:
            return 0
        elif "Initial Execution" in phase:
            return 1
        elif "Review Cycle" in phase:
            return 2 + (self.current_cycle - 1) * 3
        elif "Code Revision" in phase:
            return 2 + (self.current_cycle - 1) * 3 + 1
        elif "Execution" in phase and self.current_cycle > 0:
            return 2 + (self.current_cycle - 1) * 3 + 2
        elif "Finalization" in phase:
            return self._calculate_total_steps()
        return 0

    def set_cycle(self, cycle: int):
        """Set the current cycle number."""
        self.current_cycle = cycle

    def increment_execution(self, success: bool):
        """Increment execution counters."""
        self.execution_count += 1
        if success:
            self.successful_executions += 1

    def stop(self):
        """Stop the progress display."""
        if self.progress_started:
            self.progress.stop()
            self.progress_started = False


def get_response_with_status(
    model: str,
    messages: list,
    temperature: float,
    streaming: bool,
    status: str | None,
    console: Console,
    reasoning: dict | None = None,
    provider: str | None = None,
    headless: bool = False,
    headless_manager=None,
) -> tuple[str, dict[str, object], str | None]:
    """Gets a response from the model, handling streaming if enabled.

    Returns:
        tuple[str, dict[str, object], str | None]: Response text, usage information, and optional reasoning content
    """
    start_time = time.time()
    reasoning_content = None

    # TODO: Streaming implementation has flickering and leaves artificats on scroll - needs fix
    if streaming and not headless:
        stream_gen = get_streaming_completion_with_retry(
            model=model,
            messages=messages,
            temperature=temperature,
            console=console,
            reasoning=reasoning,
            provider=provider,
        )
        usage_info: dict[str, object] = {}
        full_response = ""

        for token, response, usage in stream_gen:
            if token:
                console.print(token, end="")
            full_response = response
            usage_info = usage

        response_text = full_response
    elif headless:
        response_text, usage_info, reasoning_content = get_completion_with_retry(
            model=model,
            messages=messages,
            temperature=temperature,
            console=console,
            reasoning=reasoning,
            provider=provider,
        )
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}"),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task(
                f"[bold green]Generating response [{model}]..." if not status else status,
                total=None,
            )

            response_text, usage_info, reasoning_content = get_completion_with_retry(
                model=model,
                messages=messages,
                temperature=temperature,
                console=console,
                reasoning=reasoning,
                provider=provider,
            )
            progress.update(task, completed=True)

    elapsed_time = time.time() - start_time
    if not headless:
        console.print(
            f"[dim italic]Request completed in {elapsed_time:.2f} seconds | Input Tokens: {usage_info.get('prompt_tokens', 0)} | Output Tokens: {usage_info.get('completion_tokens', 0)} | Cost: ${usage_info.get('cost', 0):.6f}[/dim italic]"
        )

    return response_text, usage_info, reasoning_content if not streaming else None


def print_code_with_syntax(code: str, console: Console, title: str = "Code") -> None:
    """Prints code with syntax highlighting in a panel."""
    syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title=title, border_style="green"))
