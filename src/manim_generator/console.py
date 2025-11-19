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
    temperature: float | None,
    streaming: bool,
    status: str | None,
    console: Console,
    reasoning: dict | None = None,
    provider: str | None = None,
    headless: bool = False,
) -> tuple[str, dict[str, object], str | None]:
    """Gets a response from the model, handling streaming if enabled.

    Returns:
        tuple[str, dict[str, object], str | None]: Response text, usage information, and optional reasoning content
    """
    start_time = time.time()
    reasoning_content = None

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
        full_reasoning = ""
        reasoning_started = False
        answer_started = False

        for chunk in stream_gen:
            if chunk.reasoning_token:
                if not reasoning_started:
                    console.print("\n[dim #C0C0C0]Reasoning:[/dim #C0C0C0] ", end="\n")
                    reasoning_started = True
                console.print(chunk.reasoning_token, end="", style="dim #C0C0C0")
            if chunk.token:
                if reasoning_started and not answer_started:
                    console.print("\n[bold green]Answer:\n[/bold green] ", end="")
                    answer_started = True
                console.print(chunk.token, end="")
            full_response = chunk.response
            usage_info = chunk.usage
            full_reasoning = chunk.reasoning_content

        response_text = full_response
        reasoning_content = full_reasoning
    elif headless:
        result = get_completion_with_retry(
            model=model,
            messages=messages,
            temperature=temperature,
            console=console,
            reasoning=reasoning,
            provider=provider,
        )
        response_text = result.content
        usage_info = result.usage
        reasoning_content = result.reasoning
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

            result = get_completion_with_retry(
                model=model,
                messages=messages,
                temperature=temperature,
                console=console,
                reasoning=reasoning,
                provider=provider,
            )
            response_text = result.content
            usage_info = result.usage
            reasoning_content = result.reasoning
            progress.update(task, completed=True)

    elapsed_time = time.time() - start_time
    if not headless:
        reasoning_tokens = usage_info.get("reasoning_tokens", 0)
        answer_tokens = usage_info.get("answer_tokens", usage_info.get("completion_tokens", 0))
        console.print(
            "[dim italic]"
            f"Request completed in {elapsed_time:.2f} seconds | "
            f"Input Tokens: {usage_info.get('prompt_tokens', 0)} | "
            f"Output Tokens: {usage_info.get('completion_tokens', 0)} "
            f"(reasoning: {reasoning_tokens}, answer: {answer_tokens}) | "
            f"Cost: ${usage_info.get('cost', 0):.6f}"
            "[/dim italic]"
        )

    return response_text, usage_info, reasoning_content


def print_code_with_syntax(code: str, console: Console, title: str = "Code") -> None:
    """Prints code with syntax highlighting in a panel."""
    syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title=title, border_style="green"))
