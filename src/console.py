from rich.console import Console, ConsoleOptions, RenderResult
from rich.syntax import Syntax
from rich.text import Text
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn
from rich.markdown import CodeBlock, Markdown
import time
from typing import Optional

from src.utils.llm import get_completion_with_retry, get_streaming_completion_with_retry
from src.utils.progress import ProgressManager


def is_headless_mode(progress_manager: Optional[ProgressManager] = None) -> bool:
    """Check if we're running in headless mode.
    
    Args:
        progress_manager: Optional ProgressManager instance to check headless state
        
    Returns:
        bool: True if running in headless mode, False otherwise
    """
    if progress_manager is not None:
        return progress_manager.is_headless
    return False


def create_progress_only_console(progress_manager: ProgressManager) -> Console:
    """Create a console instance optimized for progress-only display in headless mode.
    
    Args:
        progress_manager: ProgressManager instance for headless mode
        
    Returns:
        Console: Configured console for minimal output
    """
    if not progress_manager.is_headless:
        return Console()
    
    # In headless mode, create a minimal console that only shows progress
    console = Console(
        quiet=False,  # Allow progress output
        stderr=True,  # Use stderr for progress to keep stdout clean
        force_terminal=False,  # Don't force terminal colors in headless mode
    )
    
    return console


def get_response_with_progress_aware_status(
    model: str,
    messages: list,
    temperature: float,
    streaming: bool,
    status: str | None,
    console: Console,
    progress_manager: Optional[ProgressManager] = None,
    reasoning: dict | None = None,
    provider: str | None = None,
) -> tuple[str, dict[str, object], str | None]:
    """Gets a response with progress-aware status display.
    
    This function adapts the status display based on whether we're in headless mode.
    In headless mode, it uses the ProgressManager for clean progress updates.
    In normal mode, it falls back to the standard Rich display.
    
    Returns:
        tuple[str, dict[str, object], str | None]: Response text, usage info, reasoning content
    """
    if is_headless_mode(progress_manager) and progress_manager is not None:
        # In headless mode, use ProgressManager for clean status updates
        if status:
            progress_manager.update_step(status)
        
        # Get response without Rich progress display
        start_time = time.time()
        response_text, usage_info, reasoning_content = get_completion_with_retry(
            model=model,
            messages=messages,
            temperature=temperature,
            console=console,
            reasoning=reasoning,
            provider=provider,
        )
        elapsed_time = time.time() - start_time
        
        # Update progress with completion
        if status:
            progress_manager.update_step(f"{status} ✓")
            
        return response_text, usage_info, reasoning_content
    else:
        # In normal mode, use standard Rich display
        return get_response_with_status(
            model, messages, temperature, streaming, status, console, reasoning, provider
        )


def prettier_code_blocks():
    """Make rich code blocks prettier and easier to copy.

    From https://github.com/samuelcolvin/aicli/blob/v0.8.0/samuelcolvin_aicli.py#L22
    """

    class SimpleCodeBlock(CodeBlock):
        def __rich_console__(
            self, console: Console, options: ConsoleOptions
        ) -> RenderResult:
            code = str(self.text).rstrip()
            yield Text(self.lexer_name, style="dim")
            # Use a dark background similar to the non-streaming Panel display
            yield Syntax(
                code,
                self.lexer_name,
                theme="monokai",
                background_color="#1e1e1e",
                word_wrap=True,
                line_numbers=True,
            )
            yield Text(f"/{self.lexer_name}", style="dim")

    Markdown.elements["fence"] = SimpleCodeBlock


def get_response_with_status(
    model: str,
    messages: list,
    temperature: float,
    streaming: bool,
    status: str | None,
    console: Console,
    reasoning: dict | None = None,
    provider: str | None = None,
) -> tuple[str, dict[str, object], str | None]:
    """Gets a response from the model, handling streaming if enabled.

    Returns:
        tuple[str, dict[str, object], str | None]: Response text, usage information, and optional reasoning content
    """
    start_time = time.time()
    reasoning_content = None

    # TODO: Streaming implementation has flickering and leaves artificats on scroll - needs fix
    if streaming:
        prettier_code_blocks()
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

        with Live("", console=console, vertical_overflow="visible") as live:
            for token, response, usage in stream_gen:
                if token:
                    full_response = response
                    live.update(Markdown(full_response))
                usage_info = usage

        response_text = full_response
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}"),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task(
                f"[bold green]Generating response [{model}]..."
                if not status
                else status,
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
    console.print(
        f"[dim italic]Request completed in {elapsed_time:.2f} seconds | Input Tokens: {usage_info.get('prompt_tokens', 0)} | Output Tokens: {usage_info.get('completion_tokens', 0)} | Cost: ${usage_info.get('cost', 0):.6f}[/dim italic]"
    )

    return response_text, usage_info, reasoning_content if not streaming else None


def configure_progress_bar(progress_manager: ProgressManager) -> Progress:
    """Configure Rich progress bar for headless mode.
    
    Args:
        progress_manager: ProgressManager instance to configure progress display for
        
    Returns:
        Progress: Configured Rich Progress instance
    """
    if not progress_manager.is_headless:
        # Return standard progress configuration for non-headless mode
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}"),
            TimeElapsedColumn(),
            transient=True,
        )
    
    # Enhanced progress bar configuration for headless mode
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.fields[stage]}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TextColumn("{task.fields[step]}"),
        console=create_progress_only_console(progress_manager),
        refresh_per_second=2,  # Reduce refresh rate for better performance
    )


def print_code_with_syntax(code: str, console: Console, title: str = "Code") -> None:
    """Prints code with syntax highlighting in a panel."""
    syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title=title, border_style="green"))
