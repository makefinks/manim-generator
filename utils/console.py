from rich.console import Console, ConsoleOptions, RenderResult
from rich.syntax import Syntax
from rich.text import Text
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.markdown import CodeBlock, Markdown
import time
from typing import Dict, Any

from utils.llm import get_completion_with_retry, get_streaming_completion_with_retry


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
                background_color="#1e1e1e",  # Dark background to match Panel style
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
) -> tuple[str, Dict[str, Any], str | None]:
    """Gets a response from the model, handling streaming if enabled.

    Returns:
        tuple[str, Dict[str, Any], str | None]: Response text, usage information, and optional reasoning content
    """
    start_time = time.time()

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
        usage_info: Dict[str, Any] = {}
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

    # Display token and cost information
    console.print(
        f"[dim italic]Request completed in {elapsed_time:.2f} seconds | Input Tokens: {usage_info.get('prompt_tokens', 0)} | Output Tokens: {usage_info.get('completion_tokens', 0)} | Cost: ${usage_info.get('cost', 0):.6f}[/dim italic]"
    )

    return response_text, usage_info, reasoning_content if not streaming else None


def print_code_with_syntax(code: str, console: Console, title: str = "Code") -> None:
    """Prints code with syntax highlighting in a panel."""
    syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title=title, border_style="green"))
