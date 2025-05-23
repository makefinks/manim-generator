from rich.console import Console
from rich.syntax import Syntax
from rich.text import Text
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
import time
from typing import Dict, Any

from utils.llm import get_completion_with_retry, get_streaming_completion_with_retry


def get_response_with_status(
    model: str,
    messages: list,
    temperature: float,
    streaming: bool,
    status: str | None,
    console: Console,
    reasoning: dict = None,
) -> tuple[str, Dict[str, Any]]:
    """Gets a response from the model, handling streaming if enabled.

    Returns:
        tuple[str, Dict[str, Any]]: Response text and usage information
    """
    start_time = time.time()

    if streaming:
        stream_output = Text("")
        stream_gen_args = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "console": console,
        }
        if reasoning is not None:
            stream_gen_args["reasoning"] = reasoning

        stream_gen = get_streaming_completion_with_retry(**stream_gen_args)
        usage_info = {}
        with Live(
            stream_output,
            refresh_per_second=10,
            vertical_overflow="visible",
            transient=True,
        ) as live:
            for token, response, usage in stream_gen:
                if token:  # Only append non-empty tokens
                    stream_output.append(token)
                    live.update(stream_output)
                usage_info = usage  # Keep updating with latest usage info
        response_text = stream_output.plain
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
            completion_args = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "console": console,
            }
            if reasoning is not None:
                completion_args["reasoning"] = reasoning

            response_text, usage_info, reasoning_content = get_completion_with_retry(**completion_args)
            progress.update(task, completed=True)

    elapsed_time = time.time() - start_time

    # Display token and cost information
    console.print(
        f"[dim italic]Request completed in {elapsed_time:.2f} seconds | Tokens: {usage_info.get('total_tokens', 0)} | Cost: ${usage_info.get('cost', 0):.6f}[/dim italic]"
    )

    return response_text, usage_info, reasoning_content if not streaming else None


def print_code_with_syntax(code: str, console: Console, title: str = "Code") -> None:
    """Prints code with syntax highlighting in a panel."""
    syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title=title, border_style="green"))
