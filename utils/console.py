from rich.console import Console
from rich.syntax import Syntax
from rich.text import Text
from rich.live import Live
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
import time

from utils.llm import get_completion_with_retry, get_streaming_completion_with_retry

def get_response_with_status(model: str, messages: list, temperature: float, streaming: bool, status: str | None, console: Console) -> str:
    """Gets a response from the model, handling streaming if enabled."""
    start_time = time.time()
    
    if streaming:
        stream_output = Text("")
        stream_gen = get_streaming_completion_with_retry(model, messages, temperature, console)
        with Live(stream_output, refresh_per_second=10, vertical_overflow="visible", transient=True) as live:
            for token, response in stream_gen:
                stream_output.append(token)
                live.update(stream_output)
        response_text = stream_output.plain
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}"),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task(f"[bold green]Generating response [{model}]..." if not status else status, total=None)
            response_text = get_completion_with_retry(model, messages, temperature, console)
            progress.update(task, completed=True)
    
    elapsed_time = time.time() - start_time
    console.print(f"[dim italic]Request completed in {elapsed_time:.2f} seconds[/dim italic]")
    
    return response_text


def print_code_with_syntax(code: str, console: Console, title: str = "Code") -> None:
    """Prints code with syntax highlighting in a panel."""
    syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title=title, border_style="green"))
