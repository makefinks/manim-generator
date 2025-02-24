from rich.console import Console
from rich.syntax import Syntax
from rich.text import Text
from rich.live import Live
import time

from utils.llm import get_completion_with_retry, get_streaming_completion_with_retry

def get_response_with_status(model: str, messages: list, temperature: float, streaming: bool, status: str | None, console: Console) -> str:
    """Gets a response from the model, handling streaming if enabled."""
    start_time = time.time()
    
    if streaming:
        stream_output = Text("")
        stream_gen = get_streaming_completion_with_retry(model, messages, temperature, console)
        with Live(stream_output, refresh_per_second=5, vertical_overflow="visible", transient=True) as live:
            for token, response in stream_gen:
                stream_output.append(token)
                live.update(stream_output)
        response_text = stream_output.plain
    else:
        with console.status("[bold green]Generating response..." if not status else status) as s:
            response_text = get_completion_with_retry(model, messages, temperature, console)
    
    elapsed_time = time.time() - start_time
    console.print(f"[dim italic]Request completed in {elapsed_time:.2f} seconds[/dim italic]")
    
    return response_text


def print_code_with_syntax(code: str, console: Console) -> None:
    """Prints code with syntax highlighting."""
    console.print(Syntax(code, "python", theme="monokai", line_numbers=True))