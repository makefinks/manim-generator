from rich.console import Console
from rich.syntax import Syntax
from rich.text import Text
from rich.live import Live

from utils.llm import get_completion_with_retry, get_streaming_completion_with_retry

def get_response_with_status(model: str, messages: list, temperature: float, streaming: bool, status: str | None, console: Console) -> str:
    """Gets a response from the model, handling streaming if enabled."""
    if streaming:
        stream_output = Text("")
        stream_gen = get_streaming_completion_with_retry(model, messages, temperature, console)
        with Live(stream_output, refresh_per_second=5, vertical_overflow="visible", transient=True) as live:
            for token, response in stream_gen:
                stream_output.append(token)
                live.update(stream_output)
        return stream_output.plain  # Adjust based on how you want to extract the complete response.
    else:
        with console.status("[bold green]Generating response..." if not status else status) as s:
            return get_completion_with_retry(model, messages, temperature, console)


def print_code_with_syntax(code: str, console: Console) -> None:
    """Prints code with syntax highlighting."""
    console.print(Syntax(code, "python", theme="monokai", line_numbers=True))