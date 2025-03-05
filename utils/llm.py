"""Utility functions for LLM interaction"""

import re
import time
from typing import Generator
from litellm import completion
from openai import RateLimitError
from rich.console import Console

def get_completion_with_retry(
    model: str,
    messages: list[dict],
    temperature: float,
    console: Console,
    max_retries: int = 5
) -> str:
    """
    Makes a non-streaming LLM completion request with automatic retry on rate limit errors.

    Args:
        model (str): The name of the model to use for completion.
        messages (list[dict]): List of message dictionaries for the conversation.
        temperature (int): Temperature parameter.
        console (Console): Rich console instance for logging.
        max_retries (int, optional): Maximum number of retry attempts. Defaults to 5.

    Returns:
        str: The generated completion text from the model.

    Raises:
        Exception: If max retries are exceeded and still getting rate limited.
    """
    retries = 0
    while retries < max_retries:
        try:
            response = completion(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=False                
            )
            response_content = response["choices"][0]["message"]["content"]
            return response_content

        except RateLimitError as e:
            # Extract wait time from error message if available
            match = re.search(r"try again in (\d+\.?\d*)s", str(e))
            if match:
                wait_time = float(match.group(1))
                console.log(
                    f"[bold yellow]Rate limited. Waiting for {wait_time} seconds...[/bold yellow]"
                )
                time.sleep(wait_time + 2)  # Add a buffer to the wait time
            else:
                wait_time = 2  # Default wait time
                console.log(
                    f"[bold yellow]Rate limited. Waiting for {wait_time} seconds...[/bold yellow]"
                )
                time.sleep(wait_time)
            retries += 1

    raise Exception("[bold red]Max retries exceeded. Still rate limited.[/bold red]")


def get_streaming_completion_with_retry(
    model: str,
    messages: list[dict],
    temperature: float,
    console: Console,
    max_retries: int = 5
) -> Generator[tuple[str, str], None, None]:
    """
    Makes a streaming LLM completion request with automatic retry on rate limit errors.

    Args:
        model (str): The name of the model to use for completion.
        messages (list[dict]): List of message dictionaries for the conversation.
        temperature (int): Temperature parameter.
        console (Console): Rich console instance for logging.
        max_retries (int, optional): Maximum number of retry attempts. Defaults to 5.

    Yields:
        tuple[str, str]: A tuple containing the current token and the full accumulated response so far.

    Raises:
        Exception: If max retries are exceeded and still getting rate limited.
    """
    retries = 0
    while retries < max_retries:
        try:
            response = completion(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=True
            )
            full_response = ""
            for chunk in response:
                try:
                    token = chunk.choices[0].delta.content or ""
                    full_response += token
                    yield (token, full_response)
                except Exception as e:
                    console.print(f"[bold red]Error processing stream chunk: {e}[/bold red]")
                    raise e
            return  # End of streaming
        except RateLimitError as e:
            match = re.search(r"try again in (\d+\.?\d*)s", str(e))
            if match:
                wait_time = float(match.group(1))
                console.log(
                    f"[bold yellow]Rate limited. Waiting for {wait_time} seconds...[/bold yellow]"
                )
                time.sleep(wait_time + 2)
            else:
                wait_time = 2  # Default wait time
                console.log(
                    f"[bold yellow]Rate limited. Waiting for {wait_time} seconds...[/bold yellow]"
                )
                time.sleep(wait_time)
            retries += 1

    raise Exception("[bold red]Max retries exceeded. Still rate limited.[/bold red]")

