"""Utility functions for LLM interaction"""

import re
import time
from typing import Generator, Dict, Any
from litellm import completion, completion_cost, model_cost, register_model
from openai import RateLimitError
from rich.console import Console
from rich.prompt import Prompt


def check_and_register_models(models: list[str], console: Console) -> None:
    """
    Checks if models are registered in the LiteLLM cost map.
    If not, prompts the user for costs and registers them.

    Args:
        models (list[str]): List of models to check
        console (Console): Rich console instance for output
    """
    for model in models:
        if model not in model_cost:
            console.print(
                f"[yellow]Model '{model}' is not registered in the LiteLLM cost map.[/yellow]"
            )

            # Ask for input costs
            input_cost_str = Prompt.ask(
                f"[cyan]Input token cost for '{model}' per million tokens (e.g. 0.50) [Press Enter to skip][/cyan]",
                default="",
            )

            # If nothing entered, skip registration
            if not input_cost_str.strip():
                console.print(
                    f"[yellow]Cost registration for '{model}' skipped. Cost calculation will not work.[/yellow]"
                )
                continue

            # Ask for output costs
            output_cost_str = Prompt.ask(
                f"[cyan]Output token cost for '{model}' per Million tokens (e.g. 2.00) [Press Enter to skip][/cyan]",
                default="",
            )

            # If output costs not entered, skip
            if not output_cost_str.strip():
                console.print(
                    f"[yellow]Cost registration for '{model}' skipped. Cost calculation will not work.[/yellow]"
                )
                continue

            try:
                input_cost = float(input_cost_str) / 1_000_000
                output_cost = float(output_cost_str) / 1_000_000

                # Register the model
                register_model(
                    {
                        model: {
                            "input_cost_per_token": input_cost,
                            "output_cost_per_token": output_cost,
                        }
                    }
                )

                console.print(
                    f"[green]Model '{model}' successfully registered with costs: ${input_cost}/${output_cost} per token (input/output)[/green]"
                )

            except ValueError:
                console.print(
                    f"[red]Invalid cost values entered. Cost registration for '{model}' skipped.[/red]"
                )


def get_completion_with_retry(
    model: str,
    messages: list[dict],
    temperature: float,
    console: Console,
    max_retries: int = 5,
    reasoning: dict = None,
) -> tuple[str, Dict[str, Any]]:
    """
    Makes a non-streaming LLM completion request with automatic retry on rate limit errors.

    Args:
        model (str): The name of the model to use for completion.
        messages (list[dict]): List of message dictionaries for the conversation.
        temperature (int): Temperature parameter.
        console (Console): Rich console instance for logging.
        max_retries (int, optional): Maximum number of retry attempts. Defaults to 5.

    Returns:
        tuple[str, Dict[str, Any]]: A tuple containing:
            - The generated completion text from the model
            - Usage information including tokens and cost

    Raises:
        Exception: If max retries are exceeded and still getting rate limited.
    """
    retries = 0
    while retries < max_retries:
        try:
            completion_args = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "stream": False,
            }

            if reasoning is not None:
                completion_args["reasoning"] = reasoning

            response = completion(**completion_args)
            response_content = response["choices"][0]["message"]["content"]

            try:
                cost = completion_cost(response)
            # for example if model not in litellm model map, then just ignore cost calculations
            except Exception:
                cost = 0.0

            # Extract usage information
            usage_info = {
                "model": model,
                "prompt_tokens": response.usage.prompt_tokens
                if hasattr(response, "usage") and response.usage
                else 0,
                "completion_tokens": response.usage.completion_tokens
                if hasattr(response, "usage") and response.usage
                else 0,
                "total_tokens": response.usage.total_tokens
                if hasattr(response, "usage") and response.usage
                else 0,
                "cost": cost,
            }

            # Extract reasoning content if available
            reasoning_content = None
            message = response["choices"][0]["message"]
            if hasattr(message, "reasoning_content") and message.reasoning_content:
                reasoning_content = message.reasoning_content

            return response_content, usage_info, reasoning_content

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
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")
            empty_usage = {
                "model": model,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost": 0.0,
            }
            return "Review model failed to generate response.", empty_usage, None

    raise Exception("[bold red]Max retries exceeded. Still rate limited.[/bold red]")


def get_streaming_completion_with_retry(
    model: str,
    messages: list[dict],
    temperature: float,
    console: Console,
    max_retries: int = 5,
    reasoning: dict = None,
) -> Generator[tuple[str, str, Dict[str, Any]], None, None]:
    """
    Makes a streaming LLM completion request with automatic retry on rate limit errors.

    Args:
        model (str): The name of the model to use for completion.
        messages (list[dict]): List of message dictionaries for the conversation.
        temperature (int): Temperature parameter.
        console (Console): Rich console instance for logging.
        max_retries (int, optional): Maximum number of retry attempts. Defaults to 5.

    Yields:
        tuple[str, str, Dict[str, Any]]: A tuple containing:
            - The current token
            - The full accumulated response so far
            - Usage information (empty for streaming)

    Raises:
        Exception: If max retries are exceeded and still getting rate limited.
    """
    retries = 0
    while retries < max_retries:
        try:
            completion_args = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "stream": True,
            }

            if reasoning is not None:
                completion_args["reasoning"] = reasoning

            response = completion(**completion_args)
            full_response = ""
            empty_usage = {
                "model": model,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost": 0.0,
            }

            for chunk in response:
                try:
                    token = chunk.choices[0].delta.content or ""
                    full_response += token
                    yield (token, full_response, empty_usage)
                except Exception as e:
                    console.print(
                        f"[bold red]Error processing stream chunk: {e}[/bold red]"
                    )
                    raise e

            # Return final empty usage for streaming
            yield ("", full_response, empty_usage)
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
