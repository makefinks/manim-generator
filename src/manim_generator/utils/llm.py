"""Utility functions for LLM interaction"""

import re
import time
from collections.abc import Generator
from typing import Any

import litellm
from litellm import completion, model_cost
from litellm.cost_calculator import completion_cost  # type: ignore
from litellm.utils import register_model  # type: ignore
from openai import RateLimitError
from rich.console import Console
from rich.prompt import Prompt

# for safety drop unsupported params
litellm.drop_params = True


def check_and_register_models(
    models: list[str], console: Console, headless: bool = False
) -> None:
    """
    Checks if models are registered in the LiteLLM cost map.
    If not, prompts the user for costs and registers them.

    Args:
        models (list[str]): List of models to check
        console (Console): Rich console instance for output
        headless (bool): If True, skip interactive prompts and auto-skip registration
    """
    for model in models:
        if model not in model_cost:
            if headless:
                # In headless mode, silently skip registration
                continue

            console.print(
                f"[yellow]Model '{model}' is not registered in the LiteLLM cost map.[/yellow]"
            )

            # ask for input cost
            input_cost_str = Prompt.ask(
                f"[cyan]Input token cost for '{model}' per million tokens (e.g. 0.50) [Press Enter to skip][/cyan]",
                default="",
            )

            # skip when no input
            if not input_cost_str.strip():
                console.print(
                    f"[yellow]Cost registration for '{model}' skipped. Cost calculation will not work.[/yellow]"
                )
                continue

            # ask for output cost
            output_cost_str = Prompt.ask(
                f"[cyan]Output token cost for '{model}' per Million tokens (e.g. 2.00) [Press Enter to skip][/cyan]",
                default="",
            )

            # if output costs not entered, skip
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


def _build_litellm_args(
    *,
    model: str,
    messages: list[dict],
    temperature: float,
    stream: bool,
    reasoning: dict | None,
    provider: str | None,
) -> dict[str, Any]:
    """Builds a standardized argument dict for litellm.completion."""
    args: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": stream,
    }
    if reasoning is not None:
        if model.startswith("openai/"):
            # HACK: only use reasoning effort when using openai API directly, reasoning dict does not work
            # Litellm drop_params does not work here
            args["reasoning_effort"] = reasoning["effort"]
        else:
            args["reasoning"] = reasoning
    if provider is not None:
        args["provider"] = {"order": [provider]}

    return args


def get_completion_with_retry(
    model: str,
    messages: list[dict],
    temperature: float,
    console: Console,
    max_retries: int = 5,
    reasoning: dict | None = None,
    provider: str | None = None,
) -> tuple[str, dict[str, object], str | None]:
    """
    Makes a non-streaming LLM completion request with automatic retry on rate limit errors.

    Args:
        model (str): The name of the model to use for completion.
        messages (list[dict]): List of message dictionaries for the conversation.
        temperature (float): Temperature parameter.
        console (Console): Rich console instance for logging.
        max_retries (int, optional): Maximum number of retry attempts. Defaults to 5.
        reasoning (dict | None, optional): Reasoning parameters. Defaults to None.
        provider (str | None, optional): Provider to use. Defaults to None.

    Returns:
        tuple[str, dict[str, object], str | None]: A tuple containing:
            - The generated completion text from the model
            - Usage information including tokens and cost
            - Reasoning content if available, otherwise None

    Raises:
        Exception: If max retries are exceeded and still getting rate limited.
    """
    retries = 0
    while retries < max_retries:
        try:
            completion_args = _build_litellm_args(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=False,
                reasoning=reasoning,
                provider=provider,
            )

            response = completion(**completion_args)  # type: ignore
            response_content = response["choices"][0]["message"]["content"]  # type: ignore

            try:
                # calculates cost for models - only fails if model not registered and user skipped manual registration
                cost = completion_cost(response)
            except Exception:
                cost = 0.0

            # Extract usage information
            usage_info = {
                "model": model,
                "prompt_tokens": response.usage.prompt_tokens  # type: ignore
                if hasattr(response, "usage") and response.usage  # type: ignore
                else 0,
                "completion_tokens": response.usage.completion_tokens  # type: ignore
                if hasattr(response, "usage") and response.usage  # type: ignore
                else 0,
                "total_tokens": response.usage.total_tokens  # type: ignore
                if hasattr(response, "usage") and response.usage  # type: ignore
                else 0,
                "cost": cost,
            }

            # extract reasoning content if available
            reasoning_content = None
            message = response["choices"][0]["message"]  # type: ignore
            if hasattr(message, "reasoning_content") and message.reasoning_content:
                reasoning_content = message.reasoning_content

            return response_content, usage_info, reasoning_content

        except RateLimitError as e:
            # Extract wait time from error message if available
            # this is openrouter specific
            match = re.search(r"try again in (\d+\.?\d*)s", str(e))
            if match:
                wait_time = float(match.group(1))
                console.log(
                    f"[bold yellow]Rate limited. Waiting for {wait_time} seconds...[/bold yellow]"
                )
                time.sleep(wait_time + 2)
            else:
                wait_time = 2
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

    raise Exception("[bold red]Max retries exceeded.[/bold red]")


def get_streaming_completion_with_retry(
    model: str,
    messages: list[dict],
    temperature: float,
    console: Console,
    max_retries: int = 5,
    reasoning: dict | None = None,
    provider: str | None = None,
) -> Generator[tuple[str, str, dict[str, object]], None, None]:
    """
    Makes a streaming LLM completion request with automatic retry on rate limit errors.

    Args:
        model (str): The name of the model to use for completion.
        messages (list[dict]): List of message dictionaries for the conversation.
        temperature (float): Temperature parameter.
        console (Console): Rich console instance for logging.
        max_retries (int, optional): Maximum number of retry attempts. Defaults to 5.
        reasoning (dict, optional): Reasoning parameters. Defaults to None.
        provider (str, optional): Provider to use. Defaults to None.

    Yields:
        tuple[str, str, dict[str, object]]: A tuple containing:
            - The current token
            - The full accumulated response so far
            - Usage information (empty for streaming)

    Raises:
        Exception: If max retries are exceeded and still getting rate limited.
    """
    retries = 0
    while retries < max_retries:
        try:
            completion_args = _build_litellm_args(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=True,
                reasoning=reasoning,
                provider=provider,
            )

            response = completion(**completion_args)  # type: ignore
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
                    token = chunk.choices[0].delta.content or ""  # type: ignore
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
                wait_time = 2
                console.log(
                    f"[bold yellow]Rate limited. Waiting for {wait_time} seconds...[/bold yellow]"
                )
                time.sleep(wait_time)
            retries += 1

    raise Exception("[bold red]Max retries exceeded.[/bold red]")
