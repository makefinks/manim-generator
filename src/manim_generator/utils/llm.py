"""Utility functions for LLM interaction"""

import time
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import litellm
from litellm import RateLimitError, completion, model_cost
from litellm.cost_calculator import completion_cost  # type: ignore
from litellm.utils import register_model  # type: ignore
from rich.console import Console
from rich.prompt import Prompt


@dataclass
class LiteLLMParams:
    """Container for LiteLLM completion configuration."""

    model: str
    messages: list[dict]
    stream: bool
    temperature: float | None = None
    reasoning: dict | None = None
    provider: str | None = None

    def to_kwargs(self) -> dict[str, Any]:
        """Convert parameters to litellm.completion kwargs."""
        args: dict[str, Any] = {
            "model": self.model,
            "messages": self.messages,
            "stream": self.stream,
        }
        if self.temperature is not None:
            args["temperature"] = self.temperature
        if self.reasoning is not None:
            if self._requires_openai_reasoning_effort():
                # OpenAI endpoints expect reasoning_effort instead of reasoning dict
                args["reasoning_effort"] = self.reasoning["effort"]
            else:
                args["reasoning"] = self.reasoning
        if self.provider is not None:
            args["provider"] = {"order": [self.provider]}
        return args

    def _requires_openai_reasoning_effort(self) -> bool:
        """
        Detects when to send `reasoning_effort` instead of `reasoning`.

        OpenAI's direct API uses `reasoning_effort`; routers typically accept
        the full reasoning payload.
        """
        if self.provider == "openai":
            return True
        return self.model.startswith(("openai/", "gpt-", "o1-", "o3-"))


@dataclass
class StreamChunk:
    """
    Structured streaming response payload.

    Attributes:
        token: Latest text token emitted.
        response: Accumulated response text so far.
        usage: Usage and timing info, populated when available.
        reasoning_token: Latest reasoning token emitted.
        reasoning_content: Accumulated reasoning content so far.
    """

    token: str
    response: str
    usage: dict[str, object]
    reasoning_token: str
    reasoning_content: str


@dataclass
class CompletionResult:
    """
    Structured non-streaming completion response payload.

    Attributes:
        content: Model response text.
        usage: Usage and timing information.
        reasoning: Optional reasoning content.
    """

    content: str
    usage: dict[str, object]
    reasoning: str | None


# for safety drop unsupported params
litellm.drop_params = True


def _extract_completion_details(raw_details: Any) -> dict[str, int]:
    """Convert completion token details into a JSON-serializable dict."""
    if raw_details is None:
        return {}

    keys = [
        "text_tokens",
        "reasoning_tokens",
        "accepted_prediction_tokens",
        "rejected_prediction_tokens",
        "audio_tokens",
    ]
    completion_details: dict[str, int] = {}
    for key in keys:
        if isinstance(raw_details, dict):
            value = raw_details.get(key)
        else:
            value = getattr(raw_details, key, None)

        if value is not None:
            completion_details[key] = int(value)
    return completion_details


def _build_usage_info(model: str, usage: Any, cost: float, llm_time: float) -> dict[str, object]:
    """Normalize usage payload with reasoning token details."""
    prompt_tokens = int((getattr(usage, "prompt_tokens", None) if usage else None) or 0)
    completion_tokens = int((getattr(usage, "completion_tokens", None) if usage else None) or 0)
    total_tokens = int((getattr(usage, "total_tokens", None) if usage else None) or 0)

    completion_details_raw = getattr(usage, "completion_tokens_details", None) if usage else None
    completion_details = _extract_completion_details(completion_details_raw)

    reasoning_tokens = int(completion_details.get("reasoning_tokens", 0) or 0)
    text_tokens = completion_details.get("text_tokens")

    if text_tokens is not None:
        answer_tokens = int(text_tokens or 0)
    elif reasoning_tokens:
        answer_tokens = max(completion_tokens - reasoning_tokens, 0)
    else:
        answer_tokens = completion_tokens

    usage_info: dict[str, object] = {
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": reasoning_tokens,
        "answer_tokens": answer_tokens,
        "cost": cost,
        "llm_time": llm_time,
    }

    if completion_details:
        usage_info["completion_tokens_details"] = completion_details

    return usage_info


def check_and_register_models(models: list[str], console: Console, headless: bool = False) -> None:
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


def get_completion_with_retry(
    model: str,
    messages: list[dict],
    temperature: float | None,
    console: Console,
    max_retries: int = 5,
    reasoning: dict | None = None,
    provider: str | None = None,
) -> CompletionResult:
    """
    Makes a non-streaming LLM completion request with automatic retry on rate limit errors.

    Args:
        model (str): The name of the model to use for completion.
        messages (list[dict]): List of message dictionaries for the conversation.
        temperature (float | None): Temperature parameter. If None, temperature is skipped.
        console (Console): Rich console instance for logging.
        max_retries (int, optional): Maximum number of retry attempts. Defaults to 5.
        reasoning (dict | None, optional): Reasoning parameters. Defaults to None.
        provider (str | None, optional): Provider to use. Defaults to None.

    Returns:
        CompletionResult: Structured response containing content, usage, and reasoning (if any).

    Raises:
        Exception: If max retries are exceeded and still getting rate limited.
    """
    retries = 0

    while retries < max_retries:
        try:
            params = LiteLLMParams(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=False,
                reasoning=reasoning,
                provider=provider,
            )
            completion_args = params.to_kwargs()

            request_start = time.time()
            response = completion(**completion_args)  # type: ignore
            request_end = time.time()
            llm_time = request_end - request_start

            response_content = response["choices"][0]["message"]["content"]  # type: ignore

            try:
                # calculates cost for models - only fails if model not registered and user skipped manual registration
                cost = completion_cost(response)
            except Exception:
                cost = 0.0

            # Extract usage information
            usage_info = _build_usage_info(
                model=model,
                usage=response.usage if hasattr(response, "usage") else None,
                cost=cost,
                llm_time=llm_time,
            )

            # extract reasoning content if available
            reasoning_content = None
            message = response["choices"][0]["message"]  # type: ignore
            if hasattr(message, "reasoning_content") and message.reasoning_content:
                reasoning_content = message.reasoning_content

            return CompletionResult(
                content=response_content,
                usage=usage_info,
                reasoning=reasoning_content,
            )

        except RateLimitError:
            retries += 1
            wait_time = min(2 * retries, 30)
            console.log(
                f"[bold yellow]Rate limited. Waiting for {wait_time} seconds...[/bold yellow]"
            )
            time.sleep(wait_time)
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")
            empty_usage = {
                "model": model,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost": 0.0,
                "llm_time": 0.0,
            }
            return CompletionResult(
                content="Review model failed to generate response.",
                usage=empty_usage,
                reasoning=None,
            )

    raise Exception("[bold red]Max retries exceeded.[/bold red]")


def get_streaming_completion_with_retry(
    model: str,
    messages: list[dict],
    temperature: float | None,
    console: Console,
    max_retries: int = 5,
    reasoning: dict | None = None,
    provider: str | None = None,
) -> Generator[StreamChunk, None, None]:
    """
    Makes a streaming LLM completion request with automatic retry on rate limit errors.

    Args:
        model (str): The name of the model to use for completion.
        messages (list[dict]): List of message dictionaries for the conversation.
        temperature (float | None): Temperature parameter. If None, temperature is skipped.
        console (Console): Rich console instance for logging.
        max_retries (int, optional): Maximum number of retry attempts. Defaults to 5.
        reasoning (dict, optional): Reasoning parameters. Defaults to None.
        provider (str, optional): Provider to use. Defaults to None.

    Yields:
        StreamChunk: Structured streaming payload containing the latest token,
            accumulated response, usage data, current reasoning token, and full
            reasoning content seen so far.

    Raises:
        Exception: If max retries are exceeded and still getting rate limited.
    """
    retries = 0

    while retries < max_retries:
        try:
            params = LiteLLMParams(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=True,
                reasoning=reasoning,
                provider=provider,
            )
            completion_args = params.to_kwargs()
            completion_args["stream_options"] = {"include_usage": True}

            stream_start = time.time()
            response = completion(**completion_args)  # type: ignore
            full_response = ""
            full_reasoning = ""
            final_usage = {
                "model": model,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "reasoning_tokens": 0,
                "answer_tokens": 0,
                "cost": 0.0,
                "llm_time": 0.0,
            }

            for chunk in response:
                try:
                    delta = chunk.choices[0].delta  # type: ignore
                    token = getattr(delta, "content", None) or ""
                    reasoning_token = getattr(delta, "reasoning_content", None) or ""

                    if token:
                        full_response += token
                    if reasoning_token:
                        full_reasoning += reasoning_token

                    if hasattr(chunk, "usage") and chunk.usage:  # type: ignore
                        try:
                            cost = completion_cost(chunk)
                        except Exception:
                            cost = 0.0

                        stream_end = time.time()
                        final_usage = _build_usage_info(
                            model=model,
                            usage=chunk.usage,  # type: ignore
                            cost=cost,
                            llm_time=stream_end - stream_start,
                        )

                    yield StreamChunk(
                        token=token,
                        response=full_response,
                        usage=final_usage,
                        reasoning_token=reasoning_token,
                        reasoning_content=full_reasoning,
                    )
                except Exception as e:
                    console.print(f"[bold red]Error processing stream chunk: {e}[/bold red]")
                    raise e

            yield StreamChunk(
                token="",
                response=full_response,
                usage=final_usage,
                reasoning_token="",
                reasoning_content=full_reasoning,
            )
            return
        except RateLimitError:
            retries += 1
            wait_time = min(2 * retries, 30)
            console.log(
                f"[bold yellow]Rate limited. Waiting for {wait_time} seconds...[/bold yellow]"
            )
            time.sleep(wait_time)

    raise Exception("[bold red]Max retries exceeded.[/bold red]")
