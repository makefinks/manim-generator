"""Tests for the LLM utilities."""

import unittest
from unittest.mock import MagicMock, patch

from rich.console import Console

from manim_generator.utils.llm import (
    _build_litellm_args,
    check_and_register_models,
    get_completion_with_retry,
    get_streaming_completion_with_retry,
)


class TestBuildLiteLLMArgs(unittest.TestCase):
    """Test cases for _build_litellm_args function."""

    def test_basic_args(self):
        """Test building basic arguments without reasoning or provider."""
        model = "gpt-4"
        messages = [{"role": "user", "content": "Hello"}]
        temperature = 0.5
        stream = False

        args = _build_litellm_args(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=stream,
            reasoning=None,
            provider=None,
        )

        self.assertEqual(args["model"], model)
        self.assertEqual(args["messages"], messages)
        self.assertEqual(args["temperature"], temperature)
        self.assertEqual(args["stream"], stream)
        self.assertNotIn("reasoning", args)
        self.assertNotIn("provider", args)

    def test_with_reasoning_openai(self):
        """Test building arguments with OpenAI reasoning."""
        reasoning = {"effort": "high"}
        args = _build_litellm_args(
            model="openai/gpt-4",
            messages=[],
            temperature=0.5,
            stream=False,
            reasoning=reasoning,
            provider=None,
        )

        self.assertIn("reasoning_effort", args)
        self.assertEqual(args["reasoning_effort"], "high")
        self.assertNotIn("reasoning", args)

    def test_with_reasoning_non_openai(self):
        """Test building arguments with non-OpenAI reasoning."""
        reasoning = {"effort": "medium"}
        args = _build_litellm_args(
            model="anthropic/claude-3-opus",
            messages=[],
            temperature=0.5,
            stream=False,
            reasoning=reasoning,
            provider=None,
        )

        self.assertIn("reasoning", args)
        self.assertEqual(args["reasoning"], reasoning)
        self.assertNotIn("reasoning_effort", args)

    def test_with_provider(self):
        """Test building arguments with provider."""
        args = _build_litellm_args(
            model="gpt-4",
            messages=[],
            temperature=0.5,
            stream=False,
            reasoning=None,
            provider="openrouter",
        )

        self.assertIn("provider", args)
        self.assertEqual(args["provider"], {"order": ["openrouter"]})


class TestCheckAndRegisterModels(unittest.TestCase):
    """Test cases for check_and_register_models function."""

    @patch("manim_generator.utils.llm.model_cost", {})
    @patch("manim_generator.utils.llm.Prompt.ask")
    @patch("manim_generator.utils.llm.register_model")
    def test_register_model_with_costs(self, mock_register, mock_ask):
        """Test registering a new model with costs."""
        console = Console()
        mock_ask.side_effect = ["0.50", "2.00"]

        check_and_register_models(["test-model"], console, headless=False)

        mock_register.assert_called_once()
        call_args = mock_register.call_args[0][0]
        self.assertIn("test-model", call_args)

    @patch("manim_generator.utils.llm.model_cost", {})
    def test_register_model_headless(self):
        """Test that headless mode skips registration."""
        console = Console()
        # Should not raise any errors or prompt
        check_and_register_models(["unknown-model"], console, headless=True)

    @patch("manim_generator.utils.llm.model_cost", {"gpt-4": {}})
    def test_skip_registered_model(self):
        """Test that already registered models are skipped."""
        console = Console()
        # Should not prompt since model is already registered
        check_and_register_models(["gpt-4"], console, headless=False)


class TestGetCompletionWithRetry(unittest.TestCase):
    """Test cases for get_completion_with_retry function."""

    @patch("manim_generator.utils.llm.completion_cost")
    @patch("manim_generator.utils.llm.completion")
    def test_successful_completion(self, mock_completion, mock_cost):
        """Test successful completion request."""
        # Create nested dict structure matching litellm response
        mock_response = MagicMock()
        mock_response.__getitem__ = MagicMock(
            side_effect=lambda key: {"choices": [{"message": {"content": "Test response"}}]}[key]
        )
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        mock_completion.return_value = mock_response
        mock_cost.return_value = 0.001

        console = Console()
        content, usage, reasoning = get_completion_with_retry(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test"}],
            temperature=0.5,
            console=console,
        )

        self.assertEqual(content, "Test response")
        self.assertEqual(usage["total_tokens"], 150)
        self.assertEqual(usage["cost"], 0.001)
        self.assertIsNone(reasoning)

    @patch("manim_generator.utils.llm.completion")
    def test_completion_with_reasoning(self, mock_completion):
        """Test completion with reasoning content."""
        # Create message object with reasoning_content attribute
        mock_message = MagicMock()
        mock_message.reasoning_content = "Test reasoning"

        # Create nested dict structure matching litellm response
        mock_response = MagicMock()
        mock_response.__getitem__ = MagicMock(
            side_effect=lambda key: {"choices": [{"message": mock_message}]}[key]
        )
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        mock_completion.return_value = mock_response

        console = Console()
        content, usage, reasoning = get_completion_with_retry(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test"}],
            temperature=0.5,
            console=console,
        )

        self.assertEqual(reasoning, "Test reasoning")


class TestGetStreamingCompletionWithRetry(unittest.TestCase):
    """Test cases for get_streaming_completion_with_retry function."""

    @patch("manim_generator.utils.llm.completion")
    def test_streaming_completion(self, mock_completion):
        """Test streaming completion request."""
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock(delta=MagicMock(content="Hello"))]
        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock(delta=MagicMock(content=" world"))]
        mock_completion.return_value = iter([mock_chunk1, mock_chunk2])

        console = Console()
        generator = get_streaming_completion_with_retry(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test"}],
            temperature=0.5,
            console=console,
        )

        chunks = list(generator)
        self.assertEqual(len(chunks), 3)  # 2 chunks + 1 final empty
        self.assertEqual(chunks[0][0], "Hello")
        self.assertEqual(chunks[0][1], "Hello")
        self.assertEqual(chunks[1][0], " world")
        self.assertEqual(chunks[1][1], "Hello world")


if __name__ == "__main__":
    unittest.main()
