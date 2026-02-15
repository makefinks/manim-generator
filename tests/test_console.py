"""Tests for console response helpers."""

import io
import unittest
from unittest.mock import patch

from rich.console import Console

from manim_generator.console import get_response_with_status, print_request_summary
from manim_generator.utils.llm import CompletionResult


class TestGetResponseWithStatus(unittest.TestCase):
    """Test cases for get_response_with_status."""

    @patch("manim_generator.console.get_completion_with_retry")
    def test_get_response_with_status_does_not_print_summary(self, mock_completion):
        """Response retrieval should not print summary directly."""
        usage_info = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "reasoning_tokens": 5,
            "answer_tokens": 15,
            "total_tokens": 30,
            "cost": 0.0042,
            "llm_time": 0.1,
        }
        mock_completion.return_value = CompletionResult(
            content="hello",
            usage=usage_info,
            reasoning=None,
        )

        output_buffer = io.StringIO()
        console = Console(file=output_buffer, force_terminal=False, color_system=None)

        response_text, returned_usage, reasoning = get_response_with_status(
            model="openrouter/meta-llama/llama-3.3-70b-instruct",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.7,
            streaming=False,
            status=None,
            console=console,
            headless=True,
        )

        output = output_buffer.getvalue()
        self.assertEqual(response_text, "hello")
        self.assertEqual(returned_usage, usage_info)
        self.assertIsNone(reasoning)
        self.assertNotIn("Cost: $", output)

    def test_print_request_summary_outputs_cost_and_tokens(self):
        """Summary helper should print token/cost details."""
        usage_info = {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "reasoning_tokens": 0,
            "answer_tokens": 2,
            "total_tokens": 3,
            "cost": 0.0003,
            "llm_time": 0.1,
        }
        output_buffer = io.StringIO()
        console = Console(file=output_buffer, force_terminal=False, color_system=None)

        print_request_summary(
            console=console, usage_info=usage_info, headless=True, elapsed_time=0.2
        )

        output = output_buffer.getvalue()
        self.assertIn("Cost: $0.000300", output)
        self.assertIn("Input Tokens: 1", output)
        self.assertIn("Output Tokens: 2", output)


if __name__ == "__main__":
    unittest.main()
