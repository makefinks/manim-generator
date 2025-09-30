"""Tests for the usage utilities."""

import unittest

from manim_generator.utils.usage import TokenUsageTracker, format_duration


class TestTokenUsageTracker(unittest.TestCase):
    """Test cases for TokenUsageTracker class."""

    def setUp(self):
        """Set up test fixtures."""
        self.tracker = TokenUsageTracker()

    def test_initialization(self):
        """Test that tracker initializes with correct default values."""
        self.assertEqual(self.tracker.token_usage_tracking["total_tokens"], 0)
        self.assertEqual(self.tracker.token_usage_tracking["total_cost"], 0.0)
        self.assertEqual(self.tracker.token_usage_tracking["steps"], [])

    def test_add_step(self):
        """Test adding a step to the tracker."""
        step_name = "Initial Code Generation"
        model = "gpt-4"
        usage_info = {
            "prompt_tokens": 100,
            "completion_tokens": 200,
            "total_tokens": 300,
            "cost": 0.0015,
        }

        self.tracker.add_step(step_name, model, usage_info)

        # Check that step was added correctly
        self.assertEqual(len(self.tracker.token_usage_tracking["steps"]), 1)
        step = self.tracker.token_usage_tracking["steps"][0]
        self.assertEqual(step["step"], step_name)
        self.assertEqual(step["model"], model)
        self.assertEqual(step["prompt_tokens"], 100)
        self.assertEqual(step["completion_tokens"], 200)
        self.assertEqual(step["total_tokens"], 300)
        self.assertEqual(step["cost"], 0.0015)

        # Check that totals were updated
        self.assertEqual(self.tracker.token_usage_tracking["total_tokens"], 300)
        self.assertEqual(self.tracker.token_usage_tracking["total_cost"], 0.0015)

    def test_get_tracking_data(self):
        """Test retrieving tracking data."""
        step_name = "Review Cycle 1"
        model = "gpt-4"
        usage_info = {
            "prompt_tokens": 150,
            "completion_tokens": 250,
            "total_tokens": 400,
            "cost": 0.0020,
        }

        self.tracker.add_step(step_name, model, usage_info)
        data = self.tracker.get_tracking_data()

        self.assertIsInstance(data, dict)
        self.assertIn("steps", data)
        self.assertIn("total_tokens", data)
        self.assertIn("total_cost", data)


class TestFormatDuration(unittest.TestCase):
    """Test cases for format_duration function."""

    def test_seconds_format(self):
        """Test formatting duration in seconds."""
        result = format_duration(30.5)
        self.assertEqual(result, "30.5 seconds")

    def test_minutes_format(self):
        """Test formatting duration in minutes and seconds."""
        result = format_duration(90.5)
        self.assertEqual(result, "1m 30.5s")

    def test_hours_format(self):
        """Test formatting duration in hours, minutes and seconds."""
        result = format_duration(3661.5)
        self.assertEqual(result, "1h 1m 1.5s")


if __name__ == "__main__":
    unittest.main()
