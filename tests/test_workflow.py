"""Tests for the main workflow."""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from rich.console import Console

from manim_generator.artifacts import ArtifactManager
from manim_generator.utils.usage import TokenUsageTracker
from manim_generator.workflow import ManimWorkflow


class TestManimWorkflow(unittest.TestCase):
    """Test cases for ManimWorkflow class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.console = Console()
        self.config = {
            "manim_model": "gpt-4",
            "review_model": "gpt-4",
            "temperature": 0.4,
            "review_cycles": 3,
            "output_dir": self.temp_dir,
            "streaming": False,
            "manim_logs": False,
            "force_vision": False,
            "frame_extraction_mode": "fixed_count",
            "frame_count": 3,
            "success_threshold": 80.0,
            "reasoning": None,
            "provider": None,
            "headless": False,
        }

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch("manim_generator.workflow.check_and_register_models")
    def test_workflow_initialization(self, mock_check):
        """Test that ManimWorkflow initializes correctly."""
        workflow = ManimWorkflow(
            config=self.config,
            console=self.console,
        )

        self.assertIsInstance(workflow.usage_tracker, TokenUsageTracker)
        self.assertIsInstance(workflow.artifact_manager, ArtifactManager)
        self.assertEqual(workflow.cycles_completed, 0)
        self.assertEqual(workflow.execution_count, 0)
        self.assertFalse(workflow.initial_success)

    @patch("manim_generator.workflow.check_and_register_models")
    @patch("manim_generator.workflow.get_response_with_status")
    def test_generate_initial_code(self, mock_get_response, mock_check):
        """Test initial code generation."""
        mock_get_response.return_value = (
            "```python\nfrom manim import *\n\nclass TestScene(Scene):\n    def construct(self):\n        pass\n```",
            {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "cost": 0.001},
            None,
        )

        workflow = ManimWorkflow(
            config=self.config,
            console=self.console,
        )

        code, conversation = workflow.generate_initial_code("Test video prompt")

        self.assertIn("from manim import *", code)
        self.assertIn("class TestScene", code)
        self.assertIsInstance(conversation, list)
        mock_get_response.assert_called_once()


if __name__ == "__main__":
    unittest.main()
