"""Tests for the artifacts module."""

import os
import shutil
import tempfile
import unittest

from rich.console import Console

from manim_generator.artifacts import ArtifactManager


class TestArtifactManager(unittest.TestCase):
    """Test cases for ArtifactManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.console = Console()
        self.artifact_manager = ArtifactManager(self.temp_dir, self.console)

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test that artifact manager initializes correctly."""
        self.assertEqual(self.artifact_manager.output_dir, self.temp_dir)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "steps")))

    def test_get_step_frames_path(self):
        """Test getting frames directory for a step."""
        frames_dir = self.artifact_manager.get_step_frames_path("initial")
        expected = os.path.join(self.temp_dir, "steps", "initial", "frames")
        self.assertEqual(frames_dir, expected)
        self.assertTrue(os.path.exists(frames_dir))

    def test_save_step_artifacts_with_code(self):
        """Test saving code artifact."""
        step_name = "test_step"
        code = "print('Hello, world!')"

        self.artifact_manager.save_step_artifacts(step_name, code=code)

        code_file = os.path.join(self.temp_dir, "steps", step_name, "code.py")
        self.assertTrue(os.path.exists(code_file))

        with open(code_file) as f:
            content = f.read()
        self.assertEqual(content, code)

    def test_save_step_artifacts_with_prompt(self):
        """Test saving prompt artifact."""
        step_name = "test_step"
        prompt = "Test prompt content"

        self.artifact_manager.save_step_artifacts(step_name, prompt=prompt)

        prompt_file = os.path.join(self.temp_dir, "steps", step_name, "prompt.txt")
        self.assertTrue(os.path.exists(prompt_file))

        with open(prompt_file) as f:
            content = f.read()
        self.assertEqual(content, prompt)

    def test_save_step_artifacts_with_logs(self):
        """Test saving logs artifact."""
        step_name = "test_step"
        logs = "Test log content"

        self.artifact_manager.save_step_artifacts(step_name, logs=logs)

        logs_file = os.path.join(self.temp_dir, "steps", step_name, "logs.txt")
        self.assertTrue(os.path.exists(logs_file))

        with open(logs_file) as f:
            content = f.read()
        self.assertEqual(content, logs)

    def test_save_step_artifacts_with_review(self):
        """Test saving review artifact."""
        step_name = "test_step"
        review = "# Test Review\n\nReview content"

        self.artifact_manager.save_step_artifacts(step_name, review_text=review)

        review_file = os.path.join(self.temp_dir, "steps", step_name, "review.md")
        self.assertTrue(os.path.exists(review_file))

        with open(review_file) as f:
            content = f.read()
        self.assertEqual(content, review)

    def test_save_step_artifacts_with_reasoning(self):
        """Test saving reasoning artifact."""
        step_name = "test_step"
        reasoning = "Test reasoning content"

        self.artifact_manager.save_step_artifacts(step_name, reasoning=reasoning)

        reasoning_file = os.path.join(self.temp_dir, "steps", step_name, "reasoning.txt")
        self.assertTrue(os.path.exists(reasoning_file))

        with open(reasoning_file) as f:
            content = f.read()
        self.assertEqual(content, reasoning)

    def test_save_step_artifacts_with_all(self):
        """Test saving all artifacts at once."""
        step_name = "test_step"
        code = "print('code')"
        prompt = "prompt"
        logs = "logs"
        review = "review"
        reasoning = "reasoning"

        self.artifact_manager.save_step_artifacts(
            step_name,
            code=code,
            prompt=prompt,
            logs=logs,
            review_text=review,
            reasoning=reasoning,
        )

        step_dir = os.path.join(self.temp_dir, "steps", step_name)
        self.assertTrue(os.path.exists(os.path.join(step_dir, "code.py")))
        self.assertTrue(os.path.exists(os.path.join(step_dir, "prompt.txt")))
        self.assertTrue(os.path.exists(os.path.join(step_dir, "logs.txt")))
        self.assertTrue(os.path.exists(os.path.join(step_dir, "review.md")))
        self.assertTrue(os.path.exists(os.path.join(step_dir, "reasoning.txt")))


if __name__ == "__main__":
    unittest.main()
