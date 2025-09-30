"""Tests for the prompt utilities."""

import os
import tempfile
import unittest

from manim_generator.utils.prompt import (
    convert_frames_to_message_format,
    format_previous_reviews,
    format_prompt,
)


class TestFormatPrompt(unittest.TestCase):
    """Test cases for format_prompt function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.prompts_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts"
        )

    def test_format_prompt_with_replacements(self):
        """Test formatting prompt with placeholder replacements."""
        test_prompt = "Hello {name}, your age is {age}"
        test_file = os.path.join(self.prompts_dir, "test_prompt.txt")

        if os.path.exists(self.prompts_dir):
            with open(test_file, "w") as f:
                f.write(test_prompt)

            try:
                result = format_prompt("test_prompt", {"name": "Alice", "age": "30"})
                self.assertEqual(result, "Hello Alice, your age is 30")
            finally:
                os.remove(test_file)

    def test_format_prompt_no_replacements(self):
        """Test formatting prompt without placeholders."""
        test_prompt = "Simple prompt with no placeholders"
        test_file = os.path.join(self.prompts_dir, "test_simple.txt")

        if os.path.exists(self.prompts_dir):
            with open(test_file, "w") as f:
                f.write(test_prompt)

            try:
                result = format_prompt("test_simple", {})
                self.assertEqual(result, test_prompt)
            finally:
                os.remove(test_file)


class TestFormatPreviousReviews(unittest.TestCase):
    """Test cases for format_previous_reviews function."""

    def test_format_single_review(self):
        """Test formatting a single review."""
        reviews = ["First review feedback"]
        result = format_previous_reviews(reviews)
        expected = "<review_0>\nFirst review feedback\n</review_0>"
        self.assertEqual(result, expected)

    def test_format_multiple_reviews(self):
        """Test formatting multiple reviews."""
        reviews = ["First review", "Second review", "Third review"]
        result = format_previous_reviews(reviews)
        expected = (
            "<review_0>\nFirst review\n</review_0>\n"
            "<review_1>\nSecond review\n</review_1>\n"
            "<review_2>\nThird review\n</review_2>"
        )
        self.assertEqual(result, expected)

    def test_format_empty_reviews(self):
        """Test formatting empty review list."""
        reviews = []
        result = format_previous_reviews(reviews)
        self.assertEqual(result, "")

    def test_format_review_with_special_chars(self):
        """Test formatting review with special characters."""
        reviews = ["Review with <tags> and & symbols"]
        result = format_previous_reviews(reviews)
        expected = "<review_0>\nReview with <tags> and & symbols\n</review_0>"
        self.assertEqual(result, expected)


class TestConvertFramesToMessageFormat(unittest.TestCase):
    """Test cases for convert_frames_to_message_format function."""

    def test_convert_single_frame(self):
        """Test converting single frame to message format."""
        frames = ["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA"]
        result = convert_frames_to_message_format(frames)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "image_url")
        self.assertEqual(result[0]["image_url"]["url"], frames[0])
        self.assertEqual(result[0]["image_url"]["format"], "image/png")

    def test_convert_multiple_frames(self):
        """Test converting multiple frames to message format."""
        frames = [
            "data:image/png;base64,frame1",
            "data:image/png;base64,frame2",
            "data:image/png;base64,frame3",
        ]
        result = convert_frames_to_message_format(frames)

        self.assertEqual(len(result), 3)
        for i, frame_msg in enumerate(result):
            self.assertEqual(frame_msg["type"], "image_url")
            self.assertEqual(frame_msg["image_url"]["url"], frames[i])
            self.assertEqual(frame_msg["image_url"]["format"], "image/png")

    def test_convert_empty_frames(self):
        """Test converting empty frame list."""
        frames = []
        result = convert_frames_to_message_format(frames)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
