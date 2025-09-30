"""Tests for the rendering utilities."""

import unittest
import tempfile
import os
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from manim_generator.utils.rendering import (
    calculate_scene_success_rate,
    extract_frames_from_video,
)


class TestCalculateSceneSuccessRate(unittest.TestCase):
    """Test cases for calculate_scene_success_rate function."""

    def test_all_scenes_successful_fixed_count(self):
        """Test success rate when all scenes rendered in fixed_count mode."""
        frames = ["frame1", "frame2", "frame3", "frame4", "frame5", "frame6"]
        scene_names = ["Scene1", "Scene2"]
        frames_per_scene = 3

        success_rate, rendered, total = calculate_scene_success_rate(
            frames, scene_names, frames_per_scene, "fixed_count"
        )

        self.assertEqual(success_rate, 100.0)
        self.assertEqual(rendered, 2)
        self.assertEqual(total, 2)

    def test_partial_scenes_successful_fixed_count(self):
        """Test success rate when only some scenes rendered in fixed_count mode."""
        frames = ["frame1", "frame2", "frame3"]
        scene_names = ["Scene1", "Scene2"]
        frames_per_scene = 3

        success_rate, rendered, total = calculate_scene_success_rate(
            frames, scene_names, frames_per_scene, "fixed_count"
        )

        self.assertEqual(success_rate, 50.0)
        self.assertEqual(rendered, 1)
        self.assertEqual(total, 2)

    def test_highest_density_mode(self):
        """Test success rate in highest_density mode."""
        frames = ["frame1", "frame2"]
        scene_names = ["Scene1", "Scene2", "Scene3"]

        success_rate, rendered, total = calculate_scene_success_rate(
            frames, scene_names, 1, "highest_density"
        )

        self.assertAlmostEqual(success_rate, 66.666, places=1)
        self.assertEqual(rendered, 2)
        self.assertEqual(total, 3)

    def test_no_scenes(self):
        """Test success rate with no scenes."""
        frames = []
        scene_names = []

        success_rate, rendered, total = calculate_scene_success_rate(
            frames, scene_names, 1, "fixed_count"
        )

        self.assertEqual(success_rate, 0.0)
        self.assertEqual(rendered, 0)
        self.assertEqual(total, 0)

    def test_parsing_exception(self):
        """Test success rate when scene parsing failed."""
        frames = []
        scene_names = Exception("Syntax error")

        success_rate, rendered, total = calculate_scene_success_rate(
            frames, scene_names, 1, "fixed_count"
        )

        self.assertEqual(success_rate, 0.0)
        self.assertEqual(rendered, 0)
        self.assertEqual(total, 0)


class TestExtractFramesFromVideo(unittest.TestCase):
    """Test cases for extract_frames_from_video function."""

    @patch("manim_generator.utils.rendering.cv2.VideoCapture")
    def test_fixed_count_mode(self, mock_capture):
        """Test extracting fixed count of frames."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30
        mock_cap.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
        mock_capture.return_value = mock_cap

        frames = extract_frames_from_video(
            "test_video.mp4", mode="fixed_count", frame_count=3
        )

        self.assertIsNotNone(frames)
        if frames is not None:
            self.assertEqual(len(frames), 3)
        mock_cap.release.assert_called_once()

    @patch("manim_generator.utils.rendering.cv2.VideoCapture")
    def test_highest_density_mode(self, mock_capture):
        """Test extracting highest density frame."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 10

        # Create frames with different densities
        black_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        white_frame = np.ones((100, 100, 3), dtype=np.uint8) * 255

        mock_cap.read.side_effect = [(True, black_frame), (True, white_frame)]
        mock_capture.return_value = mock_cap

        frames = extract_frames_from_video(
            "test_video.mp4", mode="highest_density", max_frames=2
        )

        self.assertIsNotNone(frames)
        if frames is not None:
            self.assertEqual(len(frames), 1)
        mock_cap.release.assert_called_once()

    @patch("manim_generator.utils.rendering.cv2.VideoCapture")
    def test_video_not_opened(self, mock_capture):
        """Test handling when video cannot be opened."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_capture.return_value = mock_cap

        frames = extract_frames_from_video("nonexistent.mp4")

        self.assertIsNone(frames)

    @patch("manim_generator.utils.rendering.cv2.VideoCapture")
    def test_empty_video(self, mock_capture):
        """Test handling empty video with zero frames."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 0
        mock_capture.return_value = mock_cap

        frames = extract_frames_from_video("empty_video.mp4")

        self.assertIsNone(frames)

    @patch("manim_generator.utils.rendering.cv2.VideoCapture")
    def test_invalid_mode(self, mock_capture):
        """Test handling invalid extraction mode."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30
        mock_capture.return_value = mock_cap

        frames = extract_frames_from_video("test_video.mp4", mode="invalid_mode")

        self.assertIsNone(frames)
        mock_cap.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
