"""CLI entry points for manim-generator."""

from manim_generator.utils.video import render_and_concat


def manual_render():
    """Manual rendering entry point."""
    render_and_concat("video.py", "output_manual", "output.mp4")


if __name__ == "__main__":
    manual_render()
