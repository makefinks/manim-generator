#!/usr/bin/env python3
"""Manual rendering script."""

from src.utils.video import render_and_concat

if __name__ == "__main__":
    render_and_concat("video.py", "output", "output.mp4")
