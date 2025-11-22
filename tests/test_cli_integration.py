"""End-to-end tests that exercise the manim-generate CLI."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

PROMPT_TEXT = """\
You are running a regression test. Simply return the exact script below without\
changing class names, structure, or animations. The instructions already contain\
the finished solution, so copy it verbatim:

```python
from manim import *


class PromptDrivenScene(Scene):
    def construct(self):
        message = Text("Integration smoke test", font_size=48)
        self.play(Write(message))
        self.wait(0.25)
```
"""


@unittest.skipUnless(
    os.environ.get("OPENROUTER_API_KEY"),
    "OPENROUTER_API_KEY is required to run the CLI integration tests.",
)
class TestCLIIntegration(unittest.TestCase):
    """Runs the CLI in headless and interactive modes to ensure artifacts exist."""

    MODEL = "openrouter/meta-llama/llama-3.3-70b-instruct"
    PROVIDER = "cerebras/fp16"

    def setUp(self):
        self.project_root = Path(__file__).resolve().parents[1]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="manim_e2e_"))
        self.prompt_file = self.temp_dir / "prompt.txt"
        self.prompt_file.write_text(PROMPT_TEXT, encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cli_run_without_headless_flag(self):
        """Integration test for the default (interactive) workflow."""
        output_dir = self.temp_dir / "non_headless"
        self._run_generator(output_dir, headless=False)

    def test_cli_run_with_headless_flag(self):
        """Integration test for the headless workflow."""
        output_dir = self.temp_dir / "headless"
        self._run_generator(output_dir, headless=True)

    def _run_generator(self, output_dir: Path, *, headless: bool):
        """Execute the CLI and validate key artifacts."""
        command = [
            "uv",
            "run",
            "manim-generate",
            "--video-data-file",
            str(self.prompt_file),
            "--manim-model",
            self.MODEL,
            "--review-model",
            self.MODEL,
            "--review-cycles",
            "1",
            "--output-dir",
            str(output_dir),
            "--provider",
            self.PROVIDER,
        ]
        if headless:
            command.append("--headless")

        mode_label = "headless" if headless else "interactive"
        print(f"[test-cli] Starting {mode_label} workflow in {output_dir}", flush=True)

        # Interactive workflow prompts for confirmation, model costs, and final render decisions.
        input_data = None
        if not headless:
            input_data = "y\n0.50\n2.00\nn\nn\n"

        result = subprocess.run(
            command,
            cwd=self.project_root,
            text=True,
            capture_output=True,
            input=input_data,
            check=False,
        )
        if result.returncode != 0:
            self.fail(
                "manim-generate failed.\n"
                f"Exit code: {result.returncode}\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

        summary_path = output_dir / "workflow_summary.json"
        video_script = output_dir / "video.py"

        self.assertTrue(summary_path.exists(), f"workflow summary missing at {summary_path}")
        self.assertTrue(video_script.exists(), f"generated script missing at {video_script}")

        with summary_path.open(encoding="utf-8") as summary_file:
            summary_data = json.load(summary_file)

        models = summary_data.get("models", {})
        execution_stats = summary_data.get("execution_stats", {})

        self.assertEqual(models.get("manim_model"), self.MODEL)
        self.assertEqual(models.get("review_model"), self.MODEL)
        self.assertEqual(execution_stats.get("review_cycles_completed"), 1)
        self.assertIn("total_tokens", summary_data.get("usage", {}))
