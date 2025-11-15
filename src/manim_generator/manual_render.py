"""CLI entry points for manim-generator."""

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from rich.console import Console

from manim_generator.utils.video import render_and_concat


def _discover_latest_output_dirs() -> Iterable[Path]:
    """Yield workflow output directories that contain a summary."""
    yield from Path(".").rglob("workflow_summary.json")


def _select_latest_output_dir(console: Console) -> Path | None:
    """Choose the most recent run directory using workflow summaries."""
    summaries = list(_discover_latest_output_dirs())
    if not summaries:
        return None

    latest_summary = max(summaries, key=lambda p: p.stat().st_mtime)
    try:
        data = json.loads(latest_summary.read_text(encoding="utf-8"))
        output_dir = data.get("input", {}).get("args", {}).get("output_dir")
        if output_dir:
            candidate = Path(output_dir)
            if not candidate.is_absolute():
                candidate = Path.cwd() / candidate
            if candidate.exists():
                console.print(
                    f"[cyan]Using workflow output directory from latest summary: {candidate}[/cyan]"
                )
                return candidate
    except Exception as exc:
        console.print(f"[yellow]Failed to parse {latest_summary}: {exc}[/yellow]")

    console.print(f"[cyan]Falling back to detected directory: {latest_summary.parent}[/cyan]")
    return latest_summary.parent


def manual_render():
    """Manual rendering entry point."""
    parser = argparse.ArgumentParser(description="Render a previously generated Manim script")
    parser.add_argument(
        "--run-dir",
        "-r",
        help="Path to the workflow output directory containing `video.py` and artifacts",
    )
    parser.add_argument(
        "--script-file",
        "-s",
        help="Path to a specific Manim script to render (skips discovery)",
    )
    parser.add_argument(
        "--media-dir",
        "-m",
        default="output_manual",
        help="Media directory passed to `manim --media_dir` during the manual render",
    )
    parser.add_argument(
        "--final-output",
        "-f",
        default="output.mp4",
        help="Filename for the concatenated manual render (stored inside --media-dir)",
    )
    args = parser.parse_args()

    console = Console()
    script_path = Path(args.script_file) if args.script_file else None

    if script_path is None:
        run_dir = Path(args.run_dir) if args.run_dir else _select_latest_output_dir(console)
        if not run_dir:
            console.print(
                "[bold red]No workflow output directory was found to re-render.[/bold red]"
            )
            return
        script_path = run_dir / "video.py"

    if not script_path.exists():
        console.print(f"[bold red]Could not find the script to render: {script_path}[/bold red]")
        return

    console.print(
        f"[bold green]Rendering script:[/bold green] {script_path} "
        f"[bold green]| media dir:[/bold green] {args.media_dir}"
    )
    render_and_concat(str(script_path), args.media_dir, args.final_output)


if __name__ == "__main__":
    manual_render()
