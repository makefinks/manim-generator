#!/usr/bin/env python3
"""Test runner for the manim-generator project."""

import os
import sys
import unittest

from rich.console import Console
from rich.panel import Panel


def run_tests():
    """Discover and run all tests."""
    console = Console()

    # Ensure we're running from the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    # Add project root to Python path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Display results summary
    console.print(
        Panel(
            f"[bold green]Tests run: {result.testsRun}[/bold green]\n"
            f"[bold red]Failures: {len(result.failures)}[/bold red]\n"
            f"[bold yellow]Errors: {len(result.errors)}[/bold yellow]",
            title="[bold]Test Results Summary",
        )
    )

    # Return exit code based on test results
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
