# Repository Guidelines

## Project Structure & Module Organization
- Core package lives in `src/manim_generator/`; `main.py` orchestrates the workflow that drafts, reviews, and executes Manim scenes, while `utils/` holds config, prompt, LLM, file, rendering, and usage helpers.
- CLI entry points are exposed via `manim-generate` (main flow) and `manim-render` (re-render an existing script from a prior run); both resolve to modules in `src/manim_generator`.
- Tests are under `tests/` with unit coverage for parsing, prompts, rendering, CLI integration, and usage tracking. Benchmark prompts sit in `bench_prompts/`; reusable prompt scaffolding is in `prompts/`. Generated media and run artifacts default to timestamped folders under `output/`.
- Reference docs, images, and diagrams are in `docs/` and `images/`; keep large assets out of Git and prefer linking.

## Build, Test, and Development Commands
- Install deps: `uv sync` (recommended) or `pip install -e .`; ensure `ffmpeg` (and LaTeX if you render formulas) is available on PATH.
- Run the workflow: `uv run manim-generate --video-data "Explain transformers"` or `uv run manim-generate --video-data-file bench_prompts/llm_explainer.txt`.
- Re-render a prior run: `uv run manim-render --run-dir output/manim_animation_YYYYMMDD_HHMMSS` (auto-detects latest run if not provided).
- Tests: `uv run pytest` (uses `pytest.ini` + `pyproject.toml` options). 
- Lint: `uv run ruff check src tests` (line length 100 target, `E501` ignored but stay near 100). Import sorting is enforced via Ruff’s `I` rule.

## Coding Style & Naming Conventions
- Python 3.10+, prefer type hints and explicit returns. Use snake_case for modules/functions, PascalCase for classes, SCREAMING_SNAKE_CASE for constants. Keep CLI flags aligned with existing argparse patterns.
- Favor small, pure helpers in `utils/` and keep CLI-only concerns in the console modules. Rich is used for user-facing output—preserve consistent formatting and color usage.
- Place reusable prompt text in `prompts/` and bench inputs in `bench_prompts/`; avoid hardcoding secret values anywhere.

## Testing Guidelines
- New logic should be covered by `pytest` cases in `tests/` mirroring the module path (e.g., `tests/test_usage.py` for `utils/usage.py`). Name tests `test_<behavior>`; use fixtures instead of global state.
- When touching CLI flow, add or update integration coverage in `tests/test_cli_integration.py`. For rendering helpers, prefer small synthetic scenes over heavyweight assets to keep runtime low.
- Keep generated artifacts out of Git; tests should write to temp dirs and clean up.
