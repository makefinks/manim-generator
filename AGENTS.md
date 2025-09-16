# AGENTS.md — Agent playbook for this repo

## Setup & tooling

- Python 3.10+ with [uv](https://github.com/astral-sh/uv) preferred: `uv sync && source .venv/bin/activate`; fall back to `pip install -r requirements.txt`.
- System deps: [Manim Community Edition](https://www.manim.community/), `ffmpeg`, and a LaTeX distribution if the animation needs it. Verify `manim` and `ffmpeg` binaries resolve before running the workflow.
- API keys for LiteLLM/OpenRouter go in a `.env` file or your environment. Never commit secrets.
- Use `uv run <cmd>` while developing to ensure the virtualenv and dependency versions match the lockfile.

## Running the generator

- Main entry point: `python generate.py` (invokes `src.main.main`). Provide prompts via `--video_data "..."` or `--video_data_file path/to/file` (defaults to `video_data.txt`).
- Common flags (see `src/utils/config.py`):
  - `--manim_model` / `--review_model` to swap models, `--review_cycles` to control iteration count.
  - `--force_vision` to attach rendered frames even if LiteLLM claims the model lacks vision; requires both models and the API provider to handle images.
  - Reasoning knobs: `--reasoning_effort` (OpenAI-style), `--reasoning_max_tokens` (Anthropic-style), `--reasoning_exclude` to hide reasoning traces from the response.
  - Rendering helpers: `--manim_logs`, `--frame_extraction_mode {fixed_count,highest_density}`, `--frame_count N`, and `--success_threshold` (percentage of scenes that must succeed to unlock the enhanced visual review prompt).
  - `--output_dir` overrides the auto-generated folder (`<llm_name>_YYYYMMDD_HHMMSS`). Each run writes artifacts to `<output_dir>/steps/*`.
- The workflow auto-checks LiteLLM’s cost table. If a model is unknown you’ll be prompted for price data in the console.

## Manual rendering & assets

- After a successful run, the accepted script is saved as `<output_dir>/video.py`. Use `python manual_render.py` to re-render at high quality and concatenate scenes (`manual_render.py` uses `src.utils.video.render_and_concat`).
- Intermediate frames, prompts, reviews, and reasoning are persisted under `<output_dir>/steps/<phase>/` for debugging. Frames live in a `frames/` subfolder; logs are collected in `logs.txt`.
- Generated usage metrics (`token_usage_report.json`) and workflow metadata (`workflow_metadata.json`) are saved in the same `output_dir` via helpers in `src/utils/usage.py`.

## Repository map

- `generate.py`: thin wrapper; hands off to `src/main.py`.
- `src/main.py`: CLI orchestration—parses args, spins up `ManimWorkflow`, prints summaries, and writes reports.
- `src/workflow.py`: core agent loop (initial generation, review cycles, artifact capture, finalization).
- `src/utils/`: supporting modules (configuration, parsing/code extraction, LLM wrappers, rendering and video helpers).
- `src/console.py`: Rich console helpers, including streaming output support.
- `src/artifacts.py`: handles per-step prompt/code/review/frames persistence.
- `prompts/`: text templates loaded via `src.utils.prompt.format_prompt`.

## Coding standards

- Follow PEP 8, 4-space indentation, ~88-100 char line length; keep imports grouped (stdlib → third-party → local) with one per line.
- Python typing uses built-in generics (`list`, `dict`, etc.) and `|` for unions. Annotate public functions and keep helpers small and purposeful.
- Prefer pathlib-style APIs for filesystem work; avoid hard-coded paths. Validate inputs before writing/reading files.
- Logging/console output should use Rich helpers (`src.console`) or module-level `logging` instances; favor structured, concise messages. Avoid stray `print` calls in workflow code.
- Handle errors explicitly (catch specific exceptions, no bare `except`). Surface issues via the Rich console or module-local loggers as appropriate.
- Google-style docstrings for modules, public classes, and functions that are part of the workflow.

## Testing & debugging

- No formal automated test suite yet. Create focused scripts under `tests/` or run modules directly with `uv run python -m <module>`.
- Use `manim` CLI directly for reproduction: `uv run manim -ql path/to/script.py SceneName`.
- When debugging rendering, inspect the per-scene logs in `<output_dir>/steps/<phase>/logs.txt` and the extracted frames under `frames/`.
- Watch for stale videos in Manim’s cache; the workflow already cleans the low-quality output after frame extraction, but high-quality manual renders may require manual cleanup.

## Common tips

- Rate limits: retries with exponential backoff live in `src/utils/llm.py`. Expect console prompts when rate-limited or when registering unknown model costs.
- Enhanced review mode only triggers once the configured success threshold is met; until then reviews focus on fixing runtime errors.
- Ensure OpenCV (`cv2`) has access to the correct video backend—frame extraction silently fails if `ffmpeg`/codecs are missing.
- Keep `prompts/*.txt` synchronized with workflow expectations when adding new placeholders; `format_prompt` will raise if a template variable is missing.
- Before committing large artifacts, clean or git-ignore generated media folders; only source and textual assets should enter version control.
