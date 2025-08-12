# AGENTS.md — Agent playbook for this repo

- Build/Run: `python manim_generator.py`; manual render: `python manual_render.py <script.py>`
- Install deps: `uv sync` (preferred) or `pip install -r requirements.txt`
- Tests: no formal suite yet; add small scripts and run directly
- Lint/format: follow PEP 8; 4 spaces; line length ~88-100; isort-style import order
- Imports: stdlib first, then third‑party, then local; one per line
- Types: Python 3.10+; use built‑in generics (list, dict, tuple); use `|` for unions; annotate public funcs
- Naming: snake_case for vars/functions; PascalCase for classes; UPPER_CASE for constants
- Errors: catch specific exceptions; avoid bare except; log via `utils.console` where applicable
- Docs: Google‑style docstrings on modules, public classes/functions; explain params/returns/raises
- Structure: main entry `manim_generator.py`; workflow `manim_workflow.py`; helpers in `utils/`; prompts in `prompts/`
- Coding principles: small functions, clear names, minimal side effects, avoid deep nesting
- I/O and paths: use pathlib; avoid hardcoded paths; validate inputs
- CLI: extend `config.py` for new flags; keep defaults safe
- Logging: prefer structured, concise messages; include context; avoid noisy prints
- Performance: be mindful of render time; cache or reuse artifacts when safe
- Security: don’t commit secrets; config via env vars or .env (git‑ignored)
- Tooling integrations: No Cursor/Copilot rules present in repo at this time
