# Guidelines for Agentic Coding Agents

## Build/Lint/Test Commands
- **Build/Run**: `python manim_generator.py`
- **Install Dependencies**: `uv sync` or `pip install -r requirements.txt`
- **Manual Render**: `python manual_render.py <path_to_script.py>`
- **Testing**: Project currently has no formal test suite

## Code Style Guidelines
- **Runtime**: Python 3.10+
- **Imports**: Standard library imports first, then third-party, then local imports; one import per line
- **Types**: Use modern type hints for function parameters and return values. Use built-in generics (list, tuple, etc.) and | for unions. Avoid typing.Optional, Tuple, List, etc.
- **Naming Conventions**: snake_case for variables and functions; PascalCase for classes; UPPER_CASE for constants
- **Error Handling**: Use try/except blocks with specific exception types; log errors appropriately
- **Formatting**: Follow PEP 8 style guide with 4-space indentation
- **Documentation**: Use Google-style docstrings for functions and classes
- **General Principles**: Keep functions focused and concise; avoid deeply nested code; use meaningful variable names

## Project Structure
- **Main entry point**: `manim_generator.py`
- **Core logic**: `manim_workflow.py`
- **Utilities**: `utils/` directory for helper functions
- **Prompts**: `prompts/` directory for LLM prompts
- **Configuration**: `config.py` for CLI argument parsing and configuration

## Dependencies
- **manim**: For animation generation
- **litellm**: For LLM abstraction
- **rich**: For enhanced console output
- **opencv-python**: For image processing
