import ast
import re


class SceneParsingError(Exception):
    """Exception raised when scene class names cannot be extracted from code."""

    pass


def parse_code_block(text: str) -> str:
    """
    Returns the first Python code block from text (None if none found).
    Handles optional 'python' specifier and trims whitespace.
    """

    match = re.search(
        r"```(?:python)?\s*\n(.*?)```",  # Matches optional 'python' and code
        text,
        re.DOTALL,  # Allows . to match newlines
    )
    return match.group(1).strip() if match else text


def extract_scene_class_names(code: str) -> list[str] | SceneParsingError:
    """Extract Scene class names from Manim code.

    Args:
        code: Python source code containing Manim scene definitions.

    Returns:
        A list of scene class names, or a SceneParsingError if parsing fails.
        Note: Returns the error as a value rather than raising to allow callers
        to handle parsing failures gracefully during the generation workflow.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return SceneParsingError(f"Syntax error in code: {e}")

    scene_names: list[str] = []
    try:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    # get scenes that inherit from 'Scene'
                    base_id = base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
                    if base_id.endswith("Scene"):
                        scene_names.append(node.name)
                        break
    except Exception as e:
        return SceneParsingError(f"Error extracting scene names: {e}")
    return scene_names
