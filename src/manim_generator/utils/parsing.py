import ast
import re


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


def extract_scene_class_names(code: str) -> list[str] | Exception:
    try:
        tree = ast.parse(code)
    except Exception as e:
        return SyntaxError(f"Syntax error in code: {str(e)}")

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
        return e
    return scene_names
