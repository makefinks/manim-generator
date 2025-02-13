"""Text utility functions for prompt formatting and message handling."""

def format_prompt(prompt_name: str, replacements: dict) -> str:
    """
    Load a prompt template and replace placeholders with provided values.

    Args:
        prompt_name: Name of the prompt file (without .txt extension)
        replacements: Dictionary of {placeholder: value} replacements

    Returns:
        Formatted prompt string with all replacements applied
    """
    with open(f"prompts/{prompt_name}.txt", "r") as file:
        prompt_template = file.read()

    # Apply all replacements sequentially
    for placeholder, value in replacements.items():
        prompt_template = prompt_template.replace(f"{{{placeholder}}}", str(value))
    return prompt_template


def format_previous_reviews(previous_reviews: list[str]) -> str:
    """
    Format a list of review feedback strings into XML-style tagged format.

    Args:
        previous_reviews: List of review feedback strings to format

    Returns:
        String containing all reviews formatted with XML-style tags, with each review
        wrapped in numbered tags like <review_0>, <review_1> etc. and joined by newlines
    """

    xml_formatted = [
        f"<review_{idx}>\n{feedback}\n</review_{idx}"
        for idx, feedback in enumerate(previous_reviews)
    ]
    return "\n".join(xml_formatted)


def convert_frames_to_message_format(frames: str) -> list[str]:
    """
    Converts base64 encoded video frames into a list of message objects which litellm expects
    https://docs.litellm.ai/docs/completion/vision
    """
    return [{"type": "image_url", "image_url": { "url": frame }} for frame in frames]

