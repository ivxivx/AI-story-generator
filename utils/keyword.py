import re

SEPARATOR = "-"


def sanitize_keyword(keyword: str) -> str:
    """
    Sanitizes the input keyword by converting it to lowercase, replacing spaces with hyphens,
    and removing non-alphabetic characters.

    Parameters:
    keyword (str): The keyword to be sanitized.

    Returns:
    str: The sanitized keyword.
    """

    lowered = keyword.lower()
    substituted = re.sub(r"\s+", SEPARATOR, lowered)
    filtered = [c for c in substituted if (c.isalpha() or c == SEPARATOR)]

    result = "".join(filtered)

    return result
