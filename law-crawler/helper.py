"""Utility functions for law-crawler."""
import re


def convert_roman_to_num(roman_num: str) -> int:
    """Convert Roman numeral or Vietnamese index to integer.

    Args:
        roman_num: Roman numeral (I, V, X, L, C, D, M) or
                   Vietnamese index (A, B, C, ...).

    Returns:
        Integer value.
    """
    roman_num = roman_num.upper()
    roman_to_num = {
        "I": 1, "V": 5, "X": 10, "L": 50,
        "C": 100, "D": 500, "M": 1000,
    }
    alphabet = list("ABCDEFGHIJ")
    num = 0
    for i, char in enumerate(roman_num):
        if char not in roman_to_num:
            num += alphabet.index(char) + 1
            continue
        if i > 0 and roman_to_num[char] > roman_to_num[roman_num[i - 1]]:
            num += roman_to_num[char] - 2 * roman_to_num[roman_num[i - 1]]
        else:
            num += roman_to_num[char]
    return num


def extract_input(input_string: str) -> str | None:
    """Extract content inside parentheses from a string.

    Args:
        input_string: String containing parenthesized content.

    Returns:
        Content inside first parentheses, or None if not found.
    """
    match = re.search(r"\((.*?)\)", input_string)
    return match.group(1) if match else None
