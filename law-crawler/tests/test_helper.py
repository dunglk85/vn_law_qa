"""Tests for law-crawler helper functions."""
import pytest


def test_convert_roman_to_num_basic():
    """Convert basic Roman numerals."""
    from src.helper import convert_roman_to_num

    assert convert_roman_to_num("I") == 1
    assert convert_roman_to_num("V") == 5
    assert convert_roman_to_num("X") == 10
    assert convert_roman_to_num("L") == 50
    assert convert_roman_to_num("D") == 500
    assert convert_roman_to_num("M") == 1000


def test_convert_roman_to_num_compound():
    """Convert compound Roman numerals."""
    from src.helper import convert_roman_to_num

    assert convert_roman_to_num("IV") == 4
    assert convert_roman_to_num("IX") == 9
    assert convert_roman_to_num("XL") == 40
    assert convert_roman_to_num("XC") == 90
    assert convert_roman_to_num("XLII") == 42


def test_convert_roman_to_num_vietnamese():
    """Convert Vietnamese index characters not in Roman numeral set."""
    from src.helper import convert_roman_to_num

    assert convert_roman_to_num("A") == 1
    assert convert_roman_to_num("B") == 2
    assert convert_roman_to_num("E") == 5


def test_convert_roman_to_num_case_insensitive():
    """Roman conversion should be case insensitive."""
    from src.helper import convert_roman_to_num

    assert convert_roman_to_num("iv") == 4
    assert convert_roman_to_num("ix") == 9


def test_extract_input_found():
    """Extract content inside parentheses."""
    from src.helper import extract_input

    assert extract_input("abc(123)def") == "123"
    assert extract_input("foo('bar')") == "'bar'"
    assert extract_input("onclick=\"select('MAPC_VALUE')\"") == "'MAPC_VALUE'"


def test_extract_input_not_found():
    """Return None when no parentheses found or parentheses are empty."""
    from src.helper import extract_input

    assert extract_input("no parentheses here") is None
    assert extract_input("") is None
    assert extract_input("()") is None


def test_extract_input_nested():
    """Extract from outermost parentheses with greedy match."""
    from src.helper import extract_input

    assert extract_input("(a(b)c)") == "a(b)c"


def test_convert_roman_to_num_unknown_char_raises():
    """I1 fix: characters outside both Roman and Vietnamese index sets must raise ValueError."""
    from src.helper import convert_roman_to_num

    with pytest.raises(ValueError, match="Unknown character"):
        convert_roman_to_num("K")

    with pytest.raises(ValueError, match="Unknown character"):
        convert_roman_to_num("Z")

    with pytest.raises(ValueError, match="Unknown character"):
        convert_roman_to_num("1")


def test_convert_roman_to_num_alphabet_boundary():
    """I1 fix: J (10th letter) is the last supported Vietnamese index."""
    from src.helper import convert_roman_to_num

    assert convert_roman_to_num("J") == 10
    with pytest.raises(ValueError):
        convert_roman_to_num("K")


def test_convert_roman_to_num_empty_string_raises():
    """P3 fix: empty string must raise ValueError."""
    from src.helper import convert_roman_to_num

    with pytest.raises(ValueError, match="Empty string"):
        convert_roman_to_num("")


def test_convert_roman_to_num_mixed_roman_vietnamese():
    """P2 fix: mixed Roman + Vietnamese characters must not crash."""
    from src.helper import convert_roman_to_num

    # A=1, C=100 → A precedes C so no subtraction → 101
    assert convert_roman_to_num("AC") == 101
    # B=2, X=10 → no subtraction (B not in roman_to_num)
    assert convert_roman_to_num("BX") == 12
