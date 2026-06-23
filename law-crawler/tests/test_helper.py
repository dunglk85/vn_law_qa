"""Tests for law-crawler helper functions."""
import pytest


def test_convert_roman_to_num_basic():
    """Convert basic Roman numerals."""
    from helper import convert_roman_to_num

    assert convert_roman_to_num("I") == 1
    assert convert_roman_to_num("V") == 5
    assert convert_roman_to_num("X") == 10
    assert convert_roman_to_num("L") == 50
    assert convert_roman_to_num("D") == 500
    assert convert_roman_to_num("M") == 1000


def test_convert_roman_to_num_compound():
    """Convert compound Roman numerals."""
    from helper import convert_roman_to_num

    assert convert_roman_to_num("IV") == 4
    assert convert_roman_to_num("IX") == 9
    assert convert_roman_to_num("XL") == 40
    assert convert_roman_to_num("XC") == 90
    assert convert_roman_to_num("XLII") == 42


def test_convert_roman_to_num_vietnamese():
    """Convert Vietnamese index characters not in Roman numeral set."""
    from helper import convert_roman_to_num

    assert convert_roman_to_num("A") == 1
    assert convert_roman_to_num("B") == 2
    assert convert_roman_to_num("E") == 5


def test_convert_roman_to_num_case_insensitive():
    """Roman conversion should be case insensitive."""
    from helper import convert_roman_to_num

    assert convert_roman_to_num("iv") == 4
    assert convert_roman_to_num("ix") == 9


def test_extract_input_found():
    """Extract content inside parentheses."""
    from helper import extract_input

    assert extract_input("abc(123)def") == "123"
    assert extract_input("foo('bar')") == "'bar'"
    assert extract_input("onclick=\"select('MAPC_VALUE')\"") == "'MAPC_VALUE'"


def test_extract_input_not_found():
    """Return None when no parentheses found."""
    from helper import extract_input

    assert extract_input("no parentheses here") is None
    assert extract_input("") is None
    assert extract_input("()") == ""


def test_extract_input_nested():
    """Extract from first parentheses only."""
    from helper import extract_input

    assert extract_input("(a(b)c)") == "a(b"


def test_convert_roman_to_num_unknown_char_raises():
    """I1 fix: characters outside both Roman and Vietnamese index sets must raise ValueError."""
    from helper import convert_roman_to_num
    import pytest

    with pytest.raises(ValueError, match="Unknown character"):
        convert_roman_to_num("K")  # K is not a Roman numeral and not in A-J

    with pytest.raises(ValueError, match="Unknown character"):
        convert_roman_to_num("Z")  # Z is outside the A-J alphabet range

    with pytest.raises(ValueError, match="Unknown character"):
        convert_roman_to_num("1")  # digits should also be rejected


def test_convert_roman_to_num_alphabet_boundary():
    """I1 fix: J (10th letter) is the last supported Vietnamese index."""
    from helper import convert_roman_to_num
    import pytest

    assert convert_roman_to_num("J") == 10   # last valid alphabet entry
    with pytest.raises(ValueError):
        convert_roman_to_num("K")            # first invalid one
