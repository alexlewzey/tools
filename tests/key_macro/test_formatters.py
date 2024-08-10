import pytest

from tools.key_macro.macros import formatters


@pytest.mark.parametrize(
    "input_, expected",
    [
        ("A", "a"),
        ("a", "a"),
        ("1kdTDjI?", "1kdtdji?"),
        ("92847", "92847"),
        ("", ""),
        (r"\N", "\\n"),
    ],
)
def test_to_lower(input_, expected):
    assert formatters._to_lower(input_) == expected


@pytest.mark.parametrize(
    "input_, expected",
    [
        ("hello    mole", "hello mole"),
        ("hello mole", "hello mole"),
        ("h\n  a \ta", "h a a"),
    ],
)
def test_split_join(input_, expected):
    assert formatters._split_join(input_) == expected


@pytest.mark.parametrize(
    "input_, expected",
    [
        ("Hello     mole", "hello_mole"),
        ("Hello Mole 1234", "hello_mole_1234"),
    ],
)
def test_to_snake(input_, expected):
    assert formatters._to_snake(input_) == expected


@pytest.mark.parametrize(
    "input_, expected",
    [
        (
            "a\nb",
            """a
b""",
        ),
        (
            """a

        b

        c""",
            "a\nb\nc",
        ),
    ],
)
def test_remove_blanklines(input_, expected):
    assert formatters._remove_blanklines(input_) == expected


@pytest.mark.parametrize(
    "input_, expected",
    [("hello there", "'hello', 'there',")],
)
def test_to_list(input_, expected):
    assert formatters._to_list(input_) == expected


@pytest.mark.parametrize(
    "input_, expected",
    [("moje is a goon", "moje is a goon\n--------------")],
)
def test_underline(input_, expected):
    assert formatters._underline(input_) == expected


@pytest.mark.parametrize(
    "input_, expected",
    [
        ("print(str(''.join('hello world')))", "str(''.join('hello world'))"),
        ("str(''.join('hello world'))", "''.join('hello world')"),
        ("''.join('hello world')", "'hello world'"),
        ("'hello world'", "'hello world'"),
    ],
)
def test_unnest_parathesis(input_, expected):
    assert formatters._unnest_parathesis(input_) == expected
