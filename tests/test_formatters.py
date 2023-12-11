from unittest.mock import patch

import pyperclip
import pytest

import clmac.macros.formatters as f


def copy_func_paste_assert(input, expected, func):
    pyperclip.copy(input)
    func()
    output = pyperclip.paste()
    assert output == expected


@pytest.mark.parametrize(
    "input, expected",
    [
        ("A", "a"),
        ("a", "a"),
        ("1kdTDjI?", "1kdtdji?"),
        ("92847", "92847"),
        ("", ""),
        (r"\N", "\\n"),
    ],
)
@patch("clmac.macros.formatters.typer")
def test_to_lower(typer_mock, input, expected):
    copy_func_paste_assert(input, expected, f.to_lower)
    typer_mock.paste.assert_called_once()


@pytest.mark.parametrize(
    "input, expected",
    [
        ("hello    mole", "hello mole"),
        ("hello mole", "hello mole"),
        ("h\n  a \ta", "h a a"),
    ],
)
@patch("clmac.macros.formatters.typer")
def test_split_join(typer_mock, input, expected):
    copy_func_paste_assert(input, expected, f.split_join)


@pytest.mark.parametrize(
    "input, expected",
    [
        ("Hello     mole", "hello_mole"),
        ("Hello Mole 1234", "hello_mole_1234"),
    ],
)
@patch("clmac.macros.formatters.typer")
def test_to_snake(typer_mock, input, expected):
    copy_func_paste_assert(input, expected, f.to_snake)


@pytest.mark.parametrize(
    "input, expected",
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
@patch("clmac.macros.formatters.typer")
def test_remove_blanklines(typer_mock, input, expected):
    copy_func_paste_assert(input, expected, f.remove_blanklines)


@pytest.mark.parametrize(
    "input, expected",
    [("hello there", "'hello', 'there',")],
)
@patch("clmac.macros.formatters.typer")
def test_to_list(typer_mock, input, expected):
    copy_func_paste_assert(input, expected, f.to_list)


@pytest.mark.parametrize(
    "input, expected",
    [("moje is a goon", "moje is a goon\n--------------")],
)
@patch("clmac.macros.formatters.typer")
def test_underline(typer_mock, input, expected):
    copy_func_paste_assert(input, expected, f.underline)


@pytest.mark.parametrize(
    "input, expected",
    [
        ("print(str(''.join('hello world')))", "str(''.join('hello world'))"),
        ("str(''.join('hello world'))", "''.join('hello world')"),
        ("''.join('hello world')", "'hello world'"),
        ("'hello world'", "'hello world'"),
    ],
)
@patch("clmac.macros.formatters.typer")
def test_unnest_parathesis(typer_mock, input, expected):
    copy_func_paste_assert(input, expected, f.unnest_parathesis)
