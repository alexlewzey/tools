#! /usr/bin/env python3
"""Wrap the clipboard with a function specified as a cmd line arg (if fun from run
window) or via user interface is run from listener_standard script."""

import functools
import logging
import re
import time
import webbrowser
from datetime import date

import black
import pandas as pd
import pyperclip
import sqlfluff
from textblob import TextBlob

from tools.key_macro.keyboard import Typer

logger = logging.getLogger(__name__)

typer = Typer()
LINE_CHAR_LIMIT = 88


def clipboard_in_out(func):
    """Decorator that grabs text from the clipboard passes it to the decorated function
    then copies the text returned by the decorated function to the clipboard."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        clip: str = pyperclip.paste()
        time.sleep(0.1)
        clip = func(clip)
        pyperclip.copy(clip)
        typer.type_text(clip)

    return wrapper


def clipboard_in_out_paste(func):
    """Decorator that grabs text from the clipboard passes it to the decorated function
    then copies the text returned by the decorated function to the clipboard."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        clip: str = pyperclip.paste()
        time.sleep(0.1)
        clip = func(clip)
        pyperclip.copy(clip)
        typer.paste()

    return wrapper


# transforms that do not add or remove characters ######################################


def _to_lower(s: str) -> str:
    return s.lower()


def _to_upper(s: str) -> str:
    return s.upper()


def _to_capitalize(s: str) -> str:
    return s.capitalize()


def _split_join(s: str) -> str:
    return " ".join(s.split())


def _wrap_string_literal(s: str) -> str:
    """Wrap python string literal maximum line length.

    input
    -----
                f"File {file.as_posix()} uploaded to bucket {self.bucket.name} with name {blob_name}"

    output
    ------
                f"File {file.as_posix()} uploaded to bucket {self.bucket.name} "
                f"with name {blob_name}"
    """  # noqa: E501
    max_line_length: int = 88
    mo = re.search(r"^\s+", s)
    if mo is None:
        raise Exception("re match object is None")
    indent_len = len(mo.group())
    target_line_len = max_line_length - indent_len - 3
    data = [(word + " ") for word in s.strip('" f').split()]
    df = pd.DataFrame(data, columns=["word"])
    df.iloc[-1, 0] = df.iloc[-1, 0][:-1]
    df["len"] = df["word"].str.len() + 1
    lines = []
    while not df.empty:
        df["cum_len"] = df["len"].cumsum()
        line = "".join(df[df.cum_len <= target_line_len].word.tolist())
        line = (indent_len * " ") + f'f"{line}"'
        lines.append(line)
        df = df[df.cum_len > target_line_len]
    output = "\n".join(lines)
    return output


def _spell_check(s: str) -> str:
    sentence = TextBlob(s)
    corrected = sentence.correct()
    return str(corrected)


def _to_snake(text: str) -> str:
    lines = [re.sub(r"[\s_]+", "_", line).lower() for line in text.splitlines()]
    return "\n".join(lines)


def _remove_blanklines(s: str) -> str:
    """Remove blank lines from a block of text and return it to the clipboard.

    input
    -----
    hello

    world

    output
    ------
    hello
    world
    """
    lines = [o.strip() for o in s.split("\n") if not o.isspace()]
    lines = [o for o in lines if o not in (None, "")]
    return "\n".join(lines)


def _to_list(s: str) -> str:
    """Format a sequence separated by white space into a python list format.

    input
    -----
    hello there

    output
    ------
    'hello', 'there'
    """
    output = ", ".join([f"'{chars}'" for chars in s.split()]) + ","
    return output


def _underline(s: str) -> str:
    """Add a dash underline to the clipboard item and return to the clipboard.

    input
    -----
    moje is a goon

    output
    ------
    moje is a goon
    --------------
    """
    output = s + "\n" + create_underline(s)
    return output


def create_underline(text: str) -> str:
    clipboard_clean: str = " ".join(text.split())
    return len(clipboard_clean) * "-"


def _format_variables(s: str) -> str:
    """Take the variables from the clipboard and format them as a string to be printed
    and return to clipboard.

    input
    ------
    days_elapsed
    weeks_elapsed

    output
    -----
    print(f'days_elapsed={days_elapsed}',
        f'weeks_elapsed={weeks_elapsed}')
    """
    variables = re.sub(r"[^\w\s]", "", s).split()
    variables = [f"{v}={{{v}:}}" for v in variables]
    output = f"print(f'{', '.join(variables)}')"
    if len(output) > 88:
        output = "', \n      f'".join(output.split(", "))
    return output


# padding formatters ###################################################################


def _pad_right_full(s: str, char: str, left_len: int = 1) -> str:
    """Add a left justify fill of hash characters to the current clipboard item to a
    length of 88 character and return it to the clipboard.

    pad_right_full('#')
    input:
    hello world
    output:
    # hello world ###################################################################...
    """
    s = s[:LINE_CHAR_LIMIT]
    s = f"{char * left_len} " + s.strip("#- ") + " "
    output = s.ljust(LINE_CHAR_LIMIT, f"{char}")
    return output


def _format_hash(s: str) -> str:
    """See: pad_right_full()"""
    return _pad_right_full(s, "#")


def _format_dash(s: str) -> str:
    """See: pad_right_full()"""
    return _pad_right_full(s, "-", left_len=2)


def _format_hash_center(s: str) -> str:
    """Add a center justify fill of hash characters to the current clipboard item to a
    length of 88 character and return it to the clipboard.

    input:
    hello world
    output:
    ##################################### hello world ###############################...
    """
    s = s[:LINE_CHAR_LIMIT]
    s = " " + s.strip() + " "
    output = f"{s:#^88}"
    return output


# specific use case string formatters ##################################################


def _unnest_parentheses(s: str) -> str:
    """Extract the content of the paraenthesis from the current clipboard selection and
    type it out.

    input:
    print(''.join('hello world'))
    output:
    ''.join('hello world')
    """
    s = s.strip()
    match = re.search(r"\(.+\)", s)
    if match:
        # Extract the content within the parenthesis
        content = match.group(0)[1:-1]
    else:
        content = s
    return content


def _format_repr(s: str) -> str:
    """Copy class properties to the clipboard, run this program, and it will format the
    properties as a human readable repr string that you can add to your class.

    input from clipboard:
        self.id = id_
        self.username = username
        self.password = password
    output of script:
        (id=self.id, username=self.username, password=self.password)
    """
    logger.info(f"clipped: {s}")
    lines = [x for x in s.split("\n") if not x.isspace()]
    test_valid_input(lines)
    properties = [x.split(".")[1].strip() for x in lines]
    names = [x.split("=")[0].strip() for x in properties]
    attributes: str = (
        "(" + ", ".join([f"{name}={{self.{name}}}" for name in names]) + ")"
    )
    output: str = (
        f"def __repr__(self):\n\treturn f'{{self.__class__.__name__}}{attributes}'"
    )
    return output


def test_valid_input(lines: list[str]) -> None:
    if not any("." in line for line in lines):
        raise InvalidInputError("Are you using the correct input? eg self.name = name")


class InvalidInputError(TypeError):
    pass


def _imports_to_requirements(s: str) -> str:
    """

    input
    -----
    import requests
    from lxml import html
    from plotly.offline import plot
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By

    output
    ------
    lxml
    requests
    plotly
    selenium
    """
    modules = {re.split(r"[ .]", line)[1] for line in s.splitlines() if line}
    return "\n".join(modules)


def _format_sql(s: str) -> str:
    """Format sql with sqlfluff.

    input
    -----
    select first_name, last_name, email, created_date from `project.dataset.users` where created_date >= '2024-01-01' limit 100

    output
    ------
    SELECT
        first_name,
        last_name,
        email,
        created_date
    FROM `project.dataset.users`
    WHERE created_date >= '2024-01-01'
    LIMIT 100
    """  # noqa: E501
    words = s.split()
    s = words[0].upper() + " " + " ".join(words[1:])
    s = sqlfluff.fix(s, dialect="bigquery")
    return s


def _format_black(s: str) -> str:
    s = black.format_str(s, mode=black.FileMode())
    return s


def _remove_urls(s: str) -> str:
    return re.sub(r"(https?://\S+)", "", s)


def _join_python_string(s: str) -> str:
    lines = (line.lstrip() for line in s.strip().splitlines())
    formatted = "(" + '"\nf"'.join(lines) + ")"
    return formatted


def open_cb_url() -> None:
    """Open the current clipboard url in the default browser."""
    print("opening url")
    url = pyperclip.paste()
    if url.startswith("www."):
        url = f"https://{url}"
    webbrowser.open(url)


def type_days_elapsed() -> None:
    start_date_streak = "2024-04-20"
    days_elapsed_streak = (date.today() - date.fromisoformat(start_date_streak)).days
    years_streak = days_elapsed_streak / 365

    start_date_caffeine = "2024-10-21"
    days_elapsed_caffeine = (
        date.today() - date.fromisoformat(start_date_caffeine)
    ).days
    years_caffeine = days_elapsed_caffeine / 365
    typer.type_text(
        f"streak: {days_elapsed_streak} - {years_streak:.2f}, "
        f"caffeine: {days_elapsed_caffeine} - {years_caffeine:.2f}"
    )


def type_journal_header() -> None:
    typer.type_date()
    typer.type_text(" ")
    type_days_elapsed()
    time.sleep(0.2)
    typer.select_line_at_caret_and_copy()
    time.sleep(0.2)
    format_hash()


def sql_count_distinct() -> None:
    word = typer.select_word_at_caret_and_copy()
    pyperclip.copy(f"count(distinct {word}) n_{word},")
    time.sleep(0.2)
    typer.paste()


# select line first formatters #########################################################


def set_equal_to_self() -> None:
    """Select text on the current line copy it and set it equal to itself
    in: df
    out: df = df"""
    text = typer.select_line_at_caret_and_copy()
    typer.caret_to_line_end()
    typer.type_text(" = " + text)
    logger.info(f"typed: {text}")


def cut_right_equality() -> None:
    """Cut the right side of the equality and add to clipboard.

    in: greeting = 'hello world'
    out: greeting =
    """
    line = typer.select_line_at_caret_and_copy()
    left, right = line.split("=", maxsplit=1)
    pyperclip.copy(right.strip())
    typer.type_text(left + "= ")


to_lower = clipboard_in_out_paste(_to_lower)
to_upper = clipboard_in_out_paste(_to_upper)
to_capitalize = clipboard_in_out_paste(_to_capitalize)
split_join = clipboard_in_out(_split_join)
wrap_string_literal = clipboard_in_out_paste(_wrap_string_literal)
spell_check = clipboard_in_out_paste(_spell_check)
to_snake = clipboard_in_out(_to_snake)
remove_blanklines = clipboard_in_out(_remove_blanklines)
to_list = clipboard_in_out(_to_list)
underline = clipboard_in_out_paste(_underline)
format_variables = clipboard_in_out_paste(_format_variables)
format_hash = clipboard_in_out_paste(_format_hash)
format_dash = clipboard_in_out_paste(_format_dash)
format_hash_center = clipboard_in_out_paste(_format_hash_center)
unnest_parentheses = clipboard_in_out(_unnest_parentheses)
format_repr = clipboard_in_out(_format_repr)
imports_to_requirements = clipboard_in_out_paste(_imports_to_requirements)
format_sql = clipboard_in_out_paste(_format_sql)
format_black = clipboard_in_out_paste(_format_black)
remove_urls = clipboard_in_out_paste(_remove_urls)
join_python_string = clipboard_in_out_paste(_join_python_string)
