#! /usr/bin/env python3
"""Wrap the clipboard with a function specified as a cmd line arg (if fun from run
window) or via user interface is run from listener_standard script."""
import functools
import logging
import re
import textwrap
import time
import traceback
from typing import List

import pyperclip
from pynput.keyboard import Key
from textblob import TextBlob

import clmac.helpers.automation as auto
from clmac.helpers.typer import Typer

logger = logging.getLogger(__name__)

typer = Typer()
LINE_CHAR_LIMIT = 88


def wrap_cb(prefix: str) -> None:
    wrapped = wrap_clipboard_in_func_and_add_to_clipboard(prefix)
    typer.type_text(wrapped)


def wrap_clipboard_in_func_and_add_to_clipboard(func_nm: str) -> str:
    clipboard: str = auto.read_clipboard()
    wrapped = func_nm + "(" + clipboard + ")"
    auto.copy_to_clipboard(wrapped)
    return wrapped


def wrap_clipboard_in_fstring() -> str:
    clipboard: str = auto.read_clipboard()
    wrapped = "f'{" + clipboard + "}'"
    auto.copy_to_clipboard(wrapped)
    typer.paste()
    typer.press_key(Key.left, num_presses=len(clipboard) + 3)
    return wrapped


def wrap_for_loop() -> None:
    clipboard: str = auto.read_clipboard()
    wrapped = f"for i in {clipboard}:"
    auto.copy_to_clipboard(wrapped)
    typer.type_text(wrapped)
    typer.enter()


def clipboard_in_out(func):
    """Decorator that grabs text from the clipboard passes it to the decorated function
    then copies the text returned by the decorated function to the clipboard."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        clip: str = auto.read_clipboard()
        clip = func(clip)
        pyperclip.copy(clip)
        typer.type_text(clip)

    return wrapper


def clipboard_in_out_paste(func):
    """Decorator that grabs text from the clipboard passes it to the decorated function
    then copies the text returned by the decorated function to the clipboard."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        clip: str = auto.read_clipboard()
        clip = func(clip)
        pyperclip.copy(clip)
        typer.paste()

    return wrapper


@clipboard_in_out
def to_snake(text: str) -> str:
    lines = [re.sub("[\s_]+", "_", line).lower() for line in text.splitlines()]
    return "\n".join(lines)


@clipboard_in_out
def snake_to_camel(name: str) -> str:
    return "".join(word.title() for word in name.split("_"))


@clipboard_in_out
def camel_to_snake(name: str) -> str:
    # return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()
    return re.sub("_+", "_", name)


def set_equal_to_self() -> None:
    """Select text on the current line copy it and set it equal to itself
    in: df
    out: df = df"""
    text = line_at_caret_to_cb()
    typer.caret_to_line_end()
    typer.type_text(" = " + text)
    logger.info(f"typed: {text}")


def cut_right_equality() -> None:
    """Cut the right side of the equality and add to clipboard.

    in: greeting = 'hello world'
    out: greeting =
    """
    line = line_at_caret_to_cb()
    left, right = line.split("=", maxsplit=1)
    pyperclip.copy(right.strip())
    typer.type_text(left + "= ")


def line_at_caret_to_cb() -> str:
    """Regardless of cursor position select the whole line and copy it to the
    clipboard."""
    line = typer.select_line_at_caret_and_copy()
    logger.info(f"line at the caret: {line}")
    return line


def word_at_caret_to_cb() -> str:
    """Select word to the left of the cursor and copy to clipboard, Equivalent to shift
    + alt + left."""
    line = typer.select_word_at_caret_and_copy()
    logger.info(f"line at the caret: {line}")
    return line


def pad_right_full(char: str, left_len: int = 1) -> None:
    """Add a left justify fill of hash characters to the current clipboard item to a
    length of 88 character and return it to the clipboard.

    pad_right_full('#')
    input:
    hello world
    output:
    # hello world ######################################################################################################
    """
    title: str = pyperclip.paste()[:LINE_CHAR_LIMIT]
    title = f"{char * left_len} " + title.strip(f"#- ") + " "
    output = title.ljust(LINE_CHAR_LIMIT, f"{char}")
    pyperclip.copy(output)
    typer.type_text(output)


def fmt_hash():
    """see: pad_right_full()"""
    pad_right_full("#")


def fmt_dash():
    """see: pad_right_full()"""
    pad_right_full("-", left_len=2)


def fmt_hash_center():
    """Add a center justify fill of hash characters to the current clipboard item to a
    length of 88 character and return it to the clipboard.

    input:
    hello world
    output:
    ##################################################### hello world ##################################################
    """
    title: str = pyperclip.paste()[:LINE_CHAR_LIMIT]
    title = " " + title.strip() + " "
    output = f"{title:#^88}"
    pyperclip.copy(output)
    typer.type_text(output)


def unnest_parathesis():
    """Extract the content of the paraenthesis from the current clipboard selection and
    type it out.

    input:
    print(''.join('hello world'))
    output:
    ''.join('hello world')
    """
    line: str = pyperclip.paste().strip()
    inner = re.search("\(.+\)", line)
    if inner:
        inner = inner.group(0)[1:-1]
    else:
        inner = line
    pyperclip.copy(inner)
    typer.type_text(inner)
    for _ in range(len(inner)):
        typer.hotkey(Key.shift, Key.left)


def wrap_text(max_len: int = 88) -> None:
    """Wrap text to a maximum line length."""
    wrapped = textwrap.fill(pyperclip.paste(), width=max_len).strip()
    pyperclip.copy(wrapped)


def rm_doublespace() -> None:
    """Copy selected text to clipboard format any consecutive spaces as a single space
    and add the return string to the clipboard."""
    clipboard = pyperclip.paste()
    text_cleaned = remove_consecutive_space(clipboard)
    pyperclip.copy(text_cleaned)


def remove_consecutive_space(text: str) -> str:
    lines = [x for x in text.split("\n")]
    line_with_no_consecutive_spaces = [" ".join(line.split()) for line in lines]
    return "\n".join(line_with_no_consecutive_spaces)


def rm_blanklines() -> None:
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
    clipboard = pyperclip.paste()
    lines = [x.strip() for x in clipboard.split("\n") if not x.isspace()]
    lines = list(filter(None, lines))
    lines = "\n".join(lines)
    pyperclip.copy(lines)
    logger.info(f"Removed blank lines...")


@clipboard_in_out
def split_join(name: str) -> str:
    return " ".join(name.split())


def fmt_repr():
    """Copy class properties to the clipboard, run this program, and it will format the
    properties as a human readable repr string that you can add to your class.

    input from clipboard:
        self.id = id_
        self.username = username
        self.password = password
    output of script:
        (id=self.id, username=self.username, password=self.password)
    """
    clip = pyperclip.paste()
    logger.info(f"clipped: {clip}")
    lines = [x for x in clip.split("\n") if not x.isspace()]

    test_valid_input(lines)
    properties = [x.split(".")[1].strip() for x in lines]
    names = [x.split("=")[0].strip() for x in properties]

    output_text: str = (
        "(" + ", ".join([f"{name}={{self.{name}}}" for name in names]) + ")"
    )
    output_text: str = (
        f"def __repr__(self):\n\treturn f'{{self.__class__.__name__}}{output_text}'"
    )

    logger.info(f"added to clipboard: {output_text}")
    pyperclip.copy(output_text)


def test_valid_input(lines: List[str]) -> None:
    if not any(["." in line for line in lines]):
        raise InvalidInputError("Are you using the correct input? eg self.name = name")


class InvalidInputError(TypeError):
    pass


def fmt_pycharm_params():
    """Format pycharm params (from clipboard) as args and copies to clipboard. you can
    return multi-line params by passing in the command line arg nl.

    clipboard inputs
    ----------------
    dstk.dptk def bin_categories_lt_thresh(df: DataFrame,
                                 thresh: float,
                                 grp_var: str,
                                 cat_var: str,
                                 cont_var: str,
                                 non_cat_vars: List[str]) -> DataFrame

      < Python 3.7 >

    clipboard outputs
    -----------------
    df=, thresh=, grp_var=, cat_var=, cont_var=, non_cat_vars=
    """
    clip = pyperclip.paste()
    match = re.search("\((.+)\)", clip, flags=re.DOTALL)
    if match:
        clip = match.group(1)
    lines = [x for x in clip.split("\n") if not x.isspace()]
    params_ = [x.split(":")[0].strip() for x in lines]
    args_: str = "=, ".join(params_) + "="
    pyperclip.copy(args_)


def fmt_list() -> None:
    """Format the clipboard as a python style list where each line represents a list
    item."""
    text = pyperclip.paste()
    items = text2list(text)
    items_str = list2str(items)
    pyperclip.copy(items_str)


def to_list() -> None:
    """Format a sequence separated by white space into a python list format.

    input
    -----
    hello there

    output
    ------
    'hello', 'there'
    """
    output = ", ".join([f"'{chars}'" for chars in pyperclip.paste().split()]) + ","
    pyperclip.copy(output)


def fmt_as_multiple_lines() -> None:
    """Transform a sequence separated by white space into a string where every item in
    the sequence is on a new line.

    input
    -----
    items = [1, 2, 3, 4]

    output
    ------
    items = [
        1,
        2,
        3,
        4
    ]
    """
    text = re.sub(r"([,\[{])", r"\1\n", pyperclip.paste())
    text = re.sub(r"([]}])", r"\n\1", text)
    text = re.sub(r"(\()([^)]+)", r"\1\n\2", text)
    text = re.sub(r"([^(]+)(\))", r"\1\n\2", text)
    pyperclip.copy(text)


def fmt_params_as_multiline() -> None:
    """

    input
    -----

    def neighbours2df(neighbors: Tuple[np.ndarray, np.ndarray], trial_idx: pd.Series, ctrl_idx: pd.Series) -> pd.DataFrame:

    output
    ------

    """
    text = re.sub(r"(,)", r"\1\n", pyperclip.paste())
    text = re.sub(r"(\()([^)]+)", r"\1\n\2", text)
    text = re.sub(r"([^(]+)(\))", r"\1\n\2", text)
    pyperclip.copy(text)
    time.sleep(0.1)
    typer.paste()


def fmt_as_pipe() -> None:
    """Transform a sequence separated by white space into a string where every item in
    the sequence is on a new line.

    input
    -----

    output
    ------
    """
    text = re.sub(r"(\))(\.)", r"\1\n\2", pyperclip.paste())
    text = re.sub(r"( = )", r"\1(", text)
    pyperclip.copy(text + "\n)")
    time.sleep(0.1)
    typer.paste()


def text2list(text: str) -> List[str]:
    """Convert a multi-line segment of text and convert it into a list where every line
    represent a list item."""
    return [x.strip(" ,\n") for x in text.split("\n") if x]


def list2str(lst: List[str]) -> str:
    """Take a list and format it as the python str."""
    text = ""
    for item in lst:
        text += f"    '{item.strip().strip(',')}',\n"

    return text


def fmt_underline():
    """Add a dash underline to the clipboard item and return to the clipboard.

    clipboard input
    ---------------

    moje is a goon

    clipboard output
    ----------------

    moje is a goon
    --------------
    """
    clip = auto.read_clipboard()
    output = clip + "\n" + make_underline(clip)
    auto.copy_to_clipboard(output)
    typer.type_text(output)


def make_underline(text: str) -> str:
    clipboard_clean: str = " ".join(text.split())
    return len(clipboard_clean) * "-"


def fmt_class_properties_multiline():
    """Format the arguments of a class as class properties to paste into the __init__
    function of the class.

    clipboard output
    ----------------
    self.arg1 = arg1
    self.arg2 = arg2
    self.arg3 = arg3
    """
    fmt_multiline = lambda params: "\n".join(
        [f"self.{param} = {param}" for param in params]
    )
    output = fmt_multiline(get_class_properties(pyperclip.paste()))
    pyperclip.copy(output)
    for line in output.splitlines():
        typer.type_text(line)
    typer.press_key(Key.enter)


def fmt_class_properties_multiassign():
    """Format the arguments of a class as class properties to paste into the __init__
    function of the class.

    clipboard output
    ----------------
    self.arg1, self.arg2, self.arg3 = arg1, arg2, arg3
    """

    fmt_multiassign = (
        lambda params: ", ".join([f"self.{param}" for param in params])
        + " = "
        + ", ".join([f"{param}" for param in params])
    )
    output = fmt_multiassign(get_class_properties(pyperclip.paste()))
    pyperclip.copy(output)
    typer.type_text(output)
    typer.press_key(Key.enter)


def get_class_properties(text: str) -> List[str]:
    if "self" in text:
        text = text.split("self,")[-1]
    params = re.search(r"\w[^)]+", text).group(0).split(",")
    return [clean_param(param) for param in params]


def clean_param(param):
    """For a single parameter string truncate anything beyond the param name.

    input
    -----
    cbs: int = None
    output
    ------
    cbs
    """
    return re.search("[^=:)]+", param).group().strip()


def sql_col_as_mil() -> None:
    """Format numeric sql clause as millions.

    input
    -----
    sum(spend) as spend,

    output
    ------
    sum(spend) / 1000000 as spend_m,
    """
    cb: str = pyperclip.paste().strip()
    left, right = cb.split(maxsplit=1)
    left += " / 1000000 "
    right = right.strip(",") + "_m,"
    typer.type_text(left + right)


def parse_sql_table() -> None:
    try:
        cb = typer.select_line_at_caret_and_copy()
        cb = " ".join(cb.split())
        table = re.search("[^\s]+\.[^\s]+\.[^\s]+", cb).group(0)
        pyperclip.copy(table)
    except AttributeError:
        logger.error(f"error: {traceback.format_exc()}")


def fmt_sql_table_as_python() -> None:
    """"""
    cb: str = " ".join(pyperclip.paste().split()).replace("select * from ", "")
    code = f''' = pygcp.read_gbq("""select * from {cb}""")'''
    pyperclip.copy(code)
    typer.paste()
    typer.caret_to_line_start()


@clipboard_in_out_paste
def fmt_print_variables(s: str) -> str:
    """Take the variables from the clipboard and format them as a string to be printed
    and return to clipboard.

    input
    ------
    days_elapsed
    weeks_elapsed

    output
    -----
    print(f'days_elapsed={days_elapsed:.3f}, weeks_elapsed={weeks_elapsed:.3f}')
    """
    variables = re.sub("[^\w\s]", "", s).split()
    variables = [f"{v}={{{v}:.3f}}" for v in variables]
    output = f"print(f'{', '.join(variables)}')"
    if len(output) > 88:
        output = "', \n      f'".join(output.split(", "))
    return output


@clipboard_in_out_paste
def swap_quotation_marks(s: str) -> str:
    """For a string taken from the clipboard replace all the quotation marks of one type
    with the other type i.e. single for double and vice versa. If a string contains a
    single quote it will all replace single marks with doubles.

    input
    -----
    {"start_date": '2023-01-07'}

    output
    ------
    {"start_date": "2023-01-07"}
    """
    s = s.replace("'", '"') if "'" in s else s.replace('"', "'")
    return s


@clipboard_in_out_paste
def correct_spelling(s: str) -> str:
    sentence = TextBlob(s)
    corrected = sentence.correct()
    return str(corrected)


@clipboard_in_out_paste
def to_lower(s: str) -> str:
    return s.lower()


@clipboard_in_out_paste
def to_upper(s: str) -> str:
    return s.upper()


@clipboard_in_out_paste
def imports_to_requirements(s: str) -> str:
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
    modules = set([re.split(r"[ .]", line)[1] for line in s.splitlines() if line])
    return "\n".join(modules)
