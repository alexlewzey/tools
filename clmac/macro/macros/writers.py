#! /usr/bin/env python3
"""
functions that automate typing which features control character which cannot be fed into the macro encoding as callable
text
"""
from typing import *

from pynput.keyboard import Key, KeyCode

from clmac.helpers.typer import Typer

typer = Typer()

REST = 0.2


class GDocWriter:
    """Object representing automation of typing (typer) in a Google Doc"""

    def __init__(self, lst: List[str]):
        self.lst = lst

    def indent(self):
        typer.hotkey(Key.cmd, KeyCode(char=']'))

    def dedent(self):
        typer.hotkey(Key.cmd, KeyCode(char='['))

    def type_headers(self):
        """Type out list as bulleted headers in Google docs with indented sub bullets"""
        for char in self.lst:
            typer.type_text(char)
            typer.enter()
        self.indent()

        for _ in self.lst:
            typer.press_key(Key.up, 2)
            typer.enter()
            self.indent()

        typer.press_key(Key.backspace, 3)
        typer.press_key(Key.down, 2)

    def __call__(self):
        self.type_headers()


def write_print() -> None:
    typer.type_text('print()')
    typer.press_key(Key.left)


def git_commit() -> None:
    """when run at the git terminal will automate add, commit and push commands"""
    text = "git add -A\n git commit -m '"
    typer.type_text(text)


def caret_to_line_start() -> None:
    typer.caret_to_line_start()


def caret_to_line_end() -> None:
    typer.caret_to_line_end()


def type_drop() -> None:
    typer.type_text('.drop(, 1, inplace=True)')
    typer.press_key(Key.left, 18)


def type_columns() -> None:
    typer.type_text('pyperclip.copy(str(.columns.tolist()))')
    typer.press_key(Key.left, 19)


def type_sort_values() -> None:
    typer.type_text('.sort_values()')
    typer.press_key(Key.left, 1)
