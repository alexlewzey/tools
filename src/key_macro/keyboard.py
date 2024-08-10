"""Tools for triggering functionality with keyboard presses.

includes a keyboard listener, key history and currently pressed keys
"""
import logging
import sys
import time
from datetime import date, datetime
from functools import partial
from typing import Callable

import pyperclip
from key_macro import core
from pynput.keyboard import Controller, Key, KeyCode

logger = logging.getLogger(__name__)

MAX_DEQUE_SIZE = 20


class Typer(Controller):
    """Container for all functionality relating to pressing keys on the keyboard."""

    def __init__(self):
        super().__init__()
        self.cmd_ctrl = Key.ctrl_l if sys.platform == "win32" else Key.cmd

    def press_key(self, key, num_presses: int = 1) -> None:
        """Simulate pressing a key a specified number of times."""
        for _ in range(num_presses):
            self.press(key)
            self.release(key)

    def hotkey(self, key_hold, key_press, rest: float = 0) -> None:
        """Perform a simple hotkey where two keys are pressed simultaneously."""
        with self.pressed(key_hold):
            self.press(key_press)
            self.release(key_press)
        time.sleep(rest)

    def hotkey_three(self, key_hold1, key_hold2, key_press, rest: float = 0) -> None:
        """Perform hotkey where three keys are held down simultaneously."""
        with self.pressed(key_hold1):
            with self.pressed(key_hold2):
                self.press(key_press)
                self.release(key_press)
        if rest:
            time.sleep(rest)

    def enter(self, rest: float = 0) -> None:
        self.press_key(Key.enter)
        if rest:
            time.sleep(rest)

    def backspace(self, rest: float = 0) -> None:
        self.press_key(Key.backspace)
        if rest:
            time.sleep(rest)

    def alt_tab(self, sleep_after: float = 0.2) -> None:
        """Hotkey alt + tab, includes a sleep after so typing does not occur while
        the windows are cycling."""
        if sys.platform == "win32":
            logger.info("alt_tab: windows")
            self.hotkey(Key.alt_l, Key.tab)
        else:
            logger.info("alt_tab: mac")
            self.hotkey(Key.cmd, Key.tab)
        time.sleep(sleep_after)

    def paste(self) -> None:
        if sys.platform == "win32":
            self.hotkey(Key.ctrl_l, KeyCode(char="v"))
        elif sys.platform == "darwin":
            self.hotkey(Key.cmd, KeyCode(char="v"))
        else:
            raise ValueError(f"{self.paste.__name__} no handler for operating system")

    def copy(self) -> str:
        self.hotkey(self.cmd_ctrl, KeyCode(char="c"))
        time.sleep(0.2)
        return pyperclip.paste()

    def paste_text(self, text: str, n_left: int | None = None) -> None:
        pyperclip.copy(text)
        self.paste()
        if n_left:
            self.press_key(Key.left, n_left)

    def partial_paste(self, text: str, n_left: int | None = None) -> Callable:
        return partial(self.paste_text, text, n_left)

    def type_text(self, text: str, sleep_after: float | None = None) -> None:
        """Types text character by character and will handle newline escape
        characters."""
        lines = text.splitlines()
        for line in lines:
            for char in line:
                self.press_key(char, num_presses=1)
            if len(lines) > 1:
                self.enter()

        if sleep_after:
            time.sleep(sleep_after)

    def type_date(self) -> None:
        self.type_text(date.today().isoformat())

    def type_timestamp(self) -> None:
        self.type(datetime.now().replace(microsecond=0).isoformat())

    def next_lines(self) -> None:
        self.caret_to_line_end()
        self.press_key(Key.enter)

    def partial_typing(
        self, text: str, n_left: int | None = None, line_end: bool = False
    ) -> Callable:
        """Return a callable that will simulate typing text when subsequently
        called."""

        def call_typing() -> None:
            if line_end:
                self.caret_to_line_end()
            self.type(text)
            if n_left:
                self.press_key(Key.left, n_left)

        return call_typing

    def __call__(self, text, n_left: int | None = None, line_end: bool = False):
        return self.partial_typing(text, n_left=n_left, line_end=line_end)

    def selection_to_clipboard(self) -> str:
        with self.pressed(self.cmd_ctrl):
            self.press(KeyCode(char="c"))
            self.release(KeyCode(char="c"))
        time.sleep(0.1)  # allow item to be added to clipboard
        result = pyperclip.paste()
        logger.info(f"selection_to_clipboard: output={result}"[:300])
        time.sleep(0.2)
        return result
    

    def select_text_before(self, length: int) -> None:
        for _ in range(length):
            self.hotkey(Key.shift, Key.left)

    def select_line_at_caret(self) -> None:
        if sys.platform == "win32":
            self.press_key(Key.end)
            self.hotkey(Key.shift, Key.home)
        elif sys.platform == "darwin":
            self.hotkey(Key.cmd, Key.right)
            self.hotkey_three(Key.cmd, Key.shift, Key.left)

    def select_word_at_caret_and_copy(self) -> str:
        """Selects the word at the cursor, if cursor is within word it will go to end
        of word and then select it."""
        self.hotkey_three(Key.alt, Key.shift, Key.left)
        return self.selection_to_clipboard()

    def select_line_at_caret_and_copy(self) -> str:
        self.select_line_at_caret()
        return self.selection_to_clipboard()

    def next_chrome_tab(self) -> None:
        """Cycle to the next Chrome tab."""
        if sys.platform == "win32":
            self.hotkey(Key.ctrl_l, Key.page_down)
        elif sys.platform == "darwin":
            self.hotkey(Key.ctrl, Key.page_down)

    def previous_chrome_tab(self) -> None:
        """Cycle to the previous Chrome tab."""
        if sys.platform == "win32":
            self.hotkey(Key.ctrl_l, Key.page_up)
        elif sys.platform == "darwin":
            self.hotkey(Key.ctrl, Key.page_up)

    def copy_url_from_next_tab(self) -> str:
        """All the sleeps are required for this to run without crashing."""
        self.next_chrome_tab()
        time.sleep(0.5)
        self.hotkey(self.cmd_ctrl, KeyCode(char="l"))
        time.sleep(0.3)
        url = self.copy()
        time.sleep(1)
        print(f"copy_url_from_next_tab: {url}")
        return url

    def caret_to_line_end(self) -> None:
        """Move caret to end of line."""
        if sys.platform == "win32":
            self.press_key(Key.end)
        elif sys.platform == "darwin":
            self.hotkey(Key.cmd, Key.right)

    def caret_to_line_start(self) -> None:
        """Move caret to start of line."""
        if sys.platform == "win32":
            self.press_key(Key.home)
        elif sys.platform == "darwin":
            self.hotkey(Key.cmd, Key.left)

    def select_previous_lines(self, n_lines: int) -> None:
        """Select a specified number of lines preceding and including the current
        line of the cursor."""
        self.press_key(Key.up, num_presses=n_lines)
        self.caret_to_line_start()
        for _ in range(n_lines):
            self.hotkey(Key.shift, Key.down)
        self.hotkey_three(Key.cmd, Key.shift, Key.right)

    def select_browser_url(self) -> str:
        self.hotkey(self.cmd_ctrl, KeyCode(char="l"))
        time.sleep(0.5)
        return self.copy()

    def get_urls(self, n_urls: int | None = None) -> list[str]:
        """"""
        urls = [self.select_browser_url()]

        if n_urls:
            print(f"n_urls: {n_urls}")
            for _ in range(n_urls - 1):
                urls.append(self.copy_url_from_next_tab())
        else:
            while True:
                url = self.copy_url_from_next_tab()
                if url == urls[-1]:
                    continue
                if url in urls:
                    break
                urls.append(url)

        logger.debug(f"returning urls: {urls}")
        return urls
