"""tools for triggering functionality with keyboard presses. includes a keyboard listener, key history and currently pressed keys"""
import logging
import sys
import time
from datetime import datetime
from typing import Callable, List
from typing import Optional

import pyperclip
from pynput.keyboard import KeyCode, Controller, Key
from clmac.helpers import core
logger = logging.getLogger(__name__)

MAX_DEQUE_SIZE = 20
typer = Controller()



class Typer(Controller):
    """container for all functionality relating to pressing keys on the keyboard"""

    def __init__(self):
        super().__init__()
        self.cmd_ctrl = Key.ctrl_l if sys.platform == 'win32' else Key.cmd

    def press_key(self, key, num_presses: int = 1) -> None:
        """simulate pressing a key a specified number of times"""
        for num_presses in range(num_presses):
            self.press(key)
            self.release(key)

    def hotkey(self, key_hold, key_press, rest: float = 0) -> None:
        """perform a simple hotkey where two keys are pressed simultaneously"""
        with self.pressed(key_hold):
            self.press(key_press)
            self.release(key_press)
        time.sleep(rest)

    def hotkey_three(self, key_hold1, key_hold2, key_press, rest: float = 0) -> None:
        """perform hotkey where three keys are held down simultaneously"""
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
        """hotkey alt + tab, includes a sleep after so typing does not occur while the windows are cycling"""
        if sys.platform == 'win32':
            logger.info(f'alt_tab: windows')
            self.hotkey(Key.alt_l, Key.tab)
        else:
            logger.info(f'alt_tab: mac')
            self.hotkey(Key.cmd, Key.tab)
        time.sleep(sleep_after)

    def paste(self) -> None:
        if sys.platform == 'win32':
            self.hotkey(Key.ctrl_l, KeyCode(char='v'))
        elif sys.platform == 'darwin':
            self.hotkey(Key.cmd, KeyCode(char='v'))
        else:
            raise ValueError(f'{self.paste.__name__}no handler for operating system')

    def copy(self) -> str:
        self.hotkey(self.cmd_ctrl, KeyCode(char='c'))
        time.sleep(0.2)
        return pyperclip.paste()

    def type_text(self, text: str, sleep_after: Optional[float] = None) -> None:
        """types text character by character and will handle newline escape characters"""
        lines = text.splitlines()
        for line in lines:
            for char in line:
                self.press_key(char, num_presses=1)
            if len(lines) > 1:
                self.enter()

        if sleep_after:
            time.sleep(sleep_after)

    def type_date(self) -> None:
        self.type_text(get_date())

    def type_timestamp(self) -> None:
        self.type(get_timestamp())

    def type_cycled_case(self) -> None:
        self.select_word_at_caret_and_copy()
        selection = pyperclip.paste()
        self.type(cycle_case(selection))
        self.select_text_before(len(selection))
        self.copy()
        self.press_key(Key.right)

    def next_lines(self) -> None:
        self.caret_to_line_end()
        self.press_key(Key.enter)

    def partial_typing(self, text: str, n_left: Optional[int] = None) -> Callable:
        """return a callable that will simulate typing text when subsequently called"""

        def call_typing() -> None:
            self.type(text)
            if n_left:
                self.press_key(Key.left, n_left)

        return call_typing

    def __call__(self, text, n_left: Optional[int] = None):
        return self.partial_typing(text, n_left)

    @core.sleep_after(0.2)
    @core.log_output()
    def selection_to_clipboard(self) -> str:
        with self.pressed(self.cmd_ctrl):
            self.press(KeyCode(char='c'))
            self.release(KeyCode(char='c'))
        time.sleep(0.1)  # allow item to be added to clipboard
        return pyperclip.paste()

    def select_text_before(self, length: int) -> None:
        for i in range(length):
            self.hotkey(Key.shift, Key.left)

    def select_line_at_caret(self) -> None:
        if sys.platform == 'win32':
            self.press_key(Key.end)
            self.hotkey(Key.shift, Key.home)
        elif sys.platform == 'darwin':
            self.hotkey(Key.cmd, Key.right)
            self.hotkey_three(Key.cmd, Key.shift, Key.left)

    def select_word_at_caret_and_copy(self) -> str:
        """selects the word at the cursor, if cursor is within word it will go to end of word
        and then select it"""
        self.hotkey_three(Key.alt, Key.shift, Key.left)
        return self.selection_to_clipboard()

    def select_line_at_caret_and_copy(self) -> str:
        self.select_line_at_caret()
        return self.selection_to_clipboard()

    def next_chrome_tab(self) -> None:
        """cycle to the next chrome tab"""
        if sys.platform == 'win32':
            self.hotkey(Key.ctrl_l, Key.page_down)
        elif sys.platform == 'darwin':
            self.hotkey(Key.ctrl, Key.page_down)

    def previous_chrome_tab(self) -> None:
        """cycle to the previous chrome tab"""
        if sys.platform == 'win32':
            self.hotkey(Key.ctrl_l, Key.page_up)
        elif sys.platform == 'darwin':
            self.hotkey(Key.ctrl, Key.page_up)

    def copy_url_from_next_tab(self) -> str:
        """all the sleeps are required for this to run without crashing"""
        self.next_chrome_tab()
        time.sleep(0.5)
        self.hotkey(self.cmd_ctrl, KeyCode(char='l'))
        time.sleep(0.3)
        url = self.copy()
        time.sleep(1)
        print(f'copy_url_from_next_tab: {url}')
        return url

    def caret_to_line_end(self) -> None:
        """move caret to end of line"""
        if sys.platform == 'win32':
            self.press_key(Key.end)
        elif sys.platform == 'darwin':
            self.hotkey(Key.cmd, Key.right)

    def caret_to_line_start(self) -> None:
        """move caret to start of line"""
        if sys.platform == 'win32':
            self.press_key(Key.home)
        elif sys.platform == 'darwin':
            self.hotkey(Key.cmd, Key.left)

    def select_previous_lines(self, n_lines: int) -> None:
        """select a specified number of lines preceding and including the current line of the cursor"""
        self.press_key(Key.up, num_presses=n_lines)
        self.caret_to_line_start()
        for _ in range(n_lines):
            self.hotkey(Key.shift, Key.down)
        self.hotkey_three(Key.cmd, Key.shift, Key.right)

    def select_browser_url(self) -> Optional[str]:
        self.hotkey(self.cmd_ctrl, KeyCode(char='l'))
        time.sleep(0.5)
        return self.copy()

    def get_urls(self, n_urls: Optional[int] = None) -> List[str]:
        """"""
        urls = []
        urls.append(self.select_browser_url())

        if n_urls:
            print(f'n_urls: {n_urls}')
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

        logger.debug(f'returning urls: {urls}')
        return urls


def get_date() -> str:
    return str(datetime.now().date())


def get_timestamp() -> str:
    now = datetime.now()
    return datetime.strftime(now, '%Y-%m-%d, %H:%M')


def cycle_case(text: str) -> str:
    """cycle through lower > upper > capitalise etc"""
    if not text.islower() and not text.isupper():
        output = text.lower()
    elif text.islower():
        output = text.upper()
    elif text.isupper():
        output = capitalise_all(text)
    else:
        raise ValueError('cycle_case something went wrong')
    return output


def capitalise_all(text: str) -> str:
    """capitalise all word in a string, ignoring special cases"""
    capitalise_ignore = ['of', 'on', 'an', 'in', 'and']
    capitalised = []
    for word in text.split():
        if word.lower() in capitalise_ignore:
            capitalised.append(word.lower())
        else:
            capitalised.append(word.capitalize())
    return ' '.join(capitalised)
