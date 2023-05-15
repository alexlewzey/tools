"""NOTE should be removed and refactored to pynputs."""
import logging

import pyautogui
import pyperclip

from clmac.helpers import core

logger = logging.getLogger(__name__)


@core.sleep_after(secs_after=0.1)
@core.log_output()
def read_clipboard() -> str:
    return pyperclip.paste().strip()


@core.log_input()
def copy_to_clipboard(text: str) -> None:
    pyperclip.copy(text)


@core.sleep_after(0.2)
def refocus() -> None:
    """Use the hot key alt + tab to switch back to the previous gui, required at the
    start of any pyautowin script that is run from the run window."""
    pyautogui.hotkey("alt", "tab")
