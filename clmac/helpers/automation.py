"""NOTE should be removed and refactored to pynputs"""
import logging

import pyautogui
import pyperclip
from slibtk import slibtk

logger = logging.getLogger(__name__)


@slibtk.sleep_after(secs_after=0.1)
@slibtk.log_output()
def read_clipboard() -> str:
    return pyperclip.paste().strip()


@slibtk.log_input()
def copy_to_clipboard(text: str) -> None:
    pyperclip.copy(text)


@slibtk.sleep_after(0.2)
def refocus() -> None:
    """use the hot key alt + tab to switch back to the previous gui, required at the start of any pyautowin script
    that is run from the run window"""
    pyautogui.hotkey('alt', 'tab')
