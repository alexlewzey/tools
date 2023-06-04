"""All my main macro and commonly used python scripts are run using a keyboard listener
that recognises sequential keystrokes that typically start with a semicolon.

note: do not call your module macro as that name is already taken in the path
"""
import logging
import subprocess
import time
from typing import Tuple

from pynput.keyboard import Key, KeyCode, Listener

from .callables import ENCODINGS, test_for_duplicates

from . import core
from . import typer as lh

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
    level=logging.DEBUG,
)

MAX_KEY_HISTORY_LENGTH: int = 30
typer = lh.Typer()


class KeyHistory(list):
    """A holder of fixed length for Key and KeyCode types passed from the on_press
    listener."""

    def __init__(self, max_length: int = 20):
        super().__init__()
        self.size_max = max_length

    def shift_que(self, item):
        self.append(item)
        self.pop(0)

    def add_key(self, key):
        if len(self) >= self.size_max:
            self.shift_que(key)
        else:
            self.append(key)


class GlobalInputs:
    CTRL_SHIFT_Q = (Key.ctrl_l, Key.shift, KeyCode(char="q"))
    CTRL_SHIFT_R = (Key.ctrl_l, Key.shift, KeyCode(char="r"))
    CTRL_C = (Key.ctrl_l, KeyCode(char="\x03"))

    currently_pressed = set()
    key_history = KeyHistory(MAX_KEY_HISTORY_LENGTH)
    encoding_lookup = dict([macro.get_set_func_pair() for macro in ENCODINGS])


def listen_for_encoding() -> None:
    """Checks if the last three keys that were typed exist in the macro encoding
    indexes, if there are it indexes and calls the function corresponding to that three
    char encoding."""
    last_three: Tuple = tuple(GlobalInputs.key_history[-3:])
    if last_three == GlobalInputs.CTRL_SHIFT_Q:
        raise SystemExit
    elif last_three == GlobalInputs.CTRL_SHIFT_R:
        restart_app()
    elif last_three in GlobalInputs.encoding_lookup.keys():
        typer.press_key(key=Key.backspace, num_presses=len(last_three))
        GlobalInputs.encoding_lookup[last_three]()


def restart_app() -> None:
    subprocess.run("clmac kel run", shell=True)
    time.sleep(0.2)
    quit()


@core.log_input()
def on_press(key):
    GlobalInputs.key_history.add_key(key)
    GlobalInputs.currently_pressed.add(key)
    listen_for_encoding()


def on_release(key):
    try:
        GlobalInputs.currently_pressed.remove(key)
    except KeyError:
        logger.debug(f"failed to remove: {key}")


def run():
    """Run the macro script."""
    test_for_duplicates()
    with Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


if __name__ == "__main__":
    run()
