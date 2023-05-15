#! /usr/bin/env python3
"""When activated will periodically check if any new items have been added to the
clipboard and add them to a list when the user ends the program by pressing esc all the
collected clipboard items will be added to the clipboard."""

import logging
import time
from threading import Thread
from typing import Any, List

import pyperclip
from pynput.keyboard import Key, Listener

logger = logging.getLogger(__name__)

clipboard_history: List = []


def output_to_clipboard(items: List) -> None:
    """Convert the output list to a string and add to the clipboard."""
    output = "\n".join([item.strip() for item in items])
    pyperclip.copy(output)


def clipboard_listener(wait: float = 1):
    """Periodically check if a new item has been added to the clipboard."""
    global clipboard_history
    while True:
        clip: Any = pyperclip.paste()
        if clip not in clipboard_history:
            clipboard_history.append(clip)
            print(f"added item: {clipboard_history}")
        time.sleep(wait)


def on_press(key):
    global clipboard_history
    logger.info(key)
    if Key.esc == key:
        output_to_clipboard(clipboard_history)
        raise SystemExit


def on_release(key):
    pass


def run():
    thread_cb_listener = Thread(target=clipboard_listener, daemon=True)
    thread_cb_listener.start()
    with Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
