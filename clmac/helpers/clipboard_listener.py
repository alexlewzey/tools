import logging
import time
from threading import Thread
from typing import Any, List

import pyperclip

from clmac.config.definitions import FILE_CLIPBOARD_HISTORY
from clmac.helpers import core

logger = logging.getLogger(__name__)


class ClipboardHistoryFile:
    def __init__(self, max_length: int = 20):
        self.path = FILE_CLIPBOARD_HISTORY
        self.max_length = max_length

    @property
    def file(self) -> str:
        return str(self.path)

    def initialise_if_not_exist(self) -> None:
        """If a clipboard history file does not exist it creates one."""
        if not FILE_CLIPBOARD_HISTORY.is_file():
            core.write_pickle([], self.file)

    def load(self) -> List:
        """Load clipboard history flagging an error if it is not of type list."""
        try:
            core.history = core.read_pickle(self.file)
        except UnicodeError:
            history = []
            core.write_pickle(history, self.file)
        assert isinstance(history, list)
        return history

    def add(self, item: str) -> None:
        history = self.load()
        if len(history) > self.max_length:
            history = history[-self.max_length :]
        history.append(item)
        core.write_pickle(history, self.file)

    def clean(self) -> None:
        """Clean up the clipboard history file by removing duplicates, white spaces and
        writing over the original."""
        clipboard_history: List = self.load()
        try:
            clipboard_history_unique = list(dict.fromkeys(clipboard_history))[-10:]
        except IndexError:
            logger.debug(
                f"clipboard history contains less than 10 items: {clipboard_history}"
            )
            return None
        logger.debug(f"over-writing clean list: {clipboard_history_unique}")
        core.write_pickle(clipboard_history_unique, self.file)

    def get_menu_items(self) -> List[str]:
        return list(reversed(self.load()))


def clipboard_listener(wait: float = 1):
    """Periodically check if a new item has been added to the clipboard."""
    global clipboard_history_file
    history: List = clipboard_history_file.load()
    while True:
        clip: Any = pyperclip.paste()
        if clip not in history:
            clipboard_history_file.add(clip)
            history = clipboard_history_file.load()
        time.sleep(wait)


clipboard_history_file = ClipboardHistoryFile()
thread_clipboard_listener = Thread(target=clipboard_listener, daemon=True)
