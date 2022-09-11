#! /usr/bin/env python3
"""
todo:
    - clean up this crap code
    - properly format the menu
    - add a scroll bar
    - include images somehow
"""
import time
import tkinter as tk
from typing import List, Dict, Tuple

import pyperclip
from pynput.keyboard import KeyCode

from clmac.helpers import typer as lh
from clmac.helpers.clipboard_listener import clipboard_history_file


def create_clipboard_index_menu(clipboard_history: List[str], max_length: int = 20) -> Dict[
    Tuple[KeyCode, KeyCode], str]:
    """
    create mapping of pynput keycodes to the most recent items in the clipboard history

    Parameters
    ----------
    max_length
    clipboard_history
        most recent clipboard items
    Returns
    -------
        keycode: clipboard item keypair mapping
    """
    menu = {}
    history_len: int = max_length if len(clipboard_history) > max_length else len(clipboard_history)
    clipboard_history = reversed(clipboard_history[-history_len:])

    for i, clip in enumerate(clipboard_history):
        if i < 10:
            menu[(KeyCode(char=str(0)), KeyCode(char=str(i)))] = clip
        elif i < 100:
            i_str = str(i)
            menu[(KeyCode(char=i_str[0]), KeyCode(char=i_str[1]))] = clip
    return menu


def format_clip_for_menu(clip: str, n_lines=2, max_char=80) -> str:
    """grab the first n lines from a clip"""
    return '\n'.join(clip.split('\n')[:n_lines])[:max_char]


def format_menu_contents(menu) -> str:
    """print clipboard menu to the terminal"""
    menu_lines = [f'{str(key)}'.ljust(10) + f'{format_clip_for_menu(clip)}' for key, clip in menu.items()]
    return '\n'.join(menu_lines)


user_input = None


def submit_text(destroy_time: float = 3):
    user_input = user_text.get()
    user_key = tuple([KeyCode(char=user_input[0]), KeyCode(char=user_input[1])])
    global window
    window.destroy()
    time.sleep(destroy_time)
    pyperclip.copy(menu_clipboard[user_key])
    typer.type_text(menu_clipboard[user_key])


typer = lh.Typer()
clipboard_items = clipboard_history_file.load()
menu_clipboard = create_clipboard_index_menu(clipboard_items)

window = tk.Tk()
window.title('clipboard_history')
window.configure(background='black')
user_text = tk.Entry(window, width=20, bg='white')
user_text.grid(row=0, column=0, sticky=tk.W)

first_index: int = 1
for keys, clip in menu_clipboard.items():
    tk.Label(window, text=str(keys), background='white').grid(row=first_index, column=0, sticky=tk.W)
    tk.Label(window, text=clip, background='white').grid(row=first_index, column=1, sticky=tk.W)
    first_index += 1

tk.Button(window, text='submit', command=submit_text).grid(row=first_index, column=0, sticky=tk.W)
window.mainloop()
