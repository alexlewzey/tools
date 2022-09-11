#! /usr/bin/env python3

import time

from clmac.helpers import automation
import pyautogui

rest = 0.3


def repeat_key(key: str, i: int):
    for i in range(i):
        pyautogui.hotkey(key)
        time.sleep(rest)


def set_pagelayout() -> None:
    pyautogui.hotkey('alt', 'f')
    time.sleep(0.3)
    repeat_key('up', 2)

    pyautogui.hotkey('enter')
    time.sleep(rest)

    repeat_key('tab', 3)

    pyautogui.hotkey('a')
    time.sleep(rest)

    repeat_key('tab', 2)

    for i in range(4):
        pyautogui.typewrite('0.1')
        pyautogui.hotkey('tab')
        time.sleep(rest)

    repeat_key('tab', 2)
    pyautogui.hotkey('enter')


def move_mouse_out_the_way() -> None:
    """move mouse to a safe location in the bottom right right of the screen"""
    position_xy = [x * 0.9 for x in pyautogui.size()]
    pyautogui.moveTo(position_xy[0], position_xy[1])


def main():
    """
    automate setting the page layout in google doc to be as big as it can
    """
    automation.refocus()
    move_mouse_out_the_way()
    set_pagelayout()


if __name__ == '__main__':
    main()
