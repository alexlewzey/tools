#! /usr/bin/env python3
"""Add the rgb of current mouse cursor position to the clipboard, default rgb
will return hex code if hex is passed as cmdline arg."""
import time

import pyautogui

from clmac.helpers import core


@core.log_output()
def get_pixel_rgb():
    return pyautogui.pixel(*pyautogui.position())


def rgb2hex(r: int, g: int, b: int) -> str:
    return f"#{r:02X}{g:02X}{b:02X}"


def main():
    while True:
        color = get_pixel_rgb()
        print(color, end="\r")
        time.sleep(0.2)


if __name__ == "__main__":
    main()
