"""Automate the mouse to rotate a 3d plotly figure at a constant speed."""

import pyautogui

start = (390.0, 989)
top = (323.0, 246.0)

pyautogui.position()
pyautogui.click()
pyautogui.moveTo(*start)
pyautogui.drag(0.0, -50.0, duration=0.5, button="left")
pyautogui.moveTo(*top)
for _ in range(4):
    pyautogui.drag(600.0, 0.0, duration=2.5, button="left", _pause=False)
    pyautogui.move(-600.0, 0.0, _pause=False)
