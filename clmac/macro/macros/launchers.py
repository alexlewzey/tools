"""
automates opening frequently used websites across multiple chrome browsers
"""
import time
import webbrowser
from pathlib import Path
from typing import Iterable

import pyperclip
import yaml

from clmac.config.definitions import URLS_YAML, LAUNCH_TXT

path = Path.cwd()

rest = 0.3


def open_urls(urls: Iterable[str]) -> None:
    """launches a list of urls with webdriver"""
    for url in urls:
        webbrowser.open(url)
        time.sleep(rest)
    time.sleep(rest)


def morning_sites() -> None:
    with LAUNCH_TXT.open() as f:
        urls = f.read().splitlines()
        open_urls(urls)


def guitar_practice():
    """open the guitar practice urls in browser"""
    with URLS_YAML.open('r') as f:
        urls = yaml.load(f)['guitar']
    open_urls(urls.values())


def clipboard2browser() -> None:
    """take a line separated list of urls from the clipboard and open each one in a separate tab"""
    cb = pyperclip.paste()
    for url in cb.splitlines():
        url = url if url.startswith('http') else 'http://' + url
        webbrowser.open(url)


