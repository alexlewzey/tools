""""""
import functools
import json
import logging
import time
import webbrowser
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
    level=logging.DEBUG,
)

SRC: Path = Path(__file__).parent
DIR_CONFIG = Path.home() / ".clmac"
DIR_CONFIG.mkdir(exist_ok=True)
PERSONAL_JSON = DIR_CONFIG / "personal.json"
CUSTOM_JSON = DIR_CONFIG / "custom.json"
URLS = DIR_CONFIG / "urls.txt"
URLS.touch(exist_ok=True)
NUMBERS = [str(i) for i in range(1, 10)]

EXE_TESSERACT: str = (
    Path.home() / "/AppData/Local/Tesseract-OCR/tesseract.exe"
).as_posix()


def open_urls() -> None:
    with URLS.open() as f:
        for url in f.read().splitlines():
            webbrowser.open(url.strip())


def create_custom_template() -> None:
    custom_template = dict(zip(NUMBERS, [""] * len(NUMBERS)))
    if not CUSTOM_JSON.exists():
        with CUSTOM_JSON.open("w") as f:
            f.write(json.dumps(custom_template, indent=4))


def create_personal_template() -> None:
    personal_template = {
        "gmail": "",
        "hotmail": "",
        "work_mail": "",
        "mobile": "",
        "name": "",
        "username": "",
        "address": "",
    }
    if not PERSONAL_JSON.exists():
        with PERSONAL_JSON.open("w") as f:
            f.write(json.dumps(personal_template, indent=4))


def update_custom_template(key: str, value: str) -> None:
    with CUSTOM_JSON.open() as f:
        content = json.loads(f.read())
    if key in NUMBERS:
        content[key] = value
    else:
        return
    with CUSTOM_JSON.open("w") as f:
        f.write(json.dumps(content, indent=4))


def read_custom_template() -> str:
    with CUSTOM_JSON.open() as f:
        return f.read()


def log_input(positional_input_index: int = 0, kw_input_key: str | None = None):
    """Logs the input (first positional argument) and output of decorated function,
    you can specify a specific kw arg to be logged as input by specifying its
    corresponding param key."""

    def outer_wrapper(func):
        @functools.wraps(func)
        def inner_wrapper(*args, **kwargs):
            if kw_input_key:
                input_arg = kwargs[kw_input_key]
            else:
                input_arg = _get_positional_arg(args, kwargs, positional_input_index)
            logger.info(f"{func.__name__}: input={input_arg}"[:300])

            result = func(*args, **kwargs)
            return result

        return inner_wrapper

    return outer_wrapper


def log_output():
    """Logs the input (first positional argument) and output of decorated function,
    you can specify a specific kw arg to be logged as input by specifying its
    corresponding param key."""

    def outer_wrapper(func):
        @functools.wraps(func)
        def inner_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            logger.info(f"{func.__name__}: output={result}"[:300])
            return result

        return inner_wrapper

    return outer_wrapper


def _get_positional_arg(args, kwargs, index: int = 0) -> Any:
    """Returns the first positional arg if there are any, if there are only kw args
    it returns the first kw arg."""
    try:
        input_arg = args[index]
    except KeyError:
        input_arg = list(kwargs.values())[index]
    return input_arg


def sleep_after(secs_after: float):
    """Call the sleep function after the decorated function is called."""
    return sleep_before_and_after(secs_before=0, secs_after=secs_after)


def sleep_before_and_after(secs_before: float = 0, secs_after: float = 0):
    """Call the sleep method before and after the decorated function is called, pass
    in the sleep duration in seconds.

    Default values are 0.
    """

    def outer_wrapper(func):
        @functools.wraps(func)
        def inner_wrapper(*args, **kwargs):
            if secs_before:
                time.sleep(secs_before)
            result = func(*args, **kwargs)
            if secs_after:
                time.sleep(secs_after)
            return result

        return inner_wrapper

    return outer_wrapper
