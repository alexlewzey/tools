""""""
import functools
import inspect
import logging
import pickle
import time
import sys
from pathlib import Path
from typing import Any, Callable, Hashable, Optional, Sequence, Union

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
    level=logging.DEBUG,
)

SRC: Path = Path(__file__).parent.parent
ROOT = SRC.parent
DIR_DATA = ROOT / "data"
DIR_HELPERS = SRC / "helpers"
DIR_CONFIG = SRC / "config"

PERSONAL_YAML = DIR_CONFIG / "personal.yaml"
NUMKEYS_YAML_0 = DIR_CONFIG / "numkeys_0.yaml"
NUMKEYS_YAML_1 = DIR_CONFIG / "numkeys_1.yaml"
URLS_YAML = DIR_CONFIG / "urls.yaml"
LAUNCH_TXT = DIR_CONFIG / "launch.txt"

FILE_CLIPBOARD_HISTORY: Path = DIR_DATA / "clipboard_history.pk"

EXE_TESSERACT: str = (
    Path.home() / "/AppData/Local/Tesseract-OCR/tesseract.exe"
).as_posix()
driver_file: str = "chromedriver.exe" if sys.platform == "win32" else "chromedriver"
EXE_CHROMEDRIVER: str = (ROOT / "bins" / driver_file).as_posix()



dir_src = Path(__file__).parent
dir_config = dir_src / "config"

file_custom_0 = dir_config / "custom_0.py"
file_custom_1 = dir_config / "custom_1.py"

numbers = "one,two,three,four,five,six,seven,eight,nine".split(",")
if not file_custom_0.exists():
    print("custom_0.py not found, creating...")
    with file_custom_0.open("w") as f:
        for number in numbers:
            f.write(f'{number} = ""\n')
if not file_custom_1.exists():
    print("custom_1.py not found, creating...")
    with file_custom_1.open("w") as f:
        for number in numbers:
            f.write(f'{number} = ""\n')




def log_input(positional_input_index: int = 0, kw_input_key: Optional[Hashable] = None):
    """Logs the input (first positional argument) and output of decorated function, you
    can specify a specific kw arg to be logged as input by specifying its corresponding
    param key."""

    def outer_wrapper(func):
        @functools.wraps(func)
        def inner_wrapper(*args, **kwargs):
            if kw_input_key:
                # noinspection PyTypeChecker
                input_arg = kwargs[kw_input_key]
            else:
                input_arg = _get_positional_arg(args, kwargs, positional_input_index)
            logger.info(f"{func.__name__}: input={input_arg}"[:300])

            result = func(*args, **kwargs)
            return result

        return inner_wrapper

    return outer_wrapper


def log_output():
    """Logs the input (first positional argument) and output of decorated function, you
    can specify a specific kw arg to be logged as input by specifying its corresponding
    param key."""

    def outer_wrapper(func):
        @functools.wraps(func)
        def inner_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            logger.info(f"{func.__name__}: output={result}"[:300])
            return result

        return inner_wrapper

    return outer_wrapper



def _get_positional_arg(args, kwargs, index: int = 0) -> Any:
    """Returns the first positional arg if there are any, if there are only kw args it
    returns the first kw arg."""
    try:
        input_arg = args[index]
    except KeyError:
        input_arg = list(kwargs.values())[index]
    return input_arg




def sleep_after(secs_after: float):
    """Call the sleep function after the decorated function is called."""
    return sleep_before_and_after(secs_before=0, secs_after=secs_after)


def sleep_before_and_after(secs_before: float = 0, secs_after: float = 0):
    """Call the sleep method before and after the decorated function is called, pass in
    the sleep duration in seconds.

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

