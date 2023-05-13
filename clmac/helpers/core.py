""""""
import functools
import inspect
import logging
import pickle
import time
from pathlib import Path
from typing import *

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
    level=logging.DEBUG,
)

dir_src = Path(__file__).parent.parent
dir_config = dir_src / "config"

file_custom_0 = dir_config / "custom_0.py"
file_custom_1 = dir_config / "custom_1.py"

numbers = "one,two,three,four,five,six,seven,eight,nine".split(",")
if not file_custom_0.exists():
    with file_custom_0.open("w") as f:
        for number in numbers:
            f.write(f'{number} = ""\n')
if not file_custom_1.exists():
    with file_custom_1.open("w") as f:
        for number in numbers:
            f.write(f'{number} = ""\n')


def hr_secs(secs: float) -> str:
    """Format seconds human readable format hours:mins:seconds."""
    secs_per_hour: int = 3600
    secs_per_min: int = 60
    hours, remainder = divmod(secs, secs_per_hour)
    mins, seconds = divmod(remainder, secs_per_min)

    return f"{int(hours):02}:{int(mins):02}:{seconds:05.2f}"


def whitespacer(s):
    return " ".join(s.split())


def log_input(positional_input_index: int = 0, kw_input_key: Optional[Hashable] = None):
    """Logs the input (first positional argument) and output of decorated
    function, you can specify a specific kw arg to be logged as input by
    specifying its corresponding param key."""

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
    """Logs the input (first positional argument) and output of decorated
    function, you can specify a specific kw arg to be logged as input by
    specifying its corresponding param key."""

    def outer_wrapper(func):
        @functools.wraps(func)
        def inner_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            logger.info(f"{func.__name__}: output={result}"[:300])
            return result

        return inner_wrapper

    return outer_wrapper


def log_input_and_output(
    input_flag=True,
    output_flag=True,
    positional_input_index: int = 0,
    kw_input_key: Optional[Hashable] = None,
):
    """Logs the input (first positional argument) and output of decorated
    function, you can specify a specific kw arg to be logged as input by
    specifying its corresponding param key."""

    def outer_wrapper(func):
        @functools.wraps(func)
        def inner_wrapper(*args, **kwargs):
            if input_flag:
                if kw_input_key:
                    # noinspection PyTypeChecker
                    input_arg = kwargs[kw_input_key]
                else:
                    input_arg = _get_positional_arg(
                        args, kwargs, positional_input_index
                    )
                logger.info(f"{func.__name__}: input={input_arg}"[:300])

            result = func(*args, **kwargs)
            if output_flag:
                logger.info(f"{func.__name__}: output={result}"[:300])
            return result

        return inner_wrapper

    return outer_wrapper


def is_iterable(o: Any) -> bool:
    try:
        iter(o)
    except TypeError:
        return False
    else:
        return True


def _get_positional_arg(args, kwargs, index: int = 0) -> Any:
    """Returns the first positional arg if there are any, if there are only kw
    args it returns the first kw arg."""
    try:
        input_arg = args[index]
    except KeyError:
        input_arg = list(kwargs.values())[index]
    return input_arg


def log_func():
    """Logs the input (first positional argument) and output of decorated
    function, you can specify a specific kw arg to be logged as input by
    specifying its corresponding param key."""

    def outer_wrapper(func):
        @functools.wraps(func)
        def inner_wrapper(*args, **kwargs):
            logger.info(f"calling: {func.__name__}")
            result = func(*args, **kwargs)
            logger.info(f"completed: {func.__name__}")
            return result

        return inner_wrapper

    return outer_wrapper


def log_func_start():
    logger.info(f"start {inspect.stack()[1][3]}")


def log_func_end():
    logger.info(f"end {inspect.stack()[1][3]}")


def sleep_before(secs_before: float):
    """Call the sleep function before the decorated function is called."""
    return sleep_before_and_after(secs_before=secs_before, secs_after=0)


def sleep_after(secs_after: float):
    """Call the sleep function after the decorated function is called."""
    return sleep_before_and_after(secs_before=0, secs_after=secs_after)


def sleep_before_and_after(secs_before: float = 0, secs_after: float = 0):
    """call the sleep method before and after the decorated function is called,
    pass in the sleep duration in seconds.

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


def timer(func):
    """Decorator that logs the time taken for the decorated func to run."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start: float = time.time()
        result = func(*args, **kwargs)
        hr_time_elapsed: str = hr_secs(time.time() - start)
        logger.info(f"time taken {func.__name__}: {hr_time_elapsed}")
        return result

    return wrapper


def runtimes(arg_values: Sequence):
    """Decorator that records the runtime (seconds) for several values of a
    single argument that is passed to the decorated func, returning the
    argument: second pairs in a dictionary."""

    def outer_wrapper(func: Callable):
        @functools.wraps(func)
        def inner_wrapper(*args, **kwargs):
            logger.info(
                f"monitoring runtimes for func={func.__name__}, values={arg_values}"
            )
            times = {}
            for value in arg_values:
                start = time.time()
                func(value, *args, **kwargs)
                seconds = time.time() - start
                times[value] = seconds
                logger.info(f"param={value} seconds={seconds}")

            return times

        return inner_wrapper

    return outer_wrapper


@timer
@log_input()
def write_pickle(obj, path: Optional[Union[Path, str]]) -> None:
    """Write object to a pickle on your computer."""
    path = Path(path)
    with path.open("wb") as f:
        pickle.dump(obj, f)


@timer
@log_output()
def read_pickle(path: Optional[Union[Path, str]]):
    """Return stored object from a pickle file."""
    path = Path(path)
    logger.info(f"reading pickle: {path}")
    with path.open("rb") as f:
        obj = pickle.load(f)
    return obj
