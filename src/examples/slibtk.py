"""Standard library tool-kit this module contains commonly used functions to process
and manipulate standard library objects."""
import logging
import time
from collections.abc import Iterator, Sequence
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def to_chunks(seq: Sequence, size: int) -> Iterator[list]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def timestamp() -> datetime:
    return datetime.now().replace(microsecond=0).isoformat()


def hr_seconds(secs: float) -> str:
    """Format seconds human readable format hours:mins:seconds."""
    secs_per_hour: int = 3600
    secs_per_min: int = 60
    hours, remainder = divmod(secs, secs_per_hour)
    mins, seconds = divmod(remainder, secs_per_min)

    return f"{int(hours):02}:{int(mins):02}:{seconds:05.2f}"


def hr_nanoseconds(ns: int):
    secs, rem = divmod(ns, 1000**3)
    ms, rem = divmod(rem, 1000**2)
    us, ns = divmod(rem, 1000)
    return f"{secs}s:{ms}ms:{us}us:{ns}ns"


def hr_secs_elapsed(start: float) -> str:
    """Format seconds from elapsed since start in human readable format
    hours:mins:seconds."""
    return hr_seconds(time.time() - start)


def max_fname(path: Path | str | None) -> Path:
    """Return the path with the largest numeric index in its name."""
    path = Path(path)
    parent, stem, suffix = path.parent, path.stem, path.suffix
    i = 0
    while True:
        p = parent / f"{stem}_{i:02}{suffix}"
        i += 1
        if not p.exists():
            break
    return p


def next_fname(path: Path | str | None, zfill: int = 2) -> Path:
    """Return next incremental file that does not exist
    (path.root)_{next_num}.(path.suffix)"""
    path = Path(path)
    parent, stem, suffix = path.parent, path.stem, path.suffix
    i = 0
    while (parent / f"{stem}_{str(i).zfill(zfill)}{suffix}").exists():
        i += 1
    return parent / f"{stem}_{str(i).zfill(zfill)}{suffix}"


def rtype(x):
    if isinstance(x, list):
        return [rtype(o) for o in x]
    elif isinstance(x, tuple):
        return tuple(rtype(o) for o in x)
    elif isinstance(x, dict):
        return {k: rtype(v) for k, v in x.items()}
    else:
        type_str = str(type(x)).split("'")[1]
        return type_str


def get_most_recently_modified_file(path: Path | str, glob: str) -> Path:
    most_recent_time = 0
    most_recent_modified_file = None
    for p in Path(path).rglob(glob):
        if p.stat().st_mtime > most_recent_time:
            most_recent_time = p.stat().st_mtime
            most_recent_modified_file = p
    return most_recent_modified_file
