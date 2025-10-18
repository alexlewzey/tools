""""""

import logging
import webbrowser
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
    level=logging.DEBUG,
)

SRC: Path = Path(__file__).parent
ROOT = SRC.parent.parent
DIR_CONFIG = Path.home() / ".key_macro"
DIR_CONFIG.mkdir(exist_ok=True)
URLS = DIR_CONFIG / "urls.txt"
URLS.touch(exist_ok=True)

EXE_TESSERACT: str = (
    Path.home() / "/AppData/Local/Tesseract-OCR/tesseract.exe"
).as_posix()


def open_urls() -> None:
    with URLS.open() as f:
        for url in f.read().splitlines():
            webbrowser.open(url.strip())
