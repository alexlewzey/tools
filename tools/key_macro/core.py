""""""
import json
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
DIR_CONFIG = Path.home() / ".key_macro"
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
