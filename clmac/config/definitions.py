"""a name space for program constants. Paths (dirs, files, executables), large string literals and personal info."""
import sys
from pathlib import Path

SRC: Path = Path(__file__).parent.parent
ROOT = SRC.parent
DIR_DATA = ROOT / 'data'
DIR_HELPERS = SRC / 'helpers'
DIR_CONFIG = SRC / 'config'

PERSONAL_YAML = DIR_CONFIG / 'personal.yaml'
NUMKEYS_YAML_0 = DIR_CONFIG / 'numkeys_0.yaml'
NUMKEYS_YAML_1 = DIR_CONFIG / 'numkeys_1.yaml'
URLS_YAML = DIR_CONFIG / 'urls.yaml'
LAUNCH_TXT = DIR_CONFIG / 'launch.txt'

FILE_CLIPBOARD_HISTORY: Path = DIR_DATA / 'clipboard_history.pk'

EXE_TESSERACT: str = (Path.home() / '/AppData/Local/Tesseract-OCR/tesseract.exe').as_posix()
driver_file: str = 'chromedriver.exe' if sys.platform == 'win32' else 'chromedriver'
EXE_CHROMEDRIVER: str = (ROOT / 'bins' / driver_file).as_posix()


