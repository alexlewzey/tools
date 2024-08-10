import sys

import cv2
import numpy as np
import pyperclip
import pytesseract
from PIL import Image, ImageGrab

from tools.key_macro.core import EXE_TESSERACT


def _img2text(img: Image.Image) -> str:
    if sys.platform == "win32":
        pytesseract.pytesseract.tesseract_cmd = EXE_TESSERACT
    opencv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    text = pytesseract.image_to_string(opencv_image).replace("`", "'")
    text = " ".join(text.split())
    return text


def img2text() -> None:
    """Convert image on clipboard to text and return to the clipboard command
    line args:

    nl: remove all new line characters from the return string
    """
    img = ImageGrab.grabclipboard()
    if isinstance(img, Image.Image):
        text = _img2text(img)
        pyperclip.copy(text)
    else:
        err_msg = f"No image detected. Clipboard={type(img)} - {img}"
        print(err_msg)


if __name__ == "__main__":
    img2text()
