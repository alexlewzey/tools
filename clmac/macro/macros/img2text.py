import sys

import cv2
import numpy as np
import pyperclip
import pytesseract
from PIL import ImageGrab

from clmac.config.definitions import EXE_TESSERACT


def img2text():
    """convert image on clipboard to text and return to the clipboard command
    line args:

    nl: remove all new line characters from the return string
    """
    pil_img = ImageGrab.grabclipboard()
    if sys.platform == "win32":
        pytesseract.pytesseract.tesseract_cmd = EXE_TESSERACT
    try:
        opencvImage = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except TypeError:
        err_msg = "Only accepts images from the clipboard. Check your clipboard contains an image..."
        print(err_msg)
        input("\nPress enter key to continue...")
        raise TypeError(err_msg)

    text = pytesseract.image_to_string(opencvImage).replace("`", "'")

    assert isinstance(text, str)
    pyperclip.copy(text)


if __name__ == "__main__":
    img2text()
