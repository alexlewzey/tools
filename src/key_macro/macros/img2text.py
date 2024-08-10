import sys

import cv2
import numpy as np
import pyperclip
import pytesseract
from key_macro.core import EXE_TESSERACT
from PIL import ImageGrab


def img2text() -> None:
    """Convert image on clipboard to text and return to the clipboard command
    line args:

    nl: remove all new line characters from the return string
    """
    pil_img = ImageGrab.grabclipboard()
    if pil_img is not None:
        if sys.platform == "win32":
            pytesseract.pytesseract.tesseract_cmd = EXE_TESSERACT
        opencv_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        text = pytesseract.image_to_string(opencv_image).replace("`", "'")
        text = " ".join(text.split())
        pyperclip.copy(text)
    else:
        err_msg = "No image detected. Check your clipboard contains an image..."
        print(err_msg)


if __name__ == "__main__":
    img2text()
