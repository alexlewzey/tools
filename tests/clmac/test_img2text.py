from unittest.mock import patch

import pyperclip
from key_macro.macros import img2text
from PIL import Image


@patch("key_macro.macros.img2text.ImageGrab")
def test_img2text(image_grab_mock, capfd):
    expected = (
        "Remember, handling clipboard operations, especially with images, can be quite "
        "platform-specific. The above method works for Windows, but you'll need to "
        "adjust it if you're working on a different operating system."
    )
    pil_img = Image.open("images/img2text_example.png")
    image_grab_mock.grabclipboard.return_value = pil_img
    img2text.img2text()
    output = pyperclip.paste()
    assert expected == output
    image_grab_mock.grabclipboard.assert_called_once()


def test_img2text_invalid_input(capfd):
    text = "hello mole"
    pyperclip.copy(text)
    img2text.img2text()
    out, err = capfd.readouterr()
    assert out == "No image detected. Check your clipboard contains an image...\n"
    assert text == pyperclip.paste()
