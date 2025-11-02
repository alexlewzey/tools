from PIL import Image

from src.key_macro.macros.img2text import _img2text


def test_img2text():
    expected = (
        "Remember, handling clipboard operations, especially with images, can be quite "
        "platform-specific. The above method works for Windows, but you'll need to "
        "adjust it if you're working on a different operating system."
    )
    img = Image.open("images/img2text_example.png")
    output = _img2text(img)
    assert expected == output
