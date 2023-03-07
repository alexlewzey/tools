from pathlib import Path
from typing import Generator, Tuple

from PIL import Image

GenPath = Generator[Path, None, None]
Rgb = Tuple[int, int, int]


def glob_search_dir(directory: str, glob: str) -> GenPath:
    """Return a generator of Path objects that match glob."""
    return Path(directory).glob(glob)


def show_colors(path: str) -> None:
    """Print all colors in the image."""
    pic = Image.open(path)
    for color in pic.getcolors():
        print(color)


def recolor_img(path: str, new_path: str, new_color: Rgb) -> None:
    """Recolor an image and save at new path."""
    pic = Image.open(path)
    width, height = pic.size
    for x in range(width):
        for y in range(height):
            rgba = pic.getpixel((x, y))
            if rgba != (255, 255, 255, 0):
                pic.putpixel((x, y), new_color)
    pic.save(new_path)


def make_dir(dir_new: str) -> Path:
    """Make a new dir and return path object."""
    new_dir = Path(".") / dir_new
    new_dir.mkdir(exist_ok=True)
    return new_dir


def batch_recolor(src: str, color: Rgb) -> None:
    """Take all images in a folder, recolor them and save them in a new
    directory."""
    globed = glob_search_dir(src, "*.png")
    rgb_str = "_".join([str(n) for n in color])
    dst = make_dir(rgb_str)

    for path in globed:
        new_file_nm = rgb_str + path.name
        new_dst = str(dst / new_file_nm)
        recolor_img(path=path.name, new_path=new_dst, new_color=color)


if __name__ == "__main__":
    new_color = (246, 119, 192)
    batch_recolor("..", new_color)
