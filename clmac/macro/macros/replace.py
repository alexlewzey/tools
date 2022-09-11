#! /usr/bin/env python3
"""

"""

import logging
from typing import List

import pyperclip

from clmac.helpers.gui_prompts import GuiTextTwoBoxes
from clmac.helpers.typer import Typer

typer = Typer()

logger = logging.getLogger(__name__)

prompts = ['find: ', 'replace: ']


def find_replace(pair: List[str]):
    selection: str = pyperclip.paste()
    logger.info(f'selection: {selection}')
    replaced = selection.replace(pair[0], pair[1])
    logger.info(f'replaced: {replaced}')
    typer.type_text(replaced)


gui_text_boxes = GuiTextTwoBoxes('find_replace', prompts=prompts, text_proc=find_replace)

if __name__ == '__main__':
    gui_text_boxes.window.mainloop()
