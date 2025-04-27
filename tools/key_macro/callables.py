"""A collection of all my available macro."""

import json
import sys
from collections import Counter
from collections.abc import Callable
from functools import partial
from pathlib import Path

from pynput.keyboard import KeyCode

from tools.key_macro.core import CUSTOM_JSON, PERSONAL_JSON
from tools.key_macro.keyboard import Typer
from tools.key_macro.macros import formatters, img2text

git_log: str = (
    "git log --graph --abbrev-commit --decorate --format=format:'%C(bold blue)%h%"
    "C(reset) - %C(bold green)(%ar)%C(reset) %C(white)%s%C(reset) %C(dim white)"
    "- %an%C(reset)%C(auto)%d%C(reset)' --all"
)
please_info: str = "Please let me know if you require any more information."
any_help: str = "Any help would be much appreciated."
please_queries: str = "Please let me know if you have any queries."
python_imports: str = """from pathlib import Path
import functools
import os
import io
import gc
import re
import hashlib
import random
from typing import *
import string
from datetime import date, datetime, timedelta
import collections
import math
import itertools
import pickle
import shutil
import json
import logging
import subprocess
import sys
import base64
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', '{:.3f}'.format)
np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=3, sci_mode=False)
"""


class MacroEncoding:
    """Represents a specific macro including its name, callable functionality and the
    keyboard encoding that triggers it."""

    def __init__(self, encoding: str, func: Callable):
        self.encoding = encoding
        self.func = func
        self.encode_set = self.get_encoding_set()

    def get_encoding_set(self) -> tuple:
        """Get a pynputs representation of the keyboard encoding."""
        return tuple(KeyCode(char=char) for char in self.encoding)

    def get_set_func_pair(self) -> tuple[tuple, Callable]:
        """Return a tuple containing the pynputs encoding set and the callable
        functionality."""
        return self.encode_set, self.func


def load_json(path: Path) -> dict:
    with path.open() as f:
        settings = json.loads(f.read())
    return settings


def type_config(config_path: Path, key: str) -> None:
    settings = load_json(config_path)
    typer.type_text(settings[key].replace("\\n", "\n"))


type_personal = partial(type_config, config_path=PERSONAL_JSON)
type_custom = partial(type_config, config_path=CUSTOM_JSON)


typer = Typer()


ENCODINGS = [
    # PERSONAL #########################################################################
    MacroEncoding(encoding=";hm", func=partial(type_personal, key="hotmail")),
    MacroEncoding(encoding=";gm", func=partial(type_personal, key="gmail")),
    MacroEncoding(encoding=";wm", func=partial(type_personal, key="work_mail")),
    MacroEncoding(encoding=";al", func=partial(type_personal, key="name")),
    MacroEncoding(encoding=";mb", func=partial(type_personal, key="mobile")),
    MacroEncoding(encoding=";un", func=partial(type_personal, key="username")),
    MacroEncoding(encoding=";ad", func=partial(type_personal, key="address")),
    # CUSTOM ###########################################################################
    MacroEncoding(encoding=";;1", func=partial(type_custom, key="1")),
    MacroEncoding(encoding=";;2", func=partial(type_custom, key="2")),
    MacroEncoding(encoding=";;3", func=partial(type_custom, key="3")),
    MacroEncoding(encoding=";;4", func=partial(type_custom, key="4")),
    MacroEncoding(encoding=";;5", func=partial(type_custom, key="5")),
    MacroEncoding(encoding=";;6", func=partial(type_custom, key="6")),
    MacroEncoding(encoding=";;7", func=partial(type_custom, key="7")),
    MacroEncoding(encoding=";;8", func=partial(type_custom, key="8")),
    MacroEncoding(encoding=";;9", func=partial(type_custom, key="9")),
    # EMAILS ###########################################################################
    MacroEncoding(encoding=";tf", func=typer("Thanks for your email. ")),
    MacroEncoding(encoding=";ah", func=typer.partial_paste(any_help)),
    MacroEncoding(encoding=";ba", func=typer("\nBest\nAlex")),
    MacroEncoding(encoding=";mt", func=typer("\n\nMany thanks\n\nAlex")),
    # PYTHON ###########################################################################
    # MacroEncoding(encoding=";ll", func=typer("label_column")),
    # MacroEncoding(encoding=";vv", func=typer("value_column")),
    # MacroEncoding(encoding=";dd", func=typer("dim_column")),
    MacroEncoding(encoding=";zs", func=typer("~/.zshrc")),
    MacroEncoding(encoding=";nn", func=typer(".notnull().mean()")),
    MacroEncoding(encoding=";;;", func=typer("print()", 1)),
    MacroEncoding(encoding=";cc", func=typer(".columns")),
    MacroEncoding(encoding=";tt", func=typer("torch.")),
    MacroEncoding(encoding=";ns", func=typer("nvidia-smi dmon")),
    # MacroEncoding(encoding=";dd", func=typer(".dtypes")),
    MacroEncoding(encoding=";ss", func=typer(".shape")),
    MacroEncoding(encoding=";ii", func=typer("def __init__(self, ):", 2)),
    MacroEncoding(encoding=";ri", func=typer(".reset_index()")),
    MacroEncoding(encoding=";si", func=typer(".set_index()", 1)),
    MacroEncoding(encoding=";hd", func=typer(".head(9)")),
    MacroEncoding(encoding=";as", func=typer("descending=True")),
    MacroEncoding(encoding=";td", func=typer(" -> pl.DataFrame:")),
    MacroEncoding(encoding=";nm", func=typer("if __name__ == '__main__':\n    ")),
    MacroEncoding(encoding=";sv", func=typer(".sort()", 1)),
    MacroEncoding(encoding=";om", func=typer("1000000")),
    MacroEncoding(encoding=";mo", func=typer("1_000_000")),
    MacroEncoding(encoding=";vc", func=typer(".value_counts(sort=True)")),
    MacroEncoding(encoding=";mu", func=typer(".info(memory_usage='deep')")),
    MacroEncoding(encoding=";tn", func=typer(" -> None:")),
    MacroEncoding(encoding=";ae", func=typer("source .venv/bin/activate")),
    MacroEncoding(
        encoding=";nw", func=typer("dt.datetime.now().replace(microsecond=0)")
    ),
    MacroEncoding(encoding=";pi", func=typer(python_imports)),
    MacroEncoding(encoding=";rr", func=formatters.format_repr),
    MacroEncoding(encoding=";jp", func=formatters.join_python_string),
    MacroEncoding(encoding=";lt", func=formatters.to_list),
    MacroEncoding(encoding=";fv", func=formatters.format_variables),
    MacroEncoding(encoding=";bk", func=formatters.format_black),
    MacroEncoding(encoding=";2r", func=formatters.imports_to_requirements),
    MacroEncoding(encoding=";ws", func=formatters.wrap_string_literal),
    # DOCKER ###########################################################################
    MacroEncoding(encoding=";cr", func=typer("docker container ")),
    MacroEncoding(encoding=";ie", func=typer("docker image ")),
    MacroEncoding(encoding=";cm", func=typer("cat makefile")),
    # SQL/BIGQUERY #####################################################################
    MacroEncoding(encoding=";ct", func=typer("create or replace table  as", 3)),
    MacroEncoding(encoding=";c8", func=typer("count(*)/1000000 n,")),
    # MacroEncoding(encoding=";cp", func=typer("count(*) / sum(count(*)) over() pct,")),
    MacroEncoding(encoding=";ob", func=typer("order by ")),
    MacroEncoding(encoding=";bo", func=typer("order by  desc", 5)),
    MacroEncoding(encoding=";gb", func=typer("group by ")),
    MacroEncoding(encoding=";cd", func=typer("current_date()")),
    MacroEncoding(encoding=";we", func=typer("where ")),
    MacroEncoding(encoding=";st", func=typer("select ")),
    MacroEncoding(encoding=";dd", func=typer("distinct ")),
    MacroEncoding(encoding=";rn", func=typer("row_number() ")),
    MacroEncoding(encoding=";pb", func=typer("partition by ")),
    MacroEncoding(encoding=";ua", func=typer("union all ")),
    MacroEncoding(encoding=";ll", func=typer("limit 1000")),
    MacroEncoding(encoding=";dt", func=formatters.sql_count_distinct),
    MacroEncoding(encoding=";sm", func=formatters.sql_sum),
    MacroEncoding(encoding=";cp", func=formatters.sql_counts_dist),
    MacroEncoding(encoding=";ci", func=formatters.sql_count_if_not_null),
    MacroEncoding(encoding=";sf", func=formatters.select_from_table),
    MacroEncoding(encoding=";gt", func=formatters.get_table_name),
    MacroEncoding(encoding=";sq", func=formatters.format_sql),
    MacroEncoding(encoding=";bq", func=formatters.bq_to_python),
    MacroEncoding(encoding=";fs", func=typer("select * \nfrom ", paste=False)),
    MacroEncoding(encoding=";rc", func=typer("regexp_contains()", 1, paste=False)),
    # GIT ##############################################################################
    MacroEncoding(encoding=";da", func=typer("deactivate")),
    MacroEncoding(encoding=";gd", func=typer("git diff ")),
    MacroEncoding(encoding=";gl", func=typer(git_log)),
    MacroEncoding(encoding=";gs", func=typer("git status")),
    MacroEncoding(encoding=";ga", func=typer("git add -A")),
    MacroEncoding(encoding=";gc", func=typer('git commit -m ""', 1)),
    MacroEncoding(encoding=";ac", func=typer('git add -A && git commit -m ""', 1)),
    MacroEncoding(
        encoding=";ap",
        func=typer('git add -A && git commit -m "" && git push', 13),
    ),
    MacroEncoding(encoding=";co", func=typer("git checkout ")),
    MacroEncoding(encoding=";bh", func=typer("git branch ")),
    MacroEncoding(encoding=";gp", func=typer("git push ")),
    MacroEncoding(encoding=";me", func=typer("git merge ")),
    MacroEncoding(encoding=";pl", func=typer("git pull ")),
    # FORMATTERS #######################################################################
    MacroEncoding(encoding=";sl", func=typer.select_line_at_caret_and_copy),
    MacroEncoding(encoding=";re", func=formatters.cut_right_equality),
    MacroEncoding(encoding=";se", func=formatters.set_equal_to_self),
    MacroEncoding(encoding=";2s", func=formatters.to_snake),
    MacroEncoding(encoding=";hl", func=formatters.format_hash),
    MacroEncoding(encoding=";hc", func=formatters.format_hash_center),
    MacroEncoding(encoding=";dl", func=formatters.format_dash),
    MacroEncoding(encoding=";dc", func=formatters.format_dash_center),
    MacroEncoding(encoding=";rb", func=formatters.remove_blanklines),
    MacroEncoding(encoding=";2u", func=formatters.to_upper),
    MacroEncoding(encoding=";2l", func=formatters.to_lower),
    MacroEncoding(encoding=";ul", func=formatters.underline),
    MacroEncoding(encoding=";sj", func=formatters.split_join),
    MacroEncoding(encoding=";up", func=formatters.unnest_parentheses),
    # TOOLS ############################################################################
    MacroEncoding(encoding=";i2", func=img2text.img2text),
    MacroEncoding(encoding=";de", func=typer.type_date),
    MacroEncoding(encoding=";sc", func=formatters.spell_check),
    MacroEncoding(encoding=";wb", func=formatters.open_cb_url),
    MacroEncoding(encoding=";jl", func=formatters.type_journal_header),
    # OTHER ############################################################################
    MacroEncoding(encoding=";;t", func=typer("timestamp")),
]

if sys.platform == "win32":
    from tools.key_macro.macros import text2speech

    ENCODINGS.append(MacroEncoding(encoding=";ee", func=text2speech.text2speech))


class DuplicateEncodingError(ValueError):
    pass


def test_for_duplicates() -> None:
    """On program start check for duplicate encodings across the macro that would result
    in two macro being called at once."""
    codes = [macro.encoding for macro in ENCODINGS]
    if len(codes) != len(set(codes)):
        err_msg = f"you have added a duplicate encoding: \n{Counter(codes)}"
        input(err_msg + ", press any key to continue...")
        raise DuplicateEncodingError(err_msg)
