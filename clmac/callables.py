"""A collection of all my available macro."""
import json
import sys
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Callable

from pynput.keyboard import KeyCode

from clmac.config import boilerplate
from clmac.core import CUSTOM_JSON, PERSONAL_JSON
from clmac.keyboard import Typer
from clmac.macros import formatters, img2text


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
    MacroEncoding(encoding=";hm", func=partial(type_personal, key="hotmail")),
    MacroEncoding(encoding=";gm", func=partial(type_personal, key="gmail")),
    MacroEncoding(encoding=";wm", func=partial(type_personal, key="work_mail")),
    MacroEncoding(encoding=";al", func=partial(type_personal, key="name")),
    MacroEncoding(encoding=";mb", func=partial(type_personal, key="mobile")),
    MacroEncoding(encoding=";un", func=partial(type_personal, key="username")),
    MacroEncoding(encoding=";ad", func=partial(type_personal, key="address")),
    MacroEncoding(
        encoding=";tf",
        func=typer("Thanks for your email. "),
    ),
    MacroEncoding(
        encoding=";ah",
        func=typer.partial_paste(boilerplate.any_help),
    ),
    MacroEncoding(
        encoding=";pl",
        func=typer.partial_paste(boilerplate.please_queries),
    ),
    MacroEncoding(
        encoding=";pf",
        func=typer("Please find attached the "),
    ),
    MacroEncoding(encoding=";ba", func=typer("\nBest\nAlex")),
    MacroEncoding(
        encoding=";mt",
        func=typer("\n\nMany thanks\n\nAlex"),
    ),
    MacroEncoding(
        encoding=";cc",
        func=typer("count('*') "),
    ),
    MacroEncoding(
        encoding=";ua",
        func=typer("user_agent"),
    ),
    MacroEncoding(
        encoding=";ll",
        func=typer(".limit(5).toPandas()", line_end=True),
    ),
    MacroEncoding(
        encoding=";tt",
        func=typer("timestamp"),
    ),
    MacroEncoding(
        encoding=";dt",
        func=typer("dt.datetime.now().replace(microsecond=0)"),
    ),
    MacroEncoding(encoding=";tp", func=typer(".toPandas()", line_end=True)),
    MacroEncoding(encoding=";nr", func=typer("n_request")),
    MacroEncoding(
        encoding=";dp",
        func=typer("display_plotly(fig)"),
    ),
    MacroEncoding(encoding=";ri", func=typer(".reset_index()")),
    MacroEncoding(encoding=";si", func=typer(".set_index()", 1)),
    MacroEncoding(encoding=";ii", func=typer("def __init__(self, ):", 2)),
    MacroEncoding(encoding=";;h", func=typer(".head(9)")),
    MacroEncoding(encoding=";;t", func=typer("torch.")),
    MacroEncoding(encoding=";;d", func=typer(".dtypes")),
    MacroEncoding(encoding=";;s", func=typer(".shape")),
    MacroEncoding(encoding=";as", func=typer("ascending=False")),
    MacroEncoding(
        encoding=";dc",
        func=typer("docker container "),
    ),
    MacroEncoding(encoding=";di", func=typer("docker image ")),
    MacroEncoding(
        encoding=";td",
        func=typer(" -> pd.DataFrame:"),
    ),
    MacroEncoding(
        encoding=";nm",
        func=typer("if __name__ == '__main__':\n    "),
    ),
    MacroEncoding(encoding=";;;", func=typer("print()", 1)),
    MacroEncoding(encoding=";;l", func=typer("#%%")),
    MacroEncoding(encoding=";;c", func=typer(".columns")),
    MacroEncoding(
        encoding=";3d",
        func=typer.partial_paste(boilerplate.px_3d_scatter),
    ),
    MacroEncoding(
        encoding=";mp",
        func=typer.partial_paste(boilerplate.melt_plot, 58),
    ),
    MacroEncoding(
        encoding=";ds",
        func=typer.partial_paste(boilerplate.data_sci),
    ),
    MacroEncoding(
        encoding=";sv",
        func=typer(".sort_values()", 1),
    ),
    MacroEncoding(
        encoding=";sr",
        func=typer("spark.read.format('delta').table('')", 2),
    ),
    MacroEncoding(
        encoding=";sw",
        func=typer(
            (
                ".write.format('delta').mode('append')"
                ".partitionBy('domain', 'date').toTable('')"
            ),
            2,
        ),
    ),
    MacroEncoding(
        encoding=";vc",
        func=typer(".value_counts(dropna=False)"),
    ),
    MacroEncoding(
        encoding=";mu",
        func=typer(".info(memory_usage='deep')"),
    ),
    MacroEncoding(encoding=";tn", func=typer(" -> None:")),
    MacroEncoding(encoding=";sf", func=typer("select * from ")),
    MacroEncoding(
        encoding=";fs",
        func=typer("select * \nfrom "),
    ),
    MacroEncoding(
        encoding=";sd",
        func=typer.partial_paste(boilerplate.sql_duplicates, 51),
    ),
    MacroEncoding(
        encoding=";ss",
        func=typer.partial_paste(boilerplate.sql_sum, 165),
    ),
    MacroEncoding(
        encoding=";cv",
        func=typer.partial_paste("create or replace temporary view  as", 3),
    ),
    MacroEncoding(
        encoding=";ct",
        func=typer.partial_paste("create or replace table  as", 3),
    ),
    MacroEncoding(
        encoding=";sk",
        func=typer.partial_paste('spark.sql("""""")', 4),
    ),
    MacroEncoding(
        encoding=";sx",
        func=typer(" suffixes=('', '_DROP')"),
    ),
    MacroEncoding(encoding=";da", func=typer("deactivate")),
    MacroEncoding(
        encoding=";gl",
        func=typer("git log --graph --decorate --oneline"),
    ),
    MacroEncoding(encoding=";gs", func=typer("git status")),
    MacroEncoding(encoding=";ga", func=typer("git add -A")),
    MacroEncoding(
        encoding=";gc",
        func=typer('git commit -m ""', 1),
    ),
    MacroEncoding(
        encoding=";ac",
        func=typer('git add -A && git commit -m ""', 1),
    ),
    MacroEncoding(encoding=";co", func=typer("git checkout ")),
    MacroEncoding(encoding=";gb", func=typer("git branch ")),
    MacroEncoding(encoding=";st", func=typer("pdb.set_trace()")),
    MacroEncoding(
        encoding=";pu",
        func=typer("pip install -U pip"),
    ),
    MacroEncoding(
        encoding=";lh",
        func=typer("http://localhost:"),
    ),
    MacroEncoding(encoding=";;1", func=partial(type_custom, "one")),
    MacroEncoding(encoding=";;2", func=partial(type_custom, "two")),
    MacroEncoding(encoding=";;3", func=partial(type_custom, "three")),
    MacroEncoding(encoding=";;4", func=partial(type_custom, "four")),
    MacroEncoding(encoding=";;5", func=partial(type_custom, "five")),
    MacroEncoding(encoding=";;6", func=partial(type_custom, "six")),
    MacroEncoding(encoding=";;7", func=partial(type_custom, "seven")),
    MacroEncoding(encoding=";;8", func=partial(type_custom, "eight")),
    MacroEncoding(encoding=";;9", func=partial(type_custom, "nine")),
    MacroEncoding(encoding=";i2", func=img2text.img2text),
    MacroEncoding(encoding=";ts", func=typer.type_timestamp),
    MacroEncoding(encoding=";de", func=typer.type_date),
    MacroEncoding(
        encoding=";up",
        func=formatters.unnest_parathesis,
    ),
    MacroEncoding(
        encoding=";sl",
        func=typer.select_line_at_caret_and_copy,
    ),
    MacroEncoding(
        encoding=";re",
        func=formatters.cut_right_equality,
    ),
    MacroEncoding(
        encoding=";se",
        func=formatters.set_equal_to_self,
    ),
    MacroEncoding(encoding=";rr", func=formatters.format_repr),
    MacroEncoding(encoding=";2s", func=formatters.to_snake),
    MacroEncoding(encoding=";hh", func=formatters.format_hash),
    MacroEncoding(encoding=";dh", func=formatters.format_dash),
    MacroEncoding(
        encoding=";2r",
        func=formatters.imports_to_requirements,
    ),
    MacroEncoding(
        encoding=";hc",
        func=formatters.format_hash_center,
    ),
    MacroEncoding(
        encoding=";rb",
        func=formatters.remove_blanklines,
    ),
    MacroEncoding(encoding=";wt", func=formatters.wrap_text),
    MacroEncoding(encoding=";lt", func=formatters.to_list),
    MacroEncoding(
        encoding=";ul",
        func=formatters.underline,
    ),
    MacroEncoding(encoding=";2u", func=formatters.to_upper),
    MacroEncoding(encoding=";2l", func=formatters.to_lower),
    MacroEncoding(
        encoding=";sc",
        func=formatters.spell_check,
    ),
    MacroEncoding(
        encoding=";fv",
        func=formatters.format_variables,
    ),
    MacroEncoding(
        encoding=";sj",
        func=formatters.split_join,
    ),
    MacroEncoding(
        encoding=";sq",
        func=formatters.format_sql,
    ),
    MacroEncoding(
        encoding=";wb",
        func=formatters.open_cb_url,
    ),
    MacroEncoding(
        encoding=";bk",
        func=formatters.format_black,
    ),
]


if sys.platform != "darwin":
    from clmac.macros import text2speech

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


test_for_duplicates()
