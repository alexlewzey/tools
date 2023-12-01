"""A collection of all my available macro."""


import sys
from collections import Counter
from functools import partial
from typing import Callable, Tuple

from pynput.keyboard import KeyCode

from clmac import conftk
from clmac.macros import formatters, img2text
from clmac.typer import Typer

from .config import boilerplate, custom_0


class MacroEncoding:
    """Represents a specific macro including its name, callable functionality and the
    keyboard encoding that triggers it."""

    def __init__(self, encoding: str, func: Callable):
        self.encoding = encoding
        self.func = func
        self.encode_set = self.get_encoding_set()

    def get_encoding_set(self) -> Tuple:
        """Get a pynputs representation of the keyboard encoding."""
        return tuple(KeyCode(char=char) for char in self.encoding)

    def get_set_func_pair(self) -> Tuple[Tuple, Callable]:
        """Return a tuple containing the pynputs encoding set and the callable
        functionality."""
        return self.encode_set, self.func

    def get_text_properties(self) -> Tuple[str, str, str]:
        """Return a tuple of the macro string properties."""
        return self.category, self.name, self.encoding


def load_and_type(setting: str) -> Callable:
    """Load a setting from the config file and pass into a function (that will type out
    the setting) that is returned."""

    def type_detail():
        settings = conftk.load_personal()
        try:
            typer.type_text(settings[setting].replace("\\n", "\n"))
        except TypeError:
            print(f"No {setting} found, set config with $ mcli config set -a")

    return type_detail


def load_and_type_numkey(num: int, settings_loader: Callable) -> Callable:
    def type_detail():
        settings = settings_loader()
        try:
            text = settings[num].strip().replace("\\n", "\n")
            typer.type_text(text)
        except TypeError:
            print(f"No {num} found")

    return type_detail


typer = Typer()

load_and_type_numkey_0 = partial(
    load_and_type_numkey, settings_loader=conftk.load_numkeys_0
)
load_and_type_numkey_1 = partial(
    load_and_type_numkey, settings_loader=conftk.load_numkeys_1
)

ENCODINGS = [
    MacroEncoding(encoding=";hm", func=load_and_type("hotmail")),
    MacroEncoding(encoding=";gm", func=load_and_type("gmail")),
    MacroEncoding(
        encoding=";wm",
        func=load_and_type("work_mail"),
    ),
    MacroEncoding(encoding=";al", func=load_and_type("name")),
    MacroEncoding(encoding=";mb", func=load_and_type("mobile")),
    MacroEncoding(encoding=";un", func=load_and_type("username")),
    MacroEncoding(encoding=";ad", func=load_and_type("address")),
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
        encoding=";qy",
        func=typer('query = f"""\n\n"""\ndf = spark.sql(query)', 4),
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
    MacroEncoding(encoding=";sr", func=typer("super().__init__()")),
    MacroEncoding(encoding=";st", func=typer("pdb.set_trace()")),
    MacroEncoding(
        encoding=";pu",
        func=typer("pip install -U pip"),
    ),
    MacroEncoding(
        encoding=";lh",
        func=typer("http://localhost:"),
    ),
    MacroEncoding(encoding=";;1", func=typer(custom_0.one)),
    MacroEncoding(encoding=";;2", func=typer(custom_0.two)),
    MacroEncoding(encoding=";;3", func=typer(custom_0.three)),
    MacroEncoding(encoding=";;4", func=typer(custom_0.four)),
    MacroEncoding(encoding=";;5", func=typer(custom_0.five)),
    MacroEncoding(encoding=";;6", func=typer(custom_0.six)),
    MacroEncoding(encoding=";;7", func=typer(custom_0.seven)),
    MacroEncoding(encoding=";;8", func=typer(custom_0.eight)),
    MacroEncoding(encoding=";;9", func=typer(custom_0.nine)),
    MacroEncoding(encoding=";i2", func=img2text.img2text),
    MacroEncoding(encoding=";ts", func=typer.type_timestamp),
    MacroEncoding(encoding=";de", func=typer.type_date),
    MacroEncoding(
        encoding=";up",
        func=formatters.unnest_parathesis,
    ),
    MacroEncoding(
        encoding=";2f",
        func=formatters.wrap_fstring,
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
