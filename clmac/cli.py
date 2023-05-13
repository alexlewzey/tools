#!/usr/bin/env python
"""Command line tool app / main interface."""
import shutil
import sys
from distutils.dir_util import copy_tree
from typing import Generator

import click
import yaml
from PyPDF2 import PdfFileMerger
from tabulate import tabulate

import clmac.macro.app as app
from clmac.cltools import clipurls
from clmac.cltools.tunings import show_tuning
from clmac.config import conftk, definitions
from clmac.helpers.core import Callable, Optional, Path, Union, logging
from clmac.helpers.typer import Typer
from clmac.macro.encodings import ENCODINGS, MacroEncoding

GenPath = Generator[Path, None, None]

logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


@cli.group()
def clt():
    """Command line tool-box."""


@clt.command()
@click.argument("semitones", type=int)
def tuning(semitones):
    """Print a seven string tuning to the terminal dropped by the passed number
    of semitones."""
    show_tuning(semitones)


@clt.command()
def mvg(src: str, glob: str, dst: str) -> None:
    """Move files in clmac that match glob to dst."""
    globed = Path(src).glob(glob)
    _move_files(src_paths=globed, dst=dst)


def _move_files(src_paths, dst: str) -> None:
    """Move a list of files to dst."""
    path_pairs = [(path, Path(dst) / path.name) for path in src_paths]

    for src, dst in path_pairs:
        shutil.move(src.as_posix(), dst.as_posix())


@clt.command()
@click.argument("new_file", type=click.Path(file_okay=True))
@click.argument("input_files", type=click.Path(exists=True), nargs=-1)
def pdf_merge(new_file, input_files) -> None:
    """Merge together an arbitrary number of pdfs and save as a new file."""
    merger = PdfFileMerger()
    for file in input_files:
        merger.append(file)
    merger.write(new_file)
    merger.close()
    click.echo(f"new_file: {new_file}, inputs: {input_files}")


@clt.command()
@click.option("-n", default=None, type=int)
def urlclipper(n):
    """Copy all (or given no.) of urls from open Chrome browser saving out to
    clipboard."""
    clipurls.clip_urls(n_urls=n)


def to_unix(path: Path) -> Path:
    """Convert tilde to the home directory of the current operating system."""
    if path.parts[0] == "~":
        path = Path.home() / Path(*path.parts[1:])
    return path


def get_tree(path: Union[Path, str], cond: Callable) -> list[tuple[Path, int]]:
    tree = Path(path).rglob("*")
    return sorted(
        [(p, p.stat().st_size) for p in tree if cond(p)],
        key=lambda x: x[1],
        reverse=True,
    )


@clt.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False))
@click.option(
    "-n", type=int, default=10, help="Max no. of files displayed", show_default=True
)
def file_sizes(path: Union[Path, str], n: int = 10) -> None:
    """List the largest file child files of a directory."""
    path = Path(path)
    sizes = get_tree(to_unix(path), cond=lambda p: p.is_file())
    file_out = tabulate(
        [(p.as_posix(), hr_bytes(size)) for p, size in sizes[:n]],
        headers=("file", "size"),
    )
    click.echo(file_out)


@clt.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False))
@click.option(
    "-n", type=int, default=10, help="Max no. of files displayed", show_default=True
)
def dir_sizes(path: Union[Path, str], n: int = 10) -> None:
    """List the largest child directories of a directory."""
    path = Path(path)
    sizes = get_tree(to_unix(path), cond=lambda p: p.is_dir())
    dir_out = tabulate(
        [(p.as_posix(), hr_bytes(size)) for p, size in sizes[:n]],
        headers=("directory", "size"),
    )
    click.echo(dir_out)


def hr_bytes(n_bytes: int, binary=False, decimal_places=1):
    """Return bytes in a human-readable format."""
    if binary:
        factor, units = 1024, ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    else:
        factor, units = 1000, ["B", "KB", "MB", "GB", "TB", "PB"]
    for unit in units:
        if n_bytes < factor:
            break
        n_bytes /= factor
    return f"{n_bytes:.{decimal_places}f}{unit}"


@clt.command()
@click.argument("src", type=click.Path(exists=True))
@click.argument("dest", type=click.Path())
def mv(src: Union[Path, str], dest: Union[Path, str]) -> None:
    """Python implementation of unix mv command."""
    src, dest = to_unix(Path(src)), to_unix(Path(dest))
    if src.is_dir():
        (dest / src).mkdir()
        for path in sorted(src.rglob("*"), key=lambda p: p.is_file()):
            if path.is_dir():
                print(f"making dir {dest / path}")
                (dest / path).mkdir(parents=True, exist_ok=True)
            else:
                print(f"moving file {dest / path}")
                path.rename(dest / path)
        shutil.rmtree(src)
    elif src.is_file():
        if dest.is_dir():
            src.rename(dest / src.name)
        else:
            src.rename(dest)


@clt.command()
@click.argument("src", type=click.Path(exists=True))
@click.argument("dest", type=click.Path())
def cp(src: Union[Path, str], dest: Union[Path, str]) -> None:
    """Python implementation of unix cp command."""
    src, dest = to_unix(Path(src)), to_unix(Path(dest))
    if src.is_dir():
        copy_tree(src.as_posix(), (dest / src).as_posix())
    elif src.is_file():
        if dest.is_dir():
            shutil.copy2(src, dest / src.name)
        else:
            shutil.copy2(src, dest)


@clt.command()
@click.argument("src", type=click.Path(exists=True))
def rm(src) -> None:
    """Python implementation of unix rm command."""
    src = to_unix(Path(src))
    if src.is_dir():
        shutil.rmtree(src)
    elif src.is_file():
        src.unlink()


@clt.command()
@click.argument("path", nargs=-1, type=click.Path(file_okay=False))
def cd(path: Optional[Union[Path, str]]) -> None:
    typer = Typer()
    if path:
        typer.type_text(f"cd {to_unix(Path((path[0]))).as_posix()}")
    else:
        typer.type_text(f"cd {Path.home().as_posix()}")
    typer.enter()


@cli.group()
def kel():
    """Keyboard event listener."""


@kel.command("run")
def macros_run():
    """Run the macro script."""
    app.run()


@kel.command("ls")
@click.option("--search", "-s", help="Filter table by name")
def macros_ls(search):
    """List all available macro in the terminal."""
    text = _make_macro_table_text(ENCODINGS, search=search)
    click.echo(text)


def _make_macro_table_text(
    encodings: list[MacroEncoding], search: Optional[str] = None
) -> str:
    """Print in tabular format the type name and encoding of every macro in the
    program."""
    macros_tabular = [e.get_text_properties() for e in encodings]
    if search:
        macros_tabular = [macro for macro in macros_tabular if search in macro[1]]
    macros_tabular = sorted(macros_tabular, key=lambda x: (x[0], x[1]))
    return tabulate(macros_tabular, headers=("type", "name", "encoding"))


@kel.command("lp")
def ls_config():
    """List the current personal configuration settings."""
    settings = conftk.load_personal()
    click.echo(tabulate(settings.items(), headers=("setting", "value")))


@kel.command("sp")
def set_config() -> None:
    """Gives option to change any of the personal string settings where the
    existing setting is the default."""
    settings = conftk.load_personal()
    new_settings = {}
    for k, v in settings.items():
        new_settings[k] = click.prompt(k, default=v or "...", type=str)

    with definitions.PERSONAL_YAML.open("w") as f:
        yaml.dump(new_settings, f)

    click.echo("saved new settings...")


def _set_nums(keys, settings, path_yaml) -> None:
    """Set the text associated with the num key macros."""
    if not keys:
        click.echo("Requires num key args")
        sys.exit()
    for n in keys:
        settings[int(n)] = click.prompt(f"macro {n}", type=str)

    with path_yaml.open("w") as f:
        yaml.dump(settings, f)
    click.echo("saved new settings...")


@click.argument("keys", nargs=-1)
@kel.command("sn0")
def set_nums_0(keys) -> None:
    """Set the text associated with the num key macros (set 0)"""
    _set_nums(
        keys, settings=conftk.load_numkeys_0(), path_yaml=definitions.NUMKEYS_YAML_0
    )


@click.argument("keys", nargs=-1)
@kel.command("sn1")
def set_nums_1(keys) -> None:
    """Set the text associated with the num key macros (set 1)"""
    _set_nums(
        keys, settings=conftk.load_numkeys_1(), path_yaml=definitions.NUMKEYS_YAML_1
    )


def _ls_nums(settings):
    """List current numkey macro assignments."""
    click.echo(tabulate(settings.items(), headers=("setting", "value")))


@kel.command("ln0")
def ls_nums_0():
    """List current numkey macro assignments (set 0)"""
    _ls_nums(conftk.load_numkeys_0())


@kel.command("ln1")
def ls_nums_1():
    """List current numkey macro assignments (set 1)"""
    _ls_nums(conftk.load_numkeys_1())


if __name__ == "__main__":
    cli()
