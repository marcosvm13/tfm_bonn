"""Console script for tfm_bonn."""

import typer
from rich.console import Console

from tfm_bonn import utils

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for tfm_bonn."""
    console.print("Replace this message by putting your code into "
               "tfm_bonn.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    utils.do_something_useful()


if __name__ == "__main__":
    app()
