from pathlib import Path
import warnings

import click
import pandas as pd
from pandas_profiling import ProfileReport

warnings.filterwarnings("ignore")


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False),
    show_default=True,
    help="Path to csv with data.",
)
@click.option(
    "-s",
    "--save-report-path",
    default="reports/eda_report.html",
    type=click.Path(dir_okay=False, writable=True),
    show_default=True,
    help="Path to save generated EDA report.",
)
def generate_eda(dataset_path: Path, save_report_path: Path) -> None:
    """
    Script that generates an EDA report
    and saves it as .html file in reports directory.
    """
    data = pd.read_csv(dataset_path)
    profile = ProfileReport(data)
    profile.to_file(save_report_path)
    click.echo(f"Report is saved to {save_report_path}.")
