import click

import pandas as pd
from pandas_profiling import ProfileReport

import warnings
warnings.filterwarnings('ignore')


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    show_default=True,
    help="Path to csv with data."
)
def generate_eda(dataset_path):
    """Script that generates an EDA report and saves it as .html file in reports directory."""
    data = pd.read_csv(dataset_path)
    profile = ProfileReport(data)
    profile.to_file("reports/eda_report.html")
