import click

import pandas as pd


@click.command()
@click.option("-d", "--dataset-path", default="data/train.csv", show_default=True, help="Path to csv with data.")
def train(dataset_path):
    """Script that trains a model and saves it to a file."""
    data = pd.read_csv(dataset_path, index_col="Id")
    click.echo(f"data shape: {data.shape}")
