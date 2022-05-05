from pathlib import Path
from typing import Tuple

import click
import pandas as pd


def load_dataset(dataset_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(dataset_path, index_col="Id")
    X = data.drop(columns=["Cover_Type"])
    y = data["Cover_Type"]
    click.echo(f"Dataset shape: {X.shape}")
    return X, y
