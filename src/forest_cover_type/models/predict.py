from pathlib import Path

import click
import joblib
import pandas as pd

from forest_cover_type.features.build_features import build_features


@click.command()
@click.option(
    "-t",
    "--test-dataset-path",
    default="data/test.csv",
    type=click.Path(exists=True, dir_okay=False),
    show_default=True,
    help="Path to csv with test data.",
)
@click.option(
    "-s",
    "--save-preds-path",
    default="data/submission.csv",
    type=click.Path(dir_okay=False, writable=True),
    show_default=True,
    help="Path to save submission file.",
)
@click.option(
    "-m",
    "--model-path",
    default="data/best_model.joblib",
    type=click.Path(dir_okay=False, writable=True),
    show_default=True,
    help="Path to the model.",
)
def predict(test_dataset_path: Path, save_preds_path: Path, model_path: Path) -> None:
    X = pd.read_csv(test_dataset_path, index_col="Id")
    X = build_features(X)

    model = joblib.load(model_path)

    preds = model.predict(X)

    pd.DataFrame({"Id": X.index, "Cover_Type": preds}).to_csv(
        save_preds_path, index=False
    )

    click.echo(f"Submission file is saved to {save_preds_path}.")
