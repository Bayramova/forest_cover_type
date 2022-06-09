import pathlib

import click
from click.testing import CliRunner
import joblib
import numpy as np
import pandas as pd
import pytest

from forest_cover_type.features.build_features import build_features
from forest_cover_type.models.train import train


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_train_succeeds(runner: CliRunner) -> None:
    temp_dir_path = pathlib.Path().resolve()
    with runner.isolated_filesystem():
        result = runner.invoke(
            train,
            [
                "--dataset-path",
                f"{temp_dir_path}/tests/fixtures/train.csv",
                "--save-model-path",
                "model.joblib",
                "--save-best-model-path",
                "best_model.joblib",
            ],
        )
        click.echo(result.output)
        assert result.exit_code == 0
        assert "Accuracy" in result.output

        # check if the model returns values within the expected categories [1,2,3,4,5,6,7]
        saved_model = joblib.load("model.joblib")
        test = pd.read_csv(f"{temp_dir_path}/tests/fixtures/test.csv")
        test = build_features(test)
        preds = saved_model.predict(test)
        assert np.isin(
            np.unique(preds), np.array([1, 2, 3, 4, 5, 6, 7]), assume_unique=True
        ).all()
