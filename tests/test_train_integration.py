import click.testing
import pytest
import pathlib
from forest_cover_type.models.train import train


@pytest.fixture
def runner():
    return click.testing.CliRunner()


def test_train_succeeds(runner):
    path_to_data = pathlib.Path().resolve()
    with runner.isolated_filesystem():
        result = runner.invoke(train, [
                               "--dataset-path", f"{path_to_data}/tests/sample.csv", "--save-model-path", "model.joblib", "--save-best-model-path", "best_model.joblib"])
        print(result.output)
        assert result.exit_code == 0
        assert "Accuracy" in result.output
