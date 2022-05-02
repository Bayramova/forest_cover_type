import click.testing
import pytest

from forest_cover_type.models.train import train


@pytest.fixture
def runner():
    return click.testing.CliRunner()


def test_train_fails_on_invalid_dataset_path(runner):
    result = runner.invoke(train, ["--dataset-path", "data/qwerty.csv"])
    assert result.exit_code == 2
    assert "Invalid value for '-d' / '--dataset-path'" in result.output


def test_train_fails_on_invalid_save_model_path(runner):
    result = runner.invoke(train, ["--save-model-path", "models/"])
    assert result.exit_code == 2
    assert "Invalid value for '-s' / '--save-model-path'" in result.output


def test_train_fails_on_invalid_model_input(runner):
    result = runner.invoke(train, ["--model", "KNN"])
    assert result.exit_code == 2
    assert "Invalid value for '--model'" in result.output
