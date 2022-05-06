from click.testing import CliRunner
import pytest

from forest_cover_type.models.train import train


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_train_fails_on_invalid_dataset_path(runner: CliRunner) -> None:
    result = runner.invoke(train, ["--dataset-path", "data/qwerty.csv"])
    assert result.exit_code == 2
    assert "Invalid value for '-d' / '--dataset-path'" in result.output


# def test_train_fails_on_invalid_save_model_path(runner: CliRunner) -> None:
#     result = runner.invoke(
#         train,
#         ["--dataset-path", "tests/fixtures/train.csv", "--save-model-path", "data/"],
#     )
#     assert result.exit_code == 2
#     assert "Invalid value for '-s' / '--save-model-path'" in result.output


def test_train_fails_on_invalid_model_input(runner: CliRunner) -> None:
    result = runner.invoke(train, ["--model", "KNN"])
    assert result.exit_code == 2
    assert "Invalid value for '--model'" in result.output
