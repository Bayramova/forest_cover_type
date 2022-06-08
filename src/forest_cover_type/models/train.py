from pathlib import Path
import warnings

import click
from joblib import dump
import mlflow

from forest_cover_type.data.load_dataset import load_dataset
from forest_cover_type.features.build_features import build_features
from forest_cover_type.models.make_pipeline import make_pipeline
from forest_cover_type.models.select_and_evaluate import (
    get_tuned_model,
    KFoldCV,
    nestedCV,
)

warnings.filterwarnings("ignore", category=UserWarning)


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
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True),
    show_default=True,
    help="Path to save trained model.",
)
@click.option(
    "--save-best-model-path",
    default="data/best_model.joblib",
    type=click.Path(dir_okay=False, writable=True),
    show_default=True,
    help="Path to save model with best accuracy across all runs.",
)
@click.option(
    "--random-state", default=42, type=int, show_default=True, help="Random state."
)
@click.option(
    "--use-scaler",
    default=False,
    type=bool,
    show_default=True,
    help="Specifies whether to scale the data.",
)
@click.option(
    "--model",
    default="ExtraTreeClassifier",
    type=click.Choice(
        ["LogisticRegression", "RandomForestClassifier", "ExtraTreeClassifier"]
    ),
    show_default=True,
    help="Name of model for training.",
)
@click.option(
    "--nested-cv",
    default=True,
    type=bool,
    show_default=True,
    help="Specifies whether to use nested CV with GridSearch or ordinary CV.",
)
@click.option(
    "--outer-cv-folds",
    default=5,
    type=int,
    show_default=True,
    help="Number of folds in outer cross-validation.",
)
@click.option(
    "--inner-cv-folds",
    default=3,
    type=int,
    show_default=True,
    help="Number of folds in inner cross-validation.",
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    save_best_model_path: Path,
    random_state: int,
    use_scaler: bool,
    model: str,
    nested_cv: bool,
    outer_cv_folds: int,
    inner_cv_folds: int,
) -> None:
    """Script that trains a model and saves it to a file."""

    with mlflow.start_run(run_name=model):
        X, y = load_dataset(dataset_path=dataset_path)
        X = build_features(X)

        pipeline = make_pipeline(
            model=model, use_scaler=use_scaler, random_state=random_state
        )

        if nested_cv:
            scores = nestedCV(
                pipeline, model, X, y, random_state, outer_cv_folds, inner_cv_folds
            )
            trained_model, params = get_tuned_model(
                pipeline, model, X, y, random_state, outer_cv_folds
            )
        else:
            scores = KFoldCV(pipeline, X, y, random_state, outer_cv_folds)
            trained_model = pipeline.fit(X, y)

        # report performance
        accuracy = scores["test_accuracy"].mean()
        log_loss = -scores["test_neg_log_loss"].mean()
        roc_auc = scores["test_roc_auc_ovr"].mean()
        click.echo(
            f"Accuracy: {accuracy} +/- {round(scores['test_accuracy'].std() * 100, 3)}"
        )
        click.echo(
            f"Log_loss: {log_loss} +/- {round(scores['test_neg_log_loss'].std() * 100, 3)}"
        )
        click.echo(
            f"Roc_auc: {roc_auc} +/- {round(scores['test_roc_auc_ovr'].std() * 100, 3)}"
        )

        # log parameters, metrics and model to mlflow
        if nested_cv:
            mlflow.log_params(params)
        mlflow.log_param("use_scaler", use_scaler)

        mlflow.log_metrics(
            {"accuracy": accuracy, "log_loss": log_loss, "roc_auc": roc_auc}
        )
        mlflow.sklearn.log_model(trained_model, "model")

        # save model
        dump(trained_model, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")

        # search for best run
        best_run = mlflow.search_runs(
            order_by=["metrics.accuracy DESC"], max_results=1
        ).iloc[0]
        click.echo(f"\nBest run accuracy: {best_run['metrics.accuracy']}")

        # load and save best model
        best_estimator = mlflow.sklearn.load_model(f"{best_run['artifact_uri']}/model")
        dump(best_estimator, save_best_model_path)
        click.echo(f"Best model is saved to {save_best_model_path}.")
