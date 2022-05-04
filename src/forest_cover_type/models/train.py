from joblib import dump

import click
import mlflow

import numpy as np
from sklearn.model_selection import cross_validate, KFold, GridSearchCV

from forest_cover_type.data.load_dataset import load_dataset
from forest_cover_type.models.make_pipeline import make_pipeline
from forest_cover_type.features.build_features import build_features


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
    default="models/model.joblib",
    type=click.Path(dir_okay=False, writable=True),
    show_default=True,
    help="Path to save trained model.",
)
@click.option(
    "--save-best-model-path",
    default="models/best_model.joblib",
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
    "--bin-elevation",
    default=False,
    type=bool,
    show_default=True,
    help="Specifies whether to bin 'Elevation' feature.",
)
@click.option(
    "--log-transform",
    default=False,
    type=bool,
    show_default=True,
    help="Specifies whether to log-transform some of highly skewed features.",
)
@click.option(
    "--model",
    default="RandomForestClassifier",
    type=click.Choice(["LogisticRegression", "RandomForestClassifier"]),
    show_default=True,
    help="Name of model for training.",
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
    dataset_path,
    save_model_path,
    save_best_model_path,
    random_state,
    use_scaler,
    bin_elevation,
    log_transform,
    model,
    outer_cv_folds,
    inner_cv_folds,
):
    """Script that trains a model and saves it to a file."""
    with mlflow.start_run(run_name=model):
        X, y = load_dataset(dataset_path=dataset_path)
        X = build_features(X, bin_elevation=bin_elevation, log_transform=log_transform)

        pipeline = make_pipeline(
            model=model, use_scaler=use_scaler, random_state=random_state
        )

        # set up parameters grid
        if model == "LogisticRegression":
            param_grid = {
                "clf__C": np.power(10.0, range(-2, 3)),
                "clf__max_iter": list(range(500, 1000, 100)),
            }
        elif model == "RandomForestClassifier":
            param_grid = {
                "clf__n_estimators": list(range(100, 600, 100)),
                "clf__max_depth": list(range(2, 11, 2)) + [None],
                "clf__min_samples_split": list(range(2, 11, 2)),
            }

        # execute nested cv
        cv_inner = KFold(
            n_splits=inner_cv_folds, shuffle=True, random_state=random_state
        )
        gridcv = GridSearchCV(
            pipeline, param_grid, scoring="accuracy", n_jobs=-1, cv=cv_inner, refit=True
        )
        cv_outer = KFold(
            n_splits=outer_cv_folds, shuffle=True, random_state=random_state
        )
        scores = cross_validate(
            gridcv,
            X,
            y,
            cv=cv_outer,
            scoring=("accuracy", "neg_log_loss", "roc_auc_ovr"),
            n_jobs=-1,
        )

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

        # configure final model
        gridcv.fit(X, y)
        click.echo(f"Best Parameters: {gridcv.best_params_}")
        dump(gridcv.best_estimator_, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")

        # log parameters, metrics and model to mlflow
        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("bin_elevation", bin_elevation)
        mlflow.log_param("log_transform", log_transform)
        if model == "LogisticRegression":
            mlflow.log_param("logreg_c", gridcv.best_params_["clf__C"])
            mlflow.log_param("max_iter", gridcv.best_params_["clf__max_iter"])
        elif model == "RandomForestClassifier":
            mlflow.log_param("n_estimators", gridcv.best_params_["clf__n_estimators"])
            mlflow.log_param("max_depth", gridcv.best_params_["clf__max_depth"])
            mlflow.log_param(
                "min_samples_split", gridcv.best_params_["clf__min_samples_split"]
            )

        mlflow.log_metrics(
            {"accuracy": accuracy, "log_loss": log_loss, "roc_auc": roc_auc}
        )
        mlflow.sklearn.log_model(pipeline, "model")

        # search for best run
        best_run = mlflow.search_runs(
            order_by=["metrics.accuracy DESC"], max_results=1
        ).iloc[0]
        click.echo(f"\nBest run accuracy: {best_run['metrics.accuracy']}")

        # load and save best model
        best_estimator = mlflow.sklearn.load_model(f"{best_run['artifact_uri']}/model")
        dump(best_estimator, save_best_model_path)
        click.echo(f"Best model is saved to {save_best_model_path}.")
