from joblib import dump

import click
import mlflow

import pandas as pd
from sklearn.model_selection import cross_validate

from forest_cover_type.data.load_dataset import load_dataset
from forest_cover_type.models.make_pipeline import make_pipeline
from forest_cover_type.features.build_features import build_features


@click.command()
@click.option("-d", "--dataset-path", default="data/train.csv", show_default=True, help="Path to csv with data.")
@click.option("-s", "--save-model-path", default="models/model.joblib", show_default=True, help="Path to save trained model.")
@click.option("--save-best-model-path", default="models/best_model.joblib", show_default=True, help="Path to save model with best accuracy across all runs.")
@click.option("--random-state", default=42, show_default=True, help="Random state.")
@click.option("--use-scaler", default=True, show_default=True, help="Specifies whether to scale the data.")
@click.option("--bin-elevation", default=False, show_default=True, help="Specifies whether to bin 'Elevation' feature.")
@click.option("--log-transform", default=False, show_default=True, help="Specifies whether to log-transform some of highly skewed features.")
@click.option("--logreg-c", default=1.0, show_default=True, help="Inverse of regularization strength.")
@click.option("--max-iter", default=100, show_default=True, help="Maximum number of iterations taken for the solvers to converge.")
@click.option("--penalty", default="l2", show_default=True, type=click.Choice(["l2", "none"]), help="Specify the norm of the penalty.")
@click.option("--k-folds", default=5, show_default=True, help="Number of folds in cross-validation.")
@click.option("--model", default="LogisticRegression", type=click.Choice(["LogisticRegression", "RandomForestClassifier"]), show_default=True, help="Name of model for training.")
@click.option("--n-estimators", default=100, show_default=True, help="The number of trees in the forest.")
@click.option("--max-depth", default=-1, show_default=True, help="The maximum depth of the tree.")
@click.option("--min-samples-split", default=2, show_default=True, help="The minimum number of samples required to split an internal node.")
def train(dataset_path, save_model_path, save_best_model_path, random_state, use_scaler, bin_elevation, log_transform, logreg_c, max_iter, penalty, k_folds, model, n_estimators, max_depth, min_samples_split):
    """Script that trains a model and saves it to a file."""
    with mlflow.start_run(run_name=model):
        X, y = load_dataset(dataset_path=dataset_path)
        X = build_features(X, bin_elevation=bin_elevation,
                           log_transform=log_transform)

        pipeline = make_pipeline(model=model, use_scaler=use_scaler,
                                 logreg_c=logreg_c, max_iter=max_iter, penalty=penalty, random_state=random_state, n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)

        scores = cross_validate(pipeline, X, y, cv=k_folds, scoring=(
            'accuracy', 'neg_log_loss', 'roc_auc_ovr'))
        accuracy = scores['test_accuracy'].mean()
        log_loss = - scores['test_neg_log_loss'].mean()
        roc_auc = scores['test_roc_auc_ovr'].mean()
        click.echo(
            f"Mean accuracy across all CV splits: {accuracy}")
        click.echo(
            f"Mean log_loss across all CV splits: {log_loss}")
        click.echo(
            f"Mean roc_auc_ovr across all CV splits: {roc_auc}")

        pipeline.fit(X, y)
        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")

        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("bin_elevation", bin_elevation)
        if model == "LogisticRegression":
            mlflow.log_param("logreg_c", logreg_c)
            mlflow.log_param("max_iter", max_iter)
        elif model == "RandomForestClassifier":
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param(
                "max_depth", 'None' if max_depth == -1 else max_depth)

        mlflow.log_metrics(
            {"accuracy": accuracy, "log_loss": log_loss, "roc_auc": roc_auc})
        mlflow.sklearn.log_model(pipeline, "model")

        # search for best run
        best_run = mlflow.search_runs(
            order_by=['metrics.accuracy DESC'], max_results=1).iloc[0]
        click.echo(f"\nBest run accuracy: {best_run['metrics.accuracy']}")

        # load and save best model
        best_estimator = mlflow.sklearn.load_model(
            f"{best_run['artifact_uri']}\model")
        dump(best_estimator, save_best_model_path)
        click.echo(f"Best model is saved to {save_best_model_path}.")
