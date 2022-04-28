from email.policy import default
from joblib import dump

import click

import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import cross_validate

from forest_cover_type.data.load_dataset import load_dataset
from forest_cover_type.models.make_pipeline import make_pipeline


@click.command()
@click.option("-d", "--dataset-path", default="data/train.csv", show_default=True, help="Path to csv with data.")
@click.option("-s", "--save-model-path", default="models/model.joblib", show_default=True, help="Path to save trained model.")
@click.option("--test-split-ratio", default=0.2, show_default=True, help="Proportion of the dataset to include in the test split, should be between 0.0 and 1.0.")
@click.option("--random-state", default=42, show_default=True, help="Random state.")
@click.option("--use-scaler", default=True, show_default=True, help="Specifies whether to scale the data.")
@click.option("--logreg-c", default=1.0, show_default=True, help="Inverse of regularization strength.")
@click.option("--max-iter", default=100, show_default=True, help="Maximum number of iterations taken for the solvers to converge.")
@click.option("--k-folds", default=5, show_default=True, help="Number of folds in cross-validation.")
def train(dataset_path, save_model_path, test_split_ratio, random_state, use_scaler, logreg_c, max_iter, k_folds):
    """Script that trains a model and saves it to a file."""
    X_train, X_val, y_train, y_val = load_dataset(
        dataset_path=dataset_path, test_split_ratio=test_split_ratio, random_state=random_state)

    pipeline = make_pipeline(use_scaler=use_scaler,
                             logreg_c=logreg_c, max_iter=max_iter)
    scores = cross_validate(pipeline, pd.concat([X_train, X_val]), pd.concat(
        [y_train, y_val]), cv=k_folds, scoring=('accuracy', 'neg_log_loss', 'roc_auc_ovr'))
    click.echo(
        f"Mean accuracy across all CV splits: {scores['test_accuracy'].mean()}")
    click.echo(
        f"Mean neg_log_loss across all CV splits: {scores['test_neg_log_loss'].mean()}")
    click.echo(
        f"Mean roc_auc_ovr across all CV splits: {scores['test_roc_auc_ovr'].mean()}")

    pipeline.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    dump(pipeline, save_model_path)
    click.echo(f"Model is saved to {save_model_path}.")
