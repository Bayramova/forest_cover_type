from joblib import dump

import click

from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from forest_cover_type.data.load_dataset import load_dataset
from forest_cover_type.models.make_pipeline import make_pipeline


@click.command()
@click.option("-d", "--dataset-path", default="data/train.csv", show_default=True, help="Path to csv with data.")
@click.option("-s", "--save-model-path", default="models/model.joblib", show_default=True, help="Path to save trained model.")
@click.option("--test-split-ratio", default=0.2, show_default=True, help="Proportion of the dataset to include in the test split, should be between 0.0 and 1.0.")
@click.option("--random-state", default=42, show_default=True, help="Random state.")
@click.option("--logreg-c", default=1.0, show_default=True, help="Inverse of regularization strength.")
@click.option("--max-iter", default=100, show_default=True, help="Maximum number of iterations taken for the solvers to converge.")
def train(dataset_path, save_model_path, test_split_ratio, random_state, logreg_c, max_iter):
    """Script that trains a model and saves it to a file."""
    X_train, X_val, y_train, y_val = load_dataset(
        dataset_path=dataset_path, test_split_ratio=test_split_ratio, random_state=random_state)

    pipeline = make_pipeline(logreg_c=logreg_c, max_iter=max_iter)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    probs_pred = pipeline.predict_proba(X_val)
    accuracy = accuracy_score(y_true=y_val, y_pred=y_pred)
    click.echo(f"accuracy = {accuracy}")
    logloss = log_loss(y_true=y_val, y_pred=probs_pred)
    click.echo(f"logloss: {logloss}")
    roc_auc = roc_auc_score(
        y_true=y_val, y_score=probs_pred, multi_class='ovr')
    click.echo(f"roc_auc = {roc_auc}")

    dump(pipeline, save_model_path)
    click.echo(f"Model is saved to {save_model_path}.")
