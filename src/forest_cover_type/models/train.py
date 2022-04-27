import click

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


@click.command()
@click.option("-d", "--dataset-path", default="data/train.csv", show_default=True, help="Path to csv with data.")
def train(dataset_path):
    """Script that trains a model and saves it to a file."""
    data = pd.read_csv(dataset_path, index_col="Id")
    X = data.drop(columns=['Cover_Type'])
    y = data['Cover_Type']
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    scaler.fit(X_train)
    classifier = LogisticRegression(max_iter=500)
    classifier.fit(scaler.transform(X_train), y_train)
    accuracy = accuracy_score(
        y_true=y_val, y_pred=classifier.predict(scaler.transform(X_val)))
    click.echo(f"accuracy = {accuracy}")
