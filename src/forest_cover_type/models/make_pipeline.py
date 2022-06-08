from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_pipeline(model: str, use_scaler: bool, random_state: int) -> Pipeline:
    steps = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))

    if model == "LogisticRegression":
        classifier = LogisticRegression()
    elif model == "RandomForestClassifier":
        classifier = RandomForestClassifier(random_state=random_state)
    elif model == "ExtraTreeClassifier":
        classifier = ExtraTreesClassifier(random_state=random_state)

    steps.append(("clf", classifier))
    return Pipeline(steps=steps)
