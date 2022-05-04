from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_pipeline(model, use_scaler, random_state):
    steps = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))

    if model == "LogisticRegression":
        classifier = LogisticRegression()
    elif model == "RandomForestClassifier":
        classifier = RandomForestClassifier(random_state=random_state)

    steps.append(("clf", classifier))
    return Pipeline(steps=steps)
