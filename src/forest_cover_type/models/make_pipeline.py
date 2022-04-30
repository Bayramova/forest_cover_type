from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def make_pipeline(model, use_scaler, logreg_c, max_iter, random_state, n_estimators):
    steps = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))

    if model == "LogisticRegression":
        classifier = LogisticRegression(C=logreg_c, max_iter=max_iter)
    elif model == "RandomForestClassifier":
        classifier = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state)
    else:
        raise Exception(
            f"No model named {model}. Select one of the following options: LogisticRegression, RandomForestClassifier")
    steps.append(("clf", classifier))
    return Pipeline(steps=steps)
