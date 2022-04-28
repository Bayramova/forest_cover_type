from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def make_pipeline(use_scaler, logreg_c, max_iter):
    steps = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    classifier = LogisticRegression(C=logreg_c, max_iter=max_iter)
    steps.append(("clf", classifier))
    return Pipeline(steps=steps)
