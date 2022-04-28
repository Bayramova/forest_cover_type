from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def make_pipeline(logreg_c, max_iter):
    scaler = StandardScaler()
    classifier = LogisticRegression(C=logreg_c, max_iter=max_iter)
    pipeline = Pipeline(steps=[("scaler", scaler), ("clf", classifier)])
    return pipeline
