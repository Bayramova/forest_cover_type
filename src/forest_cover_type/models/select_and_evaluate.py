from typing import Any, Dict, List, Tuple

import click
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, GridSearchCV, KFold
from sklearn.pipeline import Pipeline


def nestedCV(
    pipeline: Pipeline,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
    outer_cv_folds: int,
    inner_cv_folds: int,
) -> Dict[str, List[float]]:
    param_grid = set_param_grid(model_name)

    cv_inner = KFold(n_splits=inner_cv_folds, shuffle=True, random_state=random_state)
    gridcv = GridSearchCV(
        pipeline,
        param_grid,
        scoring="accuracy",
        n_jobs=-1,
        cv=cv_inner,
        refit=True,
    )
    cv_outer = KFold(n_splits=outer_cv_folds, shuffle=True, random_state=random_state)
    scores: Dict[str, List[float]] = cross_validate(
        gridcv,
        X,
        y,
        cv=cv_outer,
        scoring=("accuracy", "neg_log_loss", "roc_auc_ovr"),
        n_jobs=-1,
    )

    return scores


def KFoldCV(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
    n_splits: int,
) -> Dict[str, List[Any]]:
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores: Dict[str, List[float]] = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=("accuracy", "neg_log_loss", "roc_auc_ovr"),
        n_jobs=-1,
    )

    return scores


def set_param_grid(model_name: str) -> dict[str, List[Any]]:
    param_grid: Dict[str, List[Any]] = dict()
    if model_name == "LogisticRegression":
        param_grid["clf__C"] = np.power(10.0, range(-2, 3))
        param_grid["clf__max_iter"] = list(range(500, 1000, 100))
    elif model_name == "RandomForestClassifier":
        param_grid["clf__n_estimators"] = list(range(100, 600, 100))
        param_grid["clf__max_features"] = ["sqrt", "log2", None]
    elif model_name == "ExtraTreeClassifier":
        param_grid["clf__n_estimators"] = list(range(100, 600, 100))
        param_grid["clf__max_features"] = ["sqrt", "log2", None]

    return param_grid


def get_tuned_model(
    pipeline: Pipeline,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
    n_splits: int,
) -> Tuple[Any, Any]:
    param_grid = set_param_grid(model_name)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    gridcv = GridSearchCV(
        pipeline,
        param_grid,
        scoring="accuracy",
        n_jobs=-1,
        cv=cv,
        refit=True,
    )
    gridcv.fit(X, y)
    click.echo(f"Best Parameters: {gridcv.best_params_}")
    return gridcv.best_estimator_, gridcv.best_params_
