import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from src.model_training import (
    get_base_models,
    _get_param_grids,
    get_param_grids_for_random_search,
    RANDOM_STATE,
)


def get_stratified_kfold(n_splits=5, shuffle=True, random_state=RANDOM_STATE):
    return StratifiedKFold(
        n_splits=n_splits, shuffle=shuffle, random_state=random_state
    )


def tune_hyperparameters(
    X_train, y_train, model_name=None, method="grid", n_iter=50, n_splits=5,
    scoring="f1", random_state=RANDOM_STATE
):
    cv = get_stratified_kfold(n_splits=n_splits, random_state=random_state)

    if method == "random":
        param_grids = get_param_grids_for_random_search()
    else:
        param_grids = _get_param_grids()

    if model_name is not None:
        model_names = [model_name]
    else:
        model_names = list(param_grids.keys())

    base_models = get_base_models(random_state=random_state)
    results = {}

    for name in model_names:
        model = base_models[name]
        param_grid = param_grids[name]

        if method == "random":
            searcher = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                random_state=random_state,
                return_train_score=True,
            )
        else:
            searcher = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                return_train_score=True,
            )

        searcher.fit(X_train, y_train)

        results[name] = {
            "best_model": searcher.best_estimator_,
            "best_params": searcher.best_params_,
            "best_score": searcher.best_score_,
            "std_test_score": searcher.cv_results_["std_test_score"][
                searcher.best_index_
            ],
            "searcher": searcher,
        }

    return results


def get_tuned_models(X_train, y_train, n_splits=5, scoring="f1",
                     random_state=RANDOM_STATE):
    return tune_hyperparameters(
        X_train, y_train, model_name=None, method="grid",
        n_splits=n_splits, scoring=scoring, random_state=random_state,
    )
