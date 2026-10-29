"""Walk-forward cross-validation, holdout metrics and the experiment log.

Shuffled K-fold cross-validation would let the model train on future games, so
each fold here trains on every earlier season and tests on the next one.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import cross_validate

from . import data as d

BURN_IN = 2  # the first two seasons only warm up Elo and form and are never trained on
SCORING = {"accuracy": "accuracy", "neg_log_loss": "neg_log_loss",
           "neg_brier_score": "neg_brier_score"}


def modeling_frame(games: pd.DataFrame, seasons) -> tuple[pd.DataFrame, pd.Series]:
    """Played, non-tied games in the given seasons, as X (game_id, season) and y."""
    m = games["played"] & ~games["tie"] & games["season"].isin(list(seasons))
    m &= games["season"] >= d.FIRST_SEASON + BURN_IN
    df = games.loc[m, ["game_id", "season"]].reset_index(drop=True)
    y = games.loc[m, "home_win"].astype(int).reset_index(drop=True)
    return df, y


def walk_forward_splits(seasons: pd.Series, first_test: int = 2011, last_test: int = 2022):
    """Yield (train_idx, test_idx) pairs that train on every season before s and test on s."""
    seasons = np.asarray(seasons)
    for s in range(first_test, last_test + 1):
        tr = np.where(seasons < s)[0]
        te = np.where(seasons == s)[0]
        if len(te):
            yield tr, te


def cv_evaluate(model, X: pd.DataFrame, y: pd.Series, **kw) -> dict:
    """Run walk-forward CV and return the mean and std of accuracy, log loss and Brier score."""
    t0 = time.time()
    res = cross_validate(model, X, y, cv=list(walk_forward_splits(X["season"], **kw)),
                         scoring=SCORING, n_jobs=None)  # one process, since the feature cache is in memory
    acc, ll, br = res["test_accuracy"], -res["test_neg_log_loss"], -res["test_neg_brier_score"]
    return {"accuracy": acc.mean(), "accuracy_std": acc.std(),
            "log_loss": ll.mean(), "log_loss_std": ll.std(),
            "brier": br.mean(), "brier_std": br.std(),
            "folds": len(acc), "seconds": round(time.time() - t0, 1)}


def holdout_report(y_true, proba) -> dict:
    """Holdout metrics, with a 95% confidence interval on accuracy."""
    y_true = np.asarray(y_true)
    pred = (proba >= 0.5).astype(int)
    correct = (pred == y_true).astype(float)
    n = len(correct)
    lo, hi = stats.t.interval(0.95, n - 1, loc=correct.mean(), scale=stats.sem(correct))
    return {"n": n, "accuracy": correct.mean(), "acc_ci_low": lo, "acc_ci_high": hi,
            "log_loss": log_loss(y_true, proba), "brier": brier_score_loss(y_true, proba)}


def log_experiment(path: str, name: str, params: dict, result: dict, stage: str) -> None:
    """Append one run to the experiment log so every model I try is on record."""
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "stage": stage,
           "model": name, "params": json.dumps(params, default=str)}
    row.update({k: (round(v, 5) if isinstance(v, float) else v) for k, v in result.items()})
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
