"""Model pipelines and simple baselines."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import GameFeatures


class AlwaysHome(BaseEstimator, ClassifierMixin):
    """Baseline that always picks the home team, at the training-set home win rate."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.p_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.p_)
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class EloOnly(BaseEstimator, ClassifierMixin):
    """Baseline that uses the raw Elo win probability, with no learning."""

    def __init__(self, elo_k=20.0, elo_hfa=48.0, elo_revert=1 / 3):
        self.elo_k, self.elo_hfa, self.elo_revert = elo_k, elo_hfa, elo_revert

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        f = GameFeatures(self.elo_k, self.elo_hfa, self.elo_revert,
                         use_form=False, use_context=False).transform(X)
        p = f["elo_prob"].to_numpy()
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_pipeline(clf, scale: bool = True, **feature_kw) -> Pipeline:
    steps = [("features", GameFeatures(**feature_kw)),
             ("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", clf))
    return Pipeline(steps)


def shortlist_candidates(seed: int = 42) -> dict:
    """Quick models from several families, mostly on default settings, for the shortlist.

    KNN uses k=50 instead of the default of 5, which is far too noisy for this data.
    """
    return {
        "always_home": AlwaysHome(),
        "elo_only": EloOnly(),
        "logistic": make_pipeline(LogisticRegression(max_iter=2000)),
        "naive_bayes": make_pipeline(GaussianNB()),
        "knn": make_pipeline(KNeighborsClassifier(n_neighbors=50)),
        "svc_rbf": make_pipeline(SVC(probability=True, random_state=seed)),
        "random_forest": make_pipeline(RandomForestClassifier(n_estimators=300, random_state=seed),
                                       scale=False),
        "extra_trees": make_pipeline(ExtraTreesClassifier(n_estimators=300, random_state=seed),
                                     scale=False),
        "hist_gb": make_pipeline(HistGradientBoostingClassifier(random_state=seed), scale=False),
    }
