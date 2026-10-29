"""Randomized search over model and feature-preparation hyperparameters.

I select on log loss and record accuracy alongside it. Accuracy moves in steps
and swings about two points from season to season, so selecting on it would
mostly chase noise.
"""
import json
import sys

import joblib
import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

from nflv1 import data, evaluation as ev, features, models

g = data.load_games("data")
tg = data.load_team_games("data")
features.register(g, tg)
X, y = ev.modeling_frame(g, data.TRAIN_SEASONS)
cv = list(ev.walk_forward_splits(X["season"]))

feature_space = {
    "features__elo_k": uniform(10, 25),          # 10 to 35
    "features__elo_hfa": uniform(10, 70),        # 10 to 80 Elo points
    "features__elo_revert": uniform(0.1, 0.4),   # 0.1 to 0.5
    "features__form_halflife": uniform(2, 14),   # 2 to 16 games
    "features__use_form": [True, False],
    "features__use_context": [True, False],
}

searches = {
    "logistic": (models.make_pipeline(LogisticRegression(max_iter=3000)),
                 {**feature_space, "clf__C": loguniform(1e-3, 10)}, 60),
    "hist_gb": (models.make_pipeline(HistGradientBoostingClassifier(random_state=42), scale=False),
                {**feature_space,
                 "clf__learning_rate": loguniform(0.01, 0.2),
                 "clf__max_iter": randint(50, 400),
                 "clf__max_leaf_nodes": randint(3, 16),
                 "clf__min_samples_leaf": randint(20, 200),
                 "clf__l2_regularization": loguniform(1e-3, 10)}, 40),
    "extra_trees": (models.make_pipeline(ExtraTreesClassifier(n_estimators=300, random_state=42),
                                         scale=False),
                    {**feature_space,
                     "clf__min_samples_leaf": randint(10, 120),
                     "clf__max_features": uniform(0.2, 0.8)}, 20),
}

which = sys.argv[1:] or list(searches)
for name in which:
    pipe, space, n_iter = searches[name]
    rs = RandomizedSearchCV(pipe, space, n_iter=n_iter, cv=cv, random_state=42,
                            scoring=ev.SCORING, refit="neg_log_loss", n_jobs=None)
    rs.fit(X, y)
    r = rs.cv_results_
    for i in range(len(r["params"])):
        ev.log_experiment("results/experiments.csv", name, r["params"][i], {
            "accuracy": r["mean_test_accuracy"][i], "accuracy_std": r["std_test_accuracy"][i],
            "log_loss": -r["mean_test_neg_log_loss"][i], "log_loss_std": r["std_test_neg_log_loss"][i],
            "brier": -r["mean_test_neg_brier_score"][i]}, "tune")
    b = rs.best_index_
    best = {"params": rs.best_params_, "accuracy": r["mean_test_accuracy"][b],
            "accuracy_std": r["std_test_accuracy"][b], "log_loss": -r["mean_test_neg_log_loss"][b],
            "brier": -r["mean_test_neg_brier_score"][b]}
    print(name, json.dumps(best, default=float), flush=True)
    joblib.dump(rs.best_estimator_.get_params(), f"results/best_params_{name}.joblib")
