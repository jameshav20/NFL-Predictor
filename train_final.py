"""Select the final model with a rule set in advance, then evaluate on the holdout once.

The rule: among the three tuned models and their soft-voting ensemble, take the
one with the lowest walk-forward log loss on 2011 to 2022. Retrain it on 2001 to
2022, score it once on 2023 to 2025, and save it. Nothing is changed after the
holdout results are seen.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline

from nflv1 import data, evaluation as ev, features, models

VERSION = "v1.0"
g = data.load_games("data")
tg = data.load_team_games("data")
features.register(g, tg)
X, y = ev.modeling_frame(g, data.TRAIN_SEASONS)
cv = list(ev.walk_forward_splits(X["season"]))


def tuned(name, est, scale):
    p = joblib.load(f"results/best_params_{name}.joblib")
    pipe = models.make_pipeline(est, scale=scale)
    keep = {k: v for k, v in p.items() if k.startswith(("features__", "clf__")) and "__" in k}
    return pipe.set_params(**{k: v for k, v in keep.items() if k in pipe.get_params()})


cands = {
    "logistic": tuned("logistic", LogisticRegression(max_iter=3000), True),
    "hist_gb": tuned("hist_gb", HistGradientBoostingClassifier(random_state=42), False),
    "extra_trees": tuned("extra_trees", ExtraTreesClassifier(n_estimators=300, random_state=42), False),
}
cands["soft_vote"] = VotingClassifier([(k, clone(v)) for k, v in cands.items()], voting="soft")

cv_rows, oof = [], {}
for name, m in cands.items():
    r = ev.cv_evaluate(m, X, y)
    ev.log_experiment("results/experiments.csv", name, {"tuned": True}, r, "final_cv")
    cv_rows.append({"model": name, **r})
    print(name, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)}, flush=True)

cv_df = pd.DataFrame(cv_rows).sort_values("log_loss")
winner = cv_df.iloc[0]["model"]
print("\nSelected by rule:", winner)
cv_df.to_csv("results/final_cv.csv", index=False)

# Calibration of the selected model, from walk-forward out-of-fold predictions.
test_idx = np.concatenate([te for _, te in cv])
p_oof = np.zeros(len(X))
for tr, te in cv:
    mm = clone(cands[winner]).fit(X.iloc[tr], y.iloc[tr])
    p_oof[te] = mm.predict_proba(X.iloc[te])[:, 1]
frac, mean_pred = calibration_curve(y.iloc[test_idx], p_oof[test_idx], n_bins=10, strategy="quantile")
pd.DataFrame({"mean_predicted": mean_pred, "observed_home_win": frac}).to_csv(
    "results/calibration_cv.csv", index=False)

# Single holdout evaluation.
final = clone(cands[winner]).fit(X, y)
Xh, yh = ev.modeling_frame(g, data.HOLDOUT_SEASONS)
hold = {"final_" + winner: final.predict_proba(Xh)[:, 1]}
base_home = models.AlwaysHome().fit(X, y)
hold["always_home"] = base_home.predict_proba(Xh)[:, 1]
hold["elo_only"] = models.EloOnly().fit(X, y).predict_proba(Xh)[:, 1]

rows = []
for name, p in hold.items():
    r = ev.holdout_report(yh, p)
    rows.append({"model": name, **r})
    ev.log_experiment("results/experiments.csv", name, {}, r, "holdout")
hold_df = pd.DataFrame(rows)
print("\nHOLDOUT 2023-2025\n", hold_df.round(4).to_string(index=False))
hold_df.to_csv("results/holdout.csv", index=False)

# Per-season holdout accuracy for the final model.
ph = hold["final_" + winner]
per_season = (pd.DataFrame({"season": Xh["season"], "correct": ((ph >= 0.5) == yh).astype(int)})
              .groupby("season")["correct"].agg(["mean", "count"]))
per_season.to_csv("results/holdout_by_season.csv")
print(per_season)

# 2026 season to date. Used for monitoring only, never for decisions.
Xl, yl = ev.modeling_frame(g, [data.LIVE_SEASON])
live_model = clone(cands[winner]).fit(*ev.modeling_frame(g, range(1999, 2026)))
live = ev.holdout_report(yl, live_model.predict_proba(Xl)[:, 1])
print("\nLIVE 2026 to date:", {k: round(v, 4) for k, v in live.items()})

# Save the unfitted pipeline (predict_week.py refits it each week) along with its scores.
hr = hold_df.iloc[0].to_dict()
joblib.dump({"version": VERSION, "pipeline": clone(cands[winner]), "fitted_through_2022": final,
             "selected": winner, "holdout": hr, "live_2026": live,
             "cv": cv_df.to_dict("records")}, "models/nfl_v1_final.joblib")
json.dump({"selected": winner, "holdout": hr, "live_2026": live}, open("results/final_summary.json", "w"),
          default=float, indent=2)
