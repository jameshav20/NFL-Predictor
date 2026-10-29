"""Weekly picks and performance monitoring.

Usage, from the project folder:
    python predict_week.py                 # refresh data, retrain, pick the next week
    python predict_week.py --no-refresh    # use the data already on disk
    python predict_week.py --week 9        # pick a specific week of the current season

Steps:
  1. Download the latest nflverse schedule and team stats (skipped with --no-refresh).
  2. Retrain the final pipeline on every completed game from 2001 on.
  3. Write picks for the week to predictions/picks_<season>_wk<NN>.csv and add
     them to predictions/log.csv.
  4. Score every logged pick whose game is final and print accuracy to date, so
     a drop in performance shows up quickly.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from nflv1 import data, evaluation as ev, features

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
MODEL_PATH = os.path.join(HERE, "models", "nfl_v1_final.joblib")
PRED_DIR = os.path.join(HERE, "predictions")
LOG_PATH = os.path.join(PRED_DIR, "log.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-refresh", action="store_true")
    ap.add_argument("--season", type=int, default=data.LIVE_SEASON)
    ap.add_argument("--week", type=int, default=None)
    args = ap.parse_args()

    if not args.no_refresh:
        print("Refreshing nflverse data ...")
        data.fetch_data(DATA_DIR, last_season=args.season)

    games = data.load_games(DATA_DIR)
    team_games = data.load_team_games(DATA_DIR)
    features.register(games, team_games)

    # Retrain on every completed game, using the tuned settings saved with the pipeline.
    bundle = joblib.load(MODEL_PATH)
    model = bundle["pipeline"]
    X_all, y_all = ev.modeling_frame(games, range(data.FIRST_SEASON, args.season + 1))
    model.fit(X_all, y_all)

    season_games = games[games["season"] == args.season]
    upcoming = season_games[~season_games["played"]]
    if args.week is None:
        if upcoming.empty:
            print("No unplayed games left this season.")
            return
        week = int(upcoming["week"].min())
    else:
        week = args.week
    slate = season_games[season_games["week"] == week]

    p_home = model.predict_proba(slate[["game_id", "season"]])[:, 1]
    picks = pd.DataFrame({
        "game_id": slate["game_id"].to_numpy(),
        "season": args.season,
        "week": week,
        "gameday": slate["gameday"].dt.date.to_numpy(),
        "away_team": slate["away_team"].to_numpy(),
        "home_team": slate["home_team"].to_numpy(),
        "home_win_prob": np.round(p_home, 3),
        "pick": np.where(p_home >= 0.5, slate["home_team"], slate["away_team"]),
        "confidence": np.round(np.maximum(p_home, 1 - p_home), 3),
        "model_version": bundle["version"],
        "predicted_at": datetime.now().isoformat(timespec="seconds"),
    }).sort_values("confidence", ascending=False)

    os.makedirs(PRED_DIR, exist_ok=True)
    out = os.path.join(PRED_DIR, f"picks_{args.season}_wk{week:02d}.csv")
    picks.to_csv(out, index=False)
    log = pd.read_csv(LOG_PATH) if os.path.exists(LOG_PATH) else pd.DataFrame()
    log = pd.concat([log, picks], ignore_index=True)
    log = log.drop_duplicates(subset=["game_id"], keep="last")   # keep the latest pick per game
    log.to_csv(LOG_PATH, index=False)

    print(f"\nWeek {week} picks ({args.season}) saved to {out}")
    print(picks[["away_team", "home_team", "pick", "confidence"]].to_string(index=False))

    # Score logged picks whose games are now final.
    res = games.set_index("game_id")
    scored = log[log["game_id"].isin(res.index[res["played"] & ~res["tie"]])].copy()
    if len(scored):
        winner = np.where(res.loc[scored["game_id"], "home_win"] == 1,
                          res.loc[scored["game_id"], "home_team"],
                          res.loc[scored["game_id"], "away_team"])
        scored["correct"] = (scored["pick"].to_numpy() == winner)
        acc = scored["correct"].mean()
        print(f"\nMonitoring: {scored['correct'].sum()}/{len(scored)} logged picks correct "
              f"({acc:.1%}). Holdout benchmark: {bundle['holdout']['accuracy']:.1%}.")
        by_week = scored.groupby(["season", "week"])["correct"].agg(["sum", "count", "mean"])
        print(by_week.tail(6).to_string())


if __name__ == "__main__":
    main()
