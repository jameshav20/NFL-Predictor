"""Download, load and clean the nflverse data.

Betting lines are excluded on purpose. Every market-derived column is dropped as
soon as the schedule is loaded, so nothing downstream can use it.
"""
from __future__ import annotations

import glob
import os
import urllib.request

import numpy as np
import pandas as pd

BASE = "https://github.com/nflverse/nflverse-data/releases/download"
SCHEDULE_URL = f"{BASE}/schedules/games.csv"
TEAM_WEEK_URL = BASE + "/stats_team/stats_team_week_{season}.csv"

FIRST_SEASON = 1999

# Schedule columns that come from betting markets. Dropped on load.
BETTING_COLS = [
    "away_moneyline", "home_moneyline", "spread_line", "away_spread_odds",
    "home_spread_odds", "total_line", "under_odds", "over_odds",
]

# Relocated franchises share one code so their ratings carry over.
FRANCHISE = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

# Season split, set before any exploration and not changed afterward.
TRAIN_SEASONS = range(1999, 2023)     # exploration, CV, tuning
HOLDOUT_SEASONS = range(2023, 2026)   # touched once, at the end
LIVE_SEASON = 2026                    # monitoring only


def fetch_data(data_dir: str, last_season: int = LIVE_SEASON) -> None:
    """Download the schedule and weekly team stats. Re-run to refresh."""
    raw = os.path.join(data_dir, "raw")
    os.makedirs(raw, exist_ok=True)
    urllib.request.urlretrieve(SCHEDULE_URL, os.path.join(raw, "games.csv"))
    for season in range(FIRST_SEASON, last_season + 1):
        urllib.request.urlretrieve(
            TEAM_WEEK_URL.format(season=season),
            os.path.join(raw, f"stats_team_week_{season}.csv"),
        )


def load_games(data_dir: str) -> pd.DataFrame:
    """Schedule and results, one row per game, with betting columns removed."""
    g = pd.read_csv(os.path.join(data_dir, "raw", "games.csv"))
    g = g.drop(columns=[c for c in BETTING_COLS if c in g.columns])
    for col in ("home_team", "away_team"):
        g[col] = g[col].replace(FRANCHISE)
    g["gameday"] = pd.to_datetime(g["gameday"])
    g["played"] = g["home_score"].notna()
    g["home_win"] = np.where(g["played"], (g["result"] > 0).astype(float), np.nan)
    g["tie"] = g["played"] & (g["result"] == 0)
    g["neutral"] = (g["location"] == "Neutral").astype(int)
    g["playoff"] = (g["game_type"] != "REG").astype(int)
    g = g.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    assert not set(BETTING_COLS) & set(g.columns), "betting column leaked"
    return g


def load_team_games(data_dir: str) -> pd.DataFrame:
    """One row per team per game: the team's offense plus its opponent's offense,
    which serves as the team's defense."""
    files = sorted(glob.glob(os.path.join(data_dir, "raw", "stats_team_week_*.csv")))
    s = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    s = s.dropna(subset=["team", "game_id"])
    s["team"] = s["team"].replace(FRANCHISE)
    s = s.drop_duplicates(subset=["game_id", "team"], keep="first")

    plays = s["attempts"] + s["sacks_suffered"] + s["carries"]
    out = pd.DataFrame({
        "game_id": s["game_id"],
        "team": s["team"],
        "off_plays": plays,
        "off_epa": s["passing_epa"].fillna(0) + s["rushing_epa"].fillna(0),
        "pass_epa": s["passing_epa"],
        "dropbacks": s["attempts"] + s["sacks_suffered"],
        "rush_epa": s["rushing_epa"],
        "carries": s["carries"],
        "sacks": s["sacks_suffered"],
        "turnovers": s["passing_interceptions"].fillna(0)
        + s["sack_fumbles_lost"].fillna(0)
        + s["rushing_fumbles_lost"].fillna(0)
        + s["receiving_fumbles_lost"].fillna(0),
        "penalty_yards": s["penalty_yards"],
    })
    opp = out.rename(columns=lambda c: c if c == "game_id" else f"opp_{c}")
    pairs = out.merge(opp, on="game_id")
    pairs = pairs[pairs["team"] != pairs["opp_team"]]
    return pairs.reset_index(drop=True)
