"""Pre-kickoff game features, packaged as a scikit-learn transformer.

Every feature for a game is computed only from games played before it. Elo is
updated after each game and rolling form is shifted back one game. Because both
only look backward, I compute them once over the full history and look them up
by game_id, so a test-season game never sees its own result.

The Elo settings and the form half-life are hyperparameters, so the
randomized search can tune data preparation along with the model.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# The data lives in a module-level registry so the transformer stays cheap to clone.
_REGISTRY: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}


def register(games: pd.DataFrame, team_games: pd.DataFrame, name: str = "default") -> None:
    _REGISTRY[name] = (games, team_games)
    _elo_table.cache_clear()
    _form_table.cache_clear()
    _context_table.cache_clear()


# --------------------------------------------------------------------------- Elo
@lru_cache(maxsize=64)
def _elo_table(name: str, k: float, hfa: float, revert: float) -> pd.DataFrame:
    """FiveThirtyEight-style Elo. Returns pre-game ratings for every game, played or not."""
    games, _ = _REGISTRY[name]
    rating: dict[str, float] = {}
    last_season: dict[str, int] = {}
    rows = []
    for g in games.itertuples(index=False):
        for t in (g.home_team, g.away_team):
            if t not in rating:
                rating[t] = 1500.0
                last_season[t] = g.season
            elif last_season[t] != g.season:          # new season, so regress toward the mean
                rating[t] = 1505.0 * revert + rating[t] * (1 - revert)
                last_season[t] = g.season
        rh, ra = rating[g.home_team], rating[g.away_team]
        bump = 0.0 if g.neutral else hfa
        p_home = 1.0 / (1.0 + 10 ** (-(rh + bump - ra) / 400.0))
        rows.append((g.game_id, rh, ra, p_home))
        if g.played:
            margin = g.home_score - g.away_score
            actual = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
            winner_diff = (rh + bump - ra) if margin >= 0 else (ra - rh - bump)
            mult = math.log(abs(margin) + 1) * 2.2 / (winner_diff * 0.001 + 2.2)
            delta = k * mult * (actual - p_home)
            rating[g.home_team] += delta
            rating[g.away_team] -= delta
    return pd.DataFrame(rows, columns=["game_id", "elo_home", "elo_away", "elo_prob"]).set_index("game_id")


# ------------------------------------------------------------------ rolling form
FORM_STATS = {
    # feature name: (numerator column, denominator column or None)
    "pts_for": ("points_for", None),
    "pts_against": ("points_against", None),
    "off_epa_play": ("off_epa", "off_plays"),
    "def_epa_play": ("opp_off_epa", "opp_off_plays"),
    "pass_epa_db": ("pass_epa", "dropbacks"),
    "rush_epa_car": ("rush_epa", "carries"),
    "def_pass_epa_db": ("opp_pass_epa", "opp_dropbacks"),
    "sack_rate": ("sacks", "dropbacks"),
    "def_sack_rate": ("opp_sacks", "opp_dropbacks"),
    "to_margin": ("to_margin", None),
    "pen_yds": ("penalty_yards", None),
}


def _long_team_table(games: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    """Two rows per scheduled game, one per team, with the game's stats if it was played."""
    cols = ["game_id", "season", "gameday", "played"]
    home = games[cols + ["home_team", "home_score", "away_score", "home_qb_id"]].rename(
        columns={"home_team": "team", "home_score": "points_for",
                 "away_score": "points_against", "home_qb_id": "qb_id"})
    away = games[cols + ["away_team", "away_score", "home_score", "away_qb_id"]].rename(
        columns={"away_team": "team", "away_score": "points_for",
                 "home_score": "points_against", "away_qb_id": "qb_id"})
    long = pd.concat([home, away], ignore_index=True)
    long = long.merge(team_games, on=["game_id", "team"], how="left")
    long["to_margin"] = long["opp_turnovers"] - long["turnovers"]
    stat_cols = {c for pair in FORM_STATS.values() for c in pair if c}
    long.loc[~long["played"], list(stat_cols)] = np.nan
    return long.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)


@lru_cache(maxsize=64)
def _form_table(name: str, halflife: float) -> pd.DataFrame:
    """Exponentially weighted pre-game averages for each team, carried across seasons."""
    games, team_games = _REGISTRY[name]
    long = _long_team_table(games, team_games)
    grp = long.groupby("team", sort=False)

    def ewm_prev(col: str) -> pd.Series:
        # Weighted mean through each row, shifted so a game only sees earlier games.
        return grp[col].transform(
            lambda s: s.ewm(halflife=halflife, ignore_na=True).mean().shift(1))

    out = long[["game_id", "team"]].copy()
    for feat, (num, den) in FORM_STATS.items():
        if den is None:
            out[feat] = ewm_prev(num)
        else:
            out[feat] = ewm_prev(num) / ewm_prev(den)
    return out.set_index(["game_id", "team"])


@lru_cache(maxsize=8)
def _context_table(name: str) -> pd.DataFrame:
    """Context known before kickoff: rest, division, site, playoff and QB change."""
    games, team_games = _REGISTRY[name]
    long = _long_team_table(games, team_games)
    long["prev_qb"] = long.groupby("team")["qb_id"].shift(1)
    long["qb_change"] = ((long["qb_id"] != long["prev_qb"]) & long["prev_qb"].notna()
                         & long["qb_id"].notna()).astype(int)
    # Games played so far this season. Form is noisier early in the year.
    long["season_games"] = long.groupby(["team", "season"]).cumcount()
    qb = long.set_index(["game_id", "team"])[["qb_change", "season_games"]]
    g = games.set_index("game_id")
    ctx = pd.DataFrame({
        "rest_diff": (g["home_rest"] - g["away_rest"]).clip(-14, 14),
        "div_game": g["div_game"],
        "neutral": g["neutral"],
        "playoff": g["playoff"],
    })
    ctx["home_qb_change"] = qb.loc[list(zip(g.index, g["home_team"])), "qb_change"].to_numpy()
    ctx["away_qb_change"] = qb.loc[list(zip(g.index, g["away_team"])), "qb_change"].to_numpy()
    ctx["season_games"] = qb.loc[list(zip(g.index, g["home_team"])), "season_games"].to_numpy()
    return ctx


class GameFeatures(BaseEstimator, TransformerMixin):
    """Turn rows with a game_id into home-minus-away features.

    Parameters
    ----------
    elo_k : Elo update size per game.
    elo_hfa : home-field bump, in Elo points.
    elo_revert : share of each rating pulled back to the mean at the start of a season.
    form_halflife : half-life of the rolling form averages, in games.
    use_elo, use_form, use_context : switches for each feature group, so the
        search can test whether a group helps.
    """

    def __init__(self, elo_k=20.0, elo_hfa=48.0, elo_revert=1 / 3, form_halflife=6.0,
                 use_elo=True, use_form=True, use_context=True, source="default"):
        self.elo_k = elo_k
        self.elo_hfa = elo_hfa
        self.elo_revert = elo_revert
        self.form_halflife = form_halflife
        self.use_elo = use_elo
        self.use_form = use_form
        self.use_context = use_context
        self.source = source

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        games, _ = _REGISTRY[self.source]
        ids = pd.Index(X["game_id"])
        g = games.set_index("game_id").loc[ids]
        parts = []
        if self.use_elo:
            elo = _elo_table(self.source, float(self.elo_k), float(self.elo_hfa),
                             float(self.elo_revert)).loc[ids]
            parts.append(pd.DataFrame({
                "elo_diff": (elo["elo_home"] - elo["elo_away"]).to_numpy(),
                "elo_prob": elo["elo_prob"].to_numpy(),
            }, index=ids))
        if self.use_form:
            form = _form_table(self.source, float(self.form_halflife))
            h = form.loc[list(zip(ids, g["home_team"]))].to_numpy()
            a = form.loc[list(zip(ids, g["away_team"]))].to_numpy()
            parts.append(pd.DataFrame(h - a, index=ids,
                                      columns=[f"{c}_diff" for c in form.columns]))
        if self.use_context:
            parts.append(_context_table(self.source).loc[ids])
        out = pd.concat(parts, axis=1)
        self.feature_names_ = list(out.columns)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_, dtype=object)
