
# NFL Predictor v1

James Havrilla III

I built this model to pick the straight-up winner of NFL games using
only team and game data. Betting lines are left out on purpose, both as
inputs and as benchmarks.

## Results

| Holdout, 2023 to 2025 (854 games) | Accuracy | 95% CI         | Log loss |
|-----------------------------------|----------|----------------|----------|
| v1 ensemble                       | 63.2%    | 60.0% to 66.5% | 0.629    |
| Plain Elo rating                  | 64.1%    | 60.8% to 67.3% | 0.635    |
| Always pick the home team         | 54.9%    | 51.6% to 58.3% | 0.689    |

The final model is a soft-voting ensemble of tuned logistic regression,
gradient boosting and Extra-Trees. On picks it is statistically tied
with a well-tuned Elo rating, and its probabilities are slightly better
calibrated. The full write-up is in [docs/report.pdf](docs/report.pdf).

## Approach

The data is nflverse's schedules and weekly team stats from 1999 to
2026. Each game gets 20 home-minus-away features built only from games
played before kickoff: Elo ratings, rolling EPA and scoring form, rest,
division games, neutral sites and quarterback changes.

I validated with walk-forward cross-validation by season, training on
every earlier season and testing on the next, and held out 2023 to 2025
to score once at the end. Tuning was a randomized search over model and
feature settings together, selected on log loss.

## Setup

```         
pip install -r requirements.txt
```

## Weekly use

```         
python predict_week.py            # refresh data, retrain, pick the next week, score past picks
python predict_week.py --week 10  # pick a specific week
```

Picks are saved to `predictions/picks_<season>_wk<NN>.csv`. Every pick
is also kept in `predictions/log.csv` and scored once the game is final.

## Project layout

| Path | Contents |
|----|----|
| `NFL_Predictor_v1.ipynb` | Walkthrough of all eight steps |
| `nflv1/data.py` | Downloads and loads the nflverse data and removes betting columns |
| `nflv1/features.py` | Pre-kickoff features as a scikit-learn transformer |
| `nflv1/models.py` | Baselines and model pipelines |
| `nflv1/evaluation.py` | Walk-forward cross-validation, holdout metrics and the experiment log |
| `tune.py` | Randomized search over model and feature settings |
| `train_final.py` | Final model selection and the one-time holdout evaluation |
| `predict_week.py` | Weekly picks and monitoring |
| `models/nfl_v1_final.joblib` | Saved final pipeline and its scores |
| `results/` | Experiment log plus cross-validation, calibration and holdout results |
| `data/raw/` | Downloaded nflverse files |

## Rebuilding from scratch

```         
python tune.py         # about 15 minutes
python train_final.py  # about 5 minutes
```
