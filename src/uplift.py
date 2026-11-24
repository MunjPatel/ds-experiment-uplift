"""T-learner uplift model and Qini / cumulative gain evaluation."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def train_t_learner(X: pd.DataFrame, y: np.ndarray, t: np.ndarray, seed: int = 42):
    """Two separate outcome models: treated vs control. Uplift = μ1 - μ0."""
    X1, y1 = X[t == 1], y[t == 1]
    X0, y0 = X[t == 0], y[t == 0]
    m1 = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=seed,
        verbosity=-1,
    )
    m0 = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=seed + 1,
        verbosity=-1,
    )
    m1.fit(X1, y1)
    m0.fit(X0, y0)
    return m1, m0


def predict_uplift(m1, m0, X: pd.DataFrame) -> np.ndarray:
    return m1.predict_proba(X)[:, 1] - m0.predict_proba(X)[:, 1]


def propensity_score_model(X: pd.DataFrame, y: np.ndarray, seed: int = 0):
    """Baseline 'target converters' model — NOT uplift (important contrast)."""
    m = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=seed,
        verbosity=-1,
    )
    m.fit(X, y)
    return m


def qini_curve(y: np.ndarray, t: np.ndarray, uplift: np.ndarray, n_bins: int = 20) -> pd.DataFrame:
    """Cumulative incremental outcome when treating top uplift scores first."""
    order = np.argsort(-uplift)
    y, t, uplift = y[order], t[order], uplift[order]
    n = len(y)
    rows = []
    for i in range(1, n_bins + 1):
        k = int(i / n_bins * n)
        yk, tk = y[:k], t[:k]
        if (tk == 1).sum() == 0 or (tk == 0).sum() == 0:
            cont = 0.0
        else:
            # estimated incremental outcomes if we treated this top-k segment
            cont = (yk[tk == 1].mean() - yk[tk == 0].mean()) * k
        rows.append({"percentile": i / n_bins, "n": k, "cum_incremental": float(cont)})
    # random baseline: overall ATE * k
    ate = y[t == 1].mean() - y[t == 0].mean()
    for r in rows:
        r["random_incremental"] = float(ate * r["n"])
    return pd.DataFrame(rows)


def uplift_at_topk(y, t, uplift, frac=0.3) -> dict:
    k = max(int(frac * len(y)), 1)
    order = np.argsort(-uplift)[:k]
    yk, tk = y[order], t[order]
    if (tk == 1).sum() == 0 or (tk == 0).sum() == 0:
        return {"frac": frac, "uplift": np.nan, "n": k}
    u = float(yk[tk == 1].mean() - yk[tk == 0].mean())
    return {"frac": frac, "uplift": u, "n": k, "visit_treat": float(yk[tk == 1].mean()), "visit_control": float(yk[tk == 0].mean())}


def prepare_splits(X, y, t, test_size=0.3, seed=42):
    return train_test_split(X, y, t, test_size=test_size, random_state=seed, stratify=y)
