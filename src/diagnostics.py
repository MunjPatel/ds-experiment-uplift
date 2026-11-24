"""Trustworthy A/B diagnostics (Kohavi / Airbnb pitfalls)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def sample_ratio_mismatch(n_treat: int, n_control: int, expected_p_treat: float = 0.5) -> dict:
    """Chi-square SRM check on assignment counts."""
    n = n_treat + n_control
    expected = np.array([expected_p_treat, 1 - expected_p_treat]) * n
    observed = np.array([n_treat, n_control], dtype=float)
    chi2, p = stats.chisquare(observed, expected)
    return {
        "n_treat": int(n_treat),
        "n_control": int(n_control),
        "expected_p_treat": expected_p_treat,
        "chi2": float(chi2),
        "p_value": float(p),
        "srm_flag": bool(p < 0.001),
    }


def ate_with_ci(y: np.ndarray, t: np.ndarray, alpha: float = 0.05) -> dict:
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=int)
    yt, yc = y[t == 1], y[t == 0]
    ate = float(yt.mean() - yc.mean())
    se = float(np.sqrt(yt.var(ddof=1) / len(yt) + yc.var(ddof=1) / len(yc)))
    z = stats.norm.ppf(1 - alpha / 2)
    return {
        "ate": ate,
        "se": se,
        "ci_low": ate - z * se,
        "ci_high": ate + z * se,
        "n_treat": int((t == 1).sum()),
        "n_control": int((t == 0).sum()),
        "mean_treat": float(yt.mean()),
        "mean_control": float(yc.mean()),
    }


def peeking_simulation(
    y: np.ndarray,
    t: np.ndarray,
    n_looks: int = 20,
    alpha: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Show how continuous peeking inflates false significance (Airbnb / Kohavi lesson)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    y, t = y[idx], t[idx]
    rows = []
    # Under null: shuffle treatment labels once for A/A-style peeking demo on same outcomes
    t_null = rng.permutation(t)
    for k in range(1, n_looks + 1):
        n = int(k / n_looks * len(y))
        n = max(n, 50)
        res = ate_with_ci(y[:n], t_null[:n], alpha=alpha)
        # naive z-test peek
        z = res["ate"] / res["se"] if res["se"] > 0 else 0.0
        p = float(2 * (1 - stats.norm.cdf(abs(z))))
        rows.append(
            {
                "look": k,
                "n": n,
                "ate": res["ate"],
                "p_value": p,
                "significant_at_0.05": p < alpha,
            }
        )
    return pd.DataFrame(rows)


def aa_false_positive_rate(y: np.ndarray, n_sims: int = 200, alpha: float = 0.05, seed: int = 1) -> dict:
    """A/A tests should be significant ~alpha of the time if the platform is healthy."""
    rng = np.random.default_rng(seed)
    hits = 0
    for i in range(n_sims):
        t = rng.integers(0, 2, size=len(y))
        res = ate_with_ci(y, t, alpha=alpha)
        z = res["ate"] / res["se"] if res["se"] > 0 else 0.0
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        hits += int(p < alpha)
    return {"n_sims": n_sims, "alpha": alpha, "empirical_significant_rate": hits / n_sims}
