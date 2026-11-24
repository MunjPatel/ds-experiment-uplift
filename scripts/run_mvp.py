from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import FEATURES, load_criteo_sample, make_synthetic_rct  # noqa: E402
from diagnostics import aa_false_positive_rate, ate_with_ci, peeking_simulation, sample_ratio_mismatch  # noqa: E402
from uplift import (  # noqa: E402
    prepare_splits,
    propensity_score_model,
    predict_uplift,
    qini_curve,
    train_t_learner,
    uplift_at_topk,
)


def main():
    out = ROOT / "outputs"
    out.mkdir(parents=True, exist_ok=True)

    data_note = "criteo_v2.1_sample"
    try:
        df = load_criteo_sample(n=350_000, seed=42, target="visit")
    except Exception as exc:  # noqa: BLE001
        print("Criteo download failed; using synthetic RCT fallback:", exc)
        df = make_synthetic_rct(n=80_000, seed=42)
        data_note = "synthetic_rct_fallback"

    y = df["outcome"].to_numpy()
    t = df["treatment"].to_numpy()
    X = df[FEATURES]

    # --- Experiment diagnostics ---
    expected_p = float(t.mean())  # Criteo is unbalanced (~0.85); use empirical assignment rate
    srm = sample_ratio_mismatch(int((t == 1).sum()), int((t == 0).sum()), expected_p_treat=expected_p)
    # For Criteo, SRM vs 50/50 would false-alarm — also report vs empirical design
    srm_50 = sample_ratio_mismatch(int((t == 1).sum()), int((t == 0).sum()), expected_p_treat=0.5)
    ate = ate_with_ci(y, t)
    peek = peeking_simulation(y, t, n_looks=25, seed=7)
    aa = aa_false_positive_rate(y, n_sims=150, seed=3)

    peek.to_csv(out / "peeking_curve.csv", index=False)
    pd.DataFrame([srm, {**srm_50, "note": "vs_0.5_for_illustration"}]).to_csv(out / "srm.csv", index=False)

    # --- Uplift modeling ---
    Xtr, Xte, ytr, yte, ttr, tte = prepare_splits(X, y, t, test_size=0.3, seed=42)
    m1, m0 = train_t_learner(Xtr, ytr, ttr)
    uplift = predict_uplift(m1, m0, Xte)

    # Propensity-to-convert baseline (often confused with uplift)
    prop_m = propensity_score_model(Xtr, ytr)
    prop_score = prop_m.predict_proba(Xte)[:, 1]

    qini_u = qini_curve(yte, tte, uplift)
    qini_p = qini_curve(yte, tte, prop_score)
    qini_u["strategy"] = "t_learner_uplift"
    qini_p["strategy"] = "propensity_to_convert"
    qini = pd.concat([qini_u, qini_p], ignore_index=True)
    qini.to_csv(out / "qini_curves.csv", index=False)

    top = pd.DataFrame(
        [
            {**uplift_at_topk(yte, tte, uplift, frac=f), "strategy": "t_learner_uplift"}
            for f in (0.1, 0.2, 0.3, 0.5)
        ]
        + [
            {**uplift_at_topk(yte, tte, prop_score, frac=f), "strategy": "propensity_to_convert"}
            for f in (0.1, 0.2, 0.3, 0.5)
        ]
    )
    top.to_csv(out / "uplift_at_topk.csv", index=False)

    # Budget story: treat top 30% by uplift vs random
    u30 = uplift_at_topk(yte, tte, uplift, frac=0.3)
    random_ate = ate["ate"]
    assumed_cpa = 1.0  # cost per treatment (demo units)
    cost_per_incr = assumed_cpa / u30["uplift"] if u30["uplift"] and u30["uplift"] > 0 else np.nan

    meta = {
        "data": data_note,
        "n_rows": int(len(df)),
        "outcome_rate": float(y.mean()),
        "treatment_rate": float(t.mean()),
        "ate": ate,
        "srm_vs_empirical_assignment": srm,
        "srm_vs_50_50_illustration": srm_50,
        "aa_test": aa,
        "peeking_ever_significant": bool(peek["significant_at_0.05"].any()),
        "uplift_at_30pct": u30,
        "random_ate": random_ate,
        "assumed_treatment_cost": assumed_cpa,
        "cost_per_incremental_visit_top30": float(cost_per_incr) if cost_per_incr == cost_per_incr else None,
        "sources": [
            "Diemert et al. (2018), A Large Scale Benchmark for Uplift Modeling, AdKDD@KDD",
            "Kohavi, Henne, Sommerfield (2007), Practical Guide to Controlled Experiments on the Web",
            "Airbnb Engineering (2014), Experiments at Airbnb",
        ],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    readout = f"""# Experiment Readout — Uplift Targeting Demo

**Dataset:** {data_note} (n={len(df):,})  
**Primary metric:** visit (or synthetic outcome)

## Ship / no-ship diagnostics
- Overall ATE: **{ate['ate']:.5f}** (95% CI [{ate['ci_low']:.5f}, {ate['ci_high']:.5f}])
- A/A false-positive rate in sims: **{aa['empirical_significant_rate']:.2%}** (target ≈ 5%)
- Peeking demo ever crossed p<0.05 under null shuffle: **{bool(peek['significant_at_0.05'].any())}**
- SRM vs empirical assignment rate: p={srm['p_value']:.3g} (flag={srm['srm_flag']})

## Targeting recommendation
Treat the **top 30%** by T-learner uplift score rather than everyone or high-propensity converters.
- Segment uplift@30%: **{u30.get('uplift')}**
- Overall ATE (random targeting equivalent): **{random_ate:.5f}**
- Demo cost per incremental visit @ top30 (cost=1 unit): **{cost_per_incr}**

## Caution
Propensity-to-convert ≠ uplift. Always-converters look “good” on propensity but waste budget.
Criteo features are anonymized projections from incrementality tests — cite Diemert et al. (2018).
"""
    (out / "EXPERIMENT_READOUT.md").write_text(readout, encoding="utf-8")
    print("P7 experiment/uplift MVP complete ->", out)
    print(pd.Series({"n": len(df), "ate": ate["ate"], "data": data_note}))


if __name__ == "__main__":
    main()
