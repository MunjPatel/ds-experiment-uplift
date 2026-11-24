# Experiment Readout — Uplift Targeting Demo

**Dataset:** criteo_v2.1_sample (n=350,000)  
**Primary metric:** visit (or synthetic outcome)

## Ship / no-ship diagnostics
- Overall ATE: **0.00984** (95% CI [0.00806, 0.01162])
- A/A false-positive rate in sims: **4.00%** (target ≈ 5%)
- Peeking demo ever crossed p<0.05 under null shuffle: **True**
- SRM vs empirical assignment rate: p=1 (flag=False)

## Targeting recommendation
Treat the **top 30%** by T-learner uplift score rather than everyone or high-propensity converters.
- Segment uplift@30%: **0.029332322454325793**
- Overall ATE (random targeting equivalent): **0.00984**
- Demo cost per incremental visit @ top30 (cost=1 unit): **34.09208396495466**

## Caution
Propensity-to-convert ≠ uplift. Always-converters look “good” on propensity but waste budget.
Criteo features are anonymized projections from incrementality tests — cite Diemert et al. (2018).
