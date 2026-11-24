# Trustworthy Experiments & Uplift Targeting

**Role focus:** Data Scientist  
**Decision:** Who should receive a costly treatment / campaign offer?

## Problem

Average treatment effects hide heterogeneity. Product and growth teams need (1) trustworthy experiment diagnostics and (2) uplift models that target **incremental** responders — not always-converters.

## What I built

1. SRM check, ATE + CI, A/A false-positive simulation, peeking curve (Kohavi / Airbnb)
2. T-learner uplift (two LightGBM outcome models) on Criteo Uplift v2.1 sample
3. Contrast vs propensity-to-convert ranking (common mistake)
4. Qini / uplift@top-k tables + experiment readout memo
5. Streamlit dashboard

## Sources (verified)

- Diemert, E., Betlei, A., Renaudin, C. & Amini, M.-R. (2018). *A Large Scale Benchmark for Uplift Modeling.* AdKDD @ KDD.
- Kohavi, R., Henne, R. & Sommerfield, D. (2007). *Practical Guide to Controlled Experiments on the Web.*
- Airbnb Engineering (2014). *Experiments at Airbnb.*

## Reproduce

```bash
cd portfolio/projects
.venv\Scripts\activate
python 07_ds_experiment_uplift/scripts/run_mvp.py
streamlit run 07_ds_experiment_uplift/app/dashboard.py
```

First run downloads ~300MB from the Hugging Face `criteo/criteo-uplift` mirror (official Criteo dataset card). If download fails, the script falls back to a documented synthetic RCT so the portfolio still builds.



1. Push this repo to GitHub (already done if you cloned from MunjPatel/ds-experiment-uplift).
2. Go to https://share.streamlit.io/ → **New app**
3. Select repo `MunjPatel/ds-experiment-uplift`, branch `main`
4. Main file path: `app/dashboard.py`
5. Python version 3.11+ recommended → Deploy

The app reads artifacts under `outputs/`. Re-run `scripts/run_mvp.py` locally if you need to refresh them before pushing.
