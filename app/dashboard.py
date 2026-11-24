from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # local shared/ for standalone + Streamlit Cloud

from shared.plot_theme import COLORS, apply_plotly_theme  # noqa: E402

st.set_page_config(page_title="Experiment + Uplift Lab", layout="wide")
st.markdown(
    f"<style>.stApp {{ background-color: {COLORS['bg']}; color: {COLORS['text']}; }}</style>",
    unsafe_allow_html=True,
)
st.title("Trustworthy Experiments & Uplift Targeting")
st.caption("Peeking pitfalls · A/A health · T-learner uplift vs propensity")

out = ROOT / "outputs"
if not (out / "meta.json").exists():
    st.warning("Outputs missing. Run `python 07_ds_experiment_uplift/scripts/run_mvp.py` first.")
    st.stop()

meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
peek = pd.read_csv(out / "peeking_curve.csv")
qini = pd.read_csv(out / "qini_curves.csv")
top = pd.read_csv(out / "uplift_at_topk.csv")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{meta['n_rows']:,}")
c2.metric("ATE", f"{meta['ate']['ate']:.5f}")
c3.metric("Data", meta["data"])
c4.metric("Uplift@30%", f"{meta['uplift_at_30pct'].get('uplift', float('nan')):.5f}")

tab1, tab2, tab3 = st.tabs(["Diagnostics", "Qini / targeting", "Readout"])

with tab1:
    fig = px.line(peek, x="look", y="p_value", markers=True, title="Peeking under null (shuffled treatment)")
    fig.add_hline(y=0.05, line_dash="dash", line_color=COLORS["warn"])
    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.write("A/A simulation:", meta["aa_test"])
    st.write("SRM vs empirical assignment:", meta["srm_vs_empirical_assignment"])
    st.info("Airbnb/Kohavi lesson: continuous peeking can create false significance — fix the horizon in advance.")

with tab2:
    fig = px.line(
        qini,
        x="percentile",
        y="cum_incremental",
        color="strategy",
        markers=True,
        title="Cumulative incremental outcomes by targeting strategy",
    )
    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(top, use_container_width=True)
    st.caption("Propensity-to-convert ranks likely visitors; uplift ranks persuadable visitors.")

with tab3:
    if (out / "EXPERIMENT_READOUT.md").exists():
        st.markdown((out / "EXPERIMENT_READOUT.md").read_text(encoding="utf-8"))
