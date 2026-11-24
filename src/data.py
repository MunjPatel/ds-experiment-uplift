"""Load Criteo Uplift v2.1 (Diemert et al., AdKDD 2018) with Hugging Face mirror."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"
HF_URL = (
    "https://huggingface.co/datasets/criteo/criteo-uplift/"
    "resolve/main/criteo-research-uplift-v2.1.csv.gz"
)
CRITEO_URL = "http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"
FEATURES = [f"f{i}" for i in range(12)]


def _download(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)


def ensure_criteo_gz(force: bool = False) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / "criteo-research-uplift-v2.1.csv.gz"
    if dest.exists() and dest.stat().st_size > 1_000_000 and not force:
        return dest
    last_err = None
    for url in (HF_URL, CRITEO_URL):
        try:
            print(f"Downloading Criteo uplift (~300MB) from {url} ...")
            _download(url, dest)
            if dest.stat().st_size > 1_000_000:
                (CACHE / "SOURCE.txt").write_text(
                    "Criteo Uplift Prediction Dataset v2.1\n"
                    "Diemert, Betlei, Renaudin, Amini — AdKDD 2018\n"
                    f"Source URL: {url}\n",
                    encoding="utf-8",
                )
                return dest
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise RuntimeError(f"Could not download Criteo uplift data: {last_err}")


def load_criteo_sample(n: int = 400_000, seed: int = 42, target: str = "visit") -> pd.DataFrame:
    """Stream-subsample rows without loading all 14M into memory."""
    gz = ensure_criteo_gz()
    rng = np.random.default_rng(seed)
    # Approximate total rows from dataset card
    total = 13_979_592
    keep_p = min(1.0, n / total * 1.35)  # slight oversample then trim
    chunks = []
    kept = 0
    with gzip.open(gz, "rt") as f:
        reader = pd.read_csv(f, chunksize=200_000)
        for chunk in reader:
            mask = rng.random(len(chunk)) < keep_p
            part = chunk.loc[mask]
            if len(part):
                chunks.append(part)
                kept += len(part)
            if kept >= n * 1.1:
                break
    df = pd.concat(chunks, ignore_index=True)
    if len(df) > n:
        # Stratify-ish trim preserving treatment mix
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    need = FEATURES + ["treatment", target]
    df = df[need].rename(columns={target: "outcome"})
    df["treatment"] = df["treatment"].astype(int)
    df["outcome"] = df["outcome"].astype(int)
    return df


def make_synthetic_rct(n: int = 80_000, seed: int = 0) -> pd.DataFrame:
    """Fallback RCT if download unavailable — documented as synthetic."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 12))
    # Heterogeneous effect: treatment helps when f0 high
    t = rng.binomial(1, 0.5, size=n)
    base = 1 / (1 + np.exp(-(0.3 * X[:, 0] - 0.2 * X[:, 1])))
    uplift = 0.08 * (X[:, 0] > 0.5).astype(float)
    p = np.clip(base + t * uplift, 0.01, 0.99)
    y = rng.binomial(1, p)
    cols = {f"f{i}": X[:, i] for i in range(12)}
    cols.update({"treatment": t, "outcome": y})
    return pd.DataFrame(cols)
