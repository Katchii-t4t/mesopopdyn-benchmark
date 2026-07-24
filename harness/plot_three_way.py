"""
plot_three_way.py -- the three-way validation:
    published PD14 rates  vs  our mesoscopic sim  vs  our microscopic sim.

The published rates ("existing data") come from the ORIGINAL microscopic
Potjans-Diesmann (2014) simulation. Here we close the loop: our own
mesoscopic model and our own (sparse, real-connectivity) microscopic model
are both compared to the published values AND to each other, per population.

Published rates: PD14 (from model_setup).
Mesoscopic: run fresh here (Numba, full scale).
Microscopic: read from results/benchmark_c_latest.json (sparse micro_c, full scale).

Produces figures/three_way_validation.png and prints the pairwise errors.

Usage:  python plot_three_way.py

Karthik Senthil Ganesh, TU Berlin / BCCN internship, 2026.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model_setup import (
    build_params, build_step_stimulus, PD14_TARGET_RATES, POP_LABELS,
)
from sim_meso import simulate

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# canonical plotting order (interleaved: groups E,I within each layer)
ORDER = ['L23e', 'L23i', 'L4e', 'L4i', 'L5e', 'L5i', 'L6e', 'L6i']


def published_by_label():
    return {lab: float(r) for lab, r in zip(POP_LABELS, PD14_TARGET_RATES)}


def meso_by_label(T=10.0, dt=1e-4, dt_rec=1e-3, seed=12345):
    params, N = build_params(1.0)
    params = build_step_stimulus(params, T, dt_rec, B=0.0)
    _, A = simulate(T, dt, dt_rec, params, 8, seed)
    n0 = int(0.5 / dt_rec)
    rates = A[:, n0:].mean(axis=1)
    return {lab: float(r) for lab, r in zip(POP_LABELS, rates)}


def micro_by_label():
    f = RESULTS / "benchmark_c_latest.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    for r in d['results']:
        if r.get('status') == 'ok' and r['backend'] == 'micro_c_sparse' \
                and abs(r['scaling'] - 1.0) < 1e-9:
            return {lab: float(v) for lab, v in
                    zip(r['pop_order'], r['mean_rate_hz'])}
    return None


def main():
    pub = published_by_label()
    meso = meso_by_label()
    micro = micro_by_label()
    if micro is None:
        print("WARNING: no full-scale sparse micro_c in benchmark_c_latest.json; "
              "run: python benchmark_c.py --backends micro_c --connectivity sparse "
              "--scalings 1.0 --T 2 (in WSL)")
        micro = {lab: np.nan for lab in ORDER}

    p = np.array([pub[l] for l in ORDER])
    m = np.array([meso[l] for l in ORDER])
    u = np.array([micro[l] for l in ORDER])

    # pairwise mean abs errors
    def mae(a, b):
        mask = ~np.isnan(a) & ~np.isnan(b)
        return float(np.abs(a[mask] - b[mask]).mean())
    print("=== three-way validation (full scale, N=77,169) ===")
    print(f"{'pop':6s} {'published':>10s} {'meso':>8s} {'micro':>8s}")
    for i, lab in enumerate(ORDER):
        print(f"{lab:6s} {p[i]:10.3f} {m[i]:8.3f} {u[i]:8.3f}")
    print(f"\nmean abs error   meso vs published : {mae(m, p):.3f} Hz")
    print(f"mean abs error   micro vs published: {mae(u, p):.3f} Hz")
    print(f"mean abs error   meso vs micro     : {mae(m, u):.3f} Hz")

    # figure: 3 bars per population
    x = np.arange(len(ORDER)); w = 0.27
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - w, p, w, label="published PD14 (orig. micro)", color="#888")
    ax.bar(x, m, w, label="our mesoscopic", color="#2c7fb8")
    ax.bar(x + w, u, w, label="our microscopic (sparse)", color="#d95f0e")
    ax.set_xticks(x); ax.set_xticklabels(ORDER)
    ax.set_ylabel("spontaneous firing rate [Hz]")
    ax.set_title("Three-way validation: published vs our meso vs our micro "
                 "(full scale, N=77,169)")
    ax.legend()
    fig.tight_layout()
    out = FIGURES / "three_way_validation.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    main()
