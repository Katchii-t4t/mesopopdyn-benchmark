"""
plot.py -- figure generation for the mesoscopic PD14 harness.

Produces three figures into ../figures/:
  1. validation_rates.png  -- simulated vs published per-population rates.
  2. benchmark_scaling.png -- runtime & realtime-factor vs network size.
  3. activity_traces.png   -- example population-activity traces (8 panels).

Reads results/validation_latest.json and results/benchmark_latest.json when
present; (re)generates an activity trace on the fly.

Usage:  python plot.py

Karthik Senthil Ganesh, TU Berlin / BCCN internship, 2026.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

from model_setup import (
    build_params, build_step_stimulus, PD14_TARGET_RATES, POP_LABELS,
)
from sim_meso import simulate

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def plot_validation():
    f = RESULTS / "validation_latest.json"
    if not f.exists():
        print("[plot] no validation_latest.json; skipping validation figure")
        return
    rep = json.loads(f.read_text())
    pops = [p['pop'] for p in rep['per_pop']]
    sim = [p['sim_rate_hz'] for p in rep['per_pop']]
    tgt = [p['target_rate_hz'] for p in rep['per_pop']]

    x = np.arange(len(pops))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w/2, tgt, w, label="PD14 published", color="#888")
    ax.bar(x + w/2, sim, w, label="meso sim", color="#2c7fb8")
    ax.set_xticks(x)
    ax.set_xticklabels(pops)
    ax.set_ylabel("spontaneous rate [Hz]")
    ax.set_title(f"Mesoscopic PD14 vs published rates "
                 f"(N={rep['N_total']}, mean abs err "
                 f"{rep['mean_abs_err_hz']:.3f} Hz)")
    ax.legend()
    fig.tight_layout()
    out = FIGURES / "validation_rates.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[plot] wrote {out.name}")


def plot_benchmark():
    f = RESULTS / "benchmark_latest.json"
    if not f.exists():
        print("[plot] no benchmark_latest.json; skipping benchmark figure")
        return
    data = json.loads(f.read_text())
    rows = [r for r in data['results']
            if r.get('status') == 'ok']
    if not rows:
        print("[plot] no ok benchmark rows; skipping benchmark figure")
        return
    rows.sort(key=lambda r: r['N_total'])
    N = [r['N_total'] for r in rows]
    tmin = [r['time_min_s'] for r in rows]
    tmean = [r['time_mean_s'] for r in rows]
    tstd = [r['time_std_s'] for r in rows]
    rtf = [r['realtime_factor'] for r in rows]
    T = rows[0]['T']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    ax1.errorbar(N, tmean, yerr=tstd, fmt='o-', color="#2c7fb8",
                 capsize=3, label="mean ± std")
    ax1.plot(N, tmin, 's--', color="#d95f0e", label="min")
    ax1.set_xlabel("network size (# neurons)")
    ax1.set_ylabel(f"wall-clock time [s] for {T:g}s bio")
    ax1.set_title("Mesoscopic runtime vs network size")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(N, rtf, 'o-', color="#31a354")
    ax2.axhline(1.0, color='grey', ls=':', label="realtime")
    ax2.set_xlabel("network size (# neurons)")
    ax2.set_ylabel("realtime factor (T / wall)")
    ax2.set_title("Speed relative to biological realtime")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = FIGURES / "benchmark_scaling.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[plot] wrote {out.name}")


def plot_activity_traces(scaling=1.0, T=2.0, dt=0.0001, dt_rec=0.001,
                         seed=12345, t_settle=0.2):
    params, N = build_params(scaling)
    params = build_step_stimulus(params, T, dt_rec, B=0.0)
    rate, A = simulate(T, dt, dt_rec, params, 8, seed)
    # drop the initial synchronous-firing transient (all units fire at t=0)
    n0 = int(t_settle / dt_rec)
    A = A[:, n0:]
    rate = rate[:, n0:]
    t = t_settle + dt_rec * np.arange(A.shape[1])

    fig, axes = plt.subplots(4, 2, figsize=(10, 8), sharex=True)
    # left column excitatory (0..3), right column inhibitory (4..7)
    order = [(0, 0, 0), (1, 0, 1), (2, 0, 2), (3, 0, 3),
             (0, 1, 4), (1, 1, 5), (2, 1, 6), (3, 1, 7)]
    for row, col, k in order:
        ax = axes[row, col]
        ax.plot(t, A[k], color="#666", lw=0.6, label=r"$A_N(t)$")
        ax.plot(t, rate[k], color="#2c7fb8", lw=1.2, label=r"$\bar A(t)$")
        ax.set_ylabel(POP_LABELS[k] + "\n[Hz]")
        ax.set_ylim(bottom=0)
        if row == 0:
            ax.legend(fontsize=8, loc='upper right')
    axes[3, 0].set_xlabel("time [s]")
    axes[3, 1].set_xlabel("time [s]")
    fig.suptitle(f"Mesoscopic PD14 population activity "
                 f"(N={int(N.sum())}, scaling={scaling})")
    fig.tight_layout()
    out = FIGURES / "activity_traces.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[plot] wrote {out.name}")


def main():
    FIGURES.mkdir(exist_ok=True)
    plot_validation()
    plot_benchmark()
    plot_activity_traces()


if __name__ == "__main__":
    main()
