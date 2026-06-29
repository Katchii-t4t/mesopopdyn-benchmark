"""
validate_multiseed.py -- multi-seed validation of the mesoscopic rates.

Runs the meso simulation across many independent finite-size-noise seeds and
reports, per population, the mean +/- std of the time-averaged firing rate.
A single-seed validation (validate.py) can't tell you how much of the residual
error is noise; this does. Output: a bar chart with error bars vs the published
PD14 rates, plus a JSON summary.

Runs on Windows (Numba meso) -- no WSL needed.

Usage:  python validate_multiseed.py --scaling 1.0 --T 10 --seeds 10

Karthik Senthil Ganesh, TU Berlin / BCCN internship, 2026.
"""

import argparse
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scaling', type=float, default=1.0)
    ap.add_argument('--T', type=float, default=10.0)
    ap.add_argument('--dt', type=float, default=1e-4)
    ap.add_argument('--dt_rec', type=float, default=1e-3)
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--t_settle', type=float, default=0.5)
    args = ap.parse_args()

    n0 = int(args.t_settle / args.dt_rec)
    rates = []  # (seeds, M)
    for s in range(args.seeds):
        params, N = build_params(args.scaling)
        params = build_step_stimulus(params, args.T, args.dt_rec, B=0.0)
        _, A = simulate(args.T, args.dt, args.dt_rec, params, 8, 1000 + s)
        rates.append(A[:, n0:].mean(axis=1))
    rates = np.array(rates)
    mean = rates.mean(axis=0)
    std = rates.std(axis=0)
    target = PD14_TARGET_RATES

    print(f"=== multi-seed validation (scaling={args.scaling}, T={args.T}, "
          f"seeds={args.seeds}, N={int(N.sum())}) ===")
    print(f"{'pop':6s} {'mean':>8s} {'std':>7s} {'target':>8s} {'|err|':>7s}")
    report = {'scaling': args.scaling, 'T': args.T, 'seeds': args.seeds,
              'N_total': int(N.sum()), 'per_pop': []}
    for i, lab in enumerate(POP_LABELS):
        err = abs(mean[i] - target[i])
        print(f"{lab:6s} {mean[i]:8.3f} {std[i]:7.3f} {target[i]:8.3f} {err:7.3f}")
        report['per_pop'].append({
            'pop': lab, 'mean_hz': float(mean[i]), 'std_hz': float(std[i]),
            'target_hz': float(target[i]), 'abs_err_hz': float(err),
            'within_1sigma': bool(err <= std[i] + 1e-9)})
    n_within = sum(p['within_1sigma'] for p in report['per_pop'])
    report['mean_abs_err_hz'] = float(np.abs(mean - target).mean())
    report['n_within_1sigma'] = n_within
    print(f"\nmean abs error {report['mean_abs_err_hz']:.4f} Hz; "
          f"{n_within}/8 populations within 1 std of published")

    # figure
    FIGURES.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
    x = np.arange(len(POP_LABELS)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(x - w/2, target, w, label="PD14 published", color="#888")
    ax.bar(x + w/2, mean, w, yerr=std, capsize=4, label=f"meso (mean±std, {args.seeds} seeds)",
           color="#2c7fb8")
    ax.set_xticks(x); ax.set_xticklabels(POP_LABELS)
    ax.set_ylabel("spontaneous rate [Hz]")
    ax.set_title(f"Multi-seed mesoscopic rates vs published "
                 f"(N={int(N.sum())}, {args.seeds} seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "validation_multiseed.png", dpi=130)
    plt.close(fig)
    (RESULTS / "validation_multiseed_latest.json").write_text(json.dumps(report, indent=2))
    print("wrote validation_multiseed.png, validation_multiseed_latest.json")


if __name__ == "__main__":
    main()
