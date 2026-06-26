"""
validate.py -- correctness validation for the mesoscopic PD14 simulation.

The original Schwalger repository validates only by eye (matplotlib traces).
This module implements quantitative, van-Albada-style correctness metrics
(van Albada et al. 2018, the de-facto PD14 benchmarking standard) adapted to
the mesoscopic / population-rate setting:

  * Per-population time-averaged spontaneous firing rate vs. the published
    PD14 full-scale rates (Potjans & Diesmann 2014, Table-derived).
  * Absolute and relative error per population.
  * Coefficient of variation of the population-activity signal (a proxy for
    finite-size fluctuation magnitude; true ISI-CV needs the micro sim).
  * An overall PASS/FAIL verdict against a configurable tolerance.

For the mesoscopic model the meaningful correctness target is the *mean rate*
per population at full network size; finite-size fluctuations are expected and
are themselves the object of Schwalger's theory, so they are reported but not
used for pass/fail.

Usage:
    python validate.py --scaling 1.0 --T 10
    python validate.py --scaling 1.0 --T 10 --tol 0.15

Karthik Senthil Ganesh, TU Berlin / BCCN internship, 2026.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from model_setup import (
    build_params, build_step_stimulus, PD14_TARGET_RATES, POP_LABELS,
)
from sim_meso import simulate

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_and_measure(scaling, T, dt, dt_rec, seed, t_settle=0.5):
    """Run the meso sim and compute per-population statistics."""
    params, N = build_params(scaling)
    params = build_step_stimulus(params, T, dt_rec, B=0.0)
    rate, A = simulate(T, dt, dt_rec, params, 8, seed)

    n0 = min(int(t_settle / dt_rec), A.shape[1] // 2)
    A_ss = A[:, n0:]               # steady-state slice of population activity
    mean_rate = A_ss.mean(axis=1)
    std_rate = A_ss.std(axis=1)
    # CV of the population-activity signal (fluctuation proxy)
    cv = np.divide(std_rate, mean_rate,
                   out=np.zeros_like(std_rate), where=mean_rate > 0)
    return mean_rate, std_rate, cv, N


def validate(scaling, T, dt, dt_rec, seed, tol):
    mean_rate, std_rate, cv, N = run_and_measure(scaling, T, dt, dt_rec, seed)
    target = PD14_TARGET_RATES

    abs_err = np.abs(mean_rate - target)
    rel_err = np.divide(abs_err, target,
                        out=np.full_like(abs_err, np.nan), where=target > 0)

    # Pass/fail is only meaningful at (near) full scale, where down-scaling
    # distortion is negligible.
    per_pop_pass = rel_err <= tol
    overall_pass = bool(np.all(per_pop_pass)) and scaling >= 0.99

    report = {
        'timestamp': datetime.now().isoformat(),
        'scaling': scaling, 'T': T, 'dt': dt, 'dt_rec': dt_rec,
        'seed': seed, 'tolerance_rel': tol,
        'N_total': int(N.sum()),
        'overall_pass': overall_pass,
        'overall_pass_note': (
            "" if scaling >= 0.99 else
            "scaling<1: rate comparison not authoritative (down-scaling "
            "distorts activity; cf. van Albada et al. 2015)."),
        'mean_abs_err_hz': float(abs_err.mean()),
        'per_pop': [],
    }
    for i, lab in enumerate(POP_LABELS):
        report['per_pop'].append({
            'pop': lab,
            'sim_rate_hz': float(mean_rate[i]),
            'target_rate_hz': float(target[i]),
            'abs_err_hz': float(abs_err[i]),
            'rel_err': None if np.isnan(rel_err[i]) else float(rel_err[i]),
            'activity_cv': float(cv[i]),
            'pass': bool(per_pop_pass[i]),
        })
    return report


def print_report(report):
    print(f"\n=== Validation report (scaling={report['scaling']}, "
          f"T={report['T']}s, N={report['N_total']}) ===")
    print(f"{'pop':6s} {'sim[Hz]':>8s} {'target':>8s} {'abs_err':>8s} "
          f"{'rel_err':>8s} {'cv':>6s}  pass")
    for p in report['per_pop']:
        rel = "  n/a " if p['rel_err'] is None else f"{p['rel_err']*100:6.1f}%"
        print(f"{p['pop']:6s} {p['sim_rate_hz']:8.3f} "
              f"{p['target_rate_hz']:8.3f} {p['abs_err_hz']:8.3f} "
              f"{rel} {p['activity_cv']:6.3f}  {'OK' if p['pass'] else 'x'}")
    print(f"\nmean abs error: {report['mean_abs_err_hz']:.4f} Hz")
    verdict = "PASS" if report['overall_pass'] else "FAIL"
    print(f"OVERALL: {verdict}")
    if report['overall_pass_note']:
        print(f"note: {report['overall_pass_note']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scaling', type=float, default=1.0)
    ap.add_argument('--T', type=float, default=10.0)
    ap.add_argument('--dt', type=float, default=0.0001)
    ap.add_argument('--dt_rec', type=float, default=0.001)
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--tol', type=float, default=0.10,
                    help="relative-error tolerance for per-pop pass/fail")
    args = ap.parse_args()

    report = validate(args.scaling, args.T, args.dt, args.dt_rec,
                      args.seed, args.tol)
    print_report(report)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "validation_latest.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nsaved: {out.name}")


if __name__ == "__main__":
    main()
