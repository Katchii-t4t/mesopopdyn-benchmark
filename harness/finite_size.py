"""
finite_size.py -- empirical test of the finite-size fluctuation scaling law.

Schwalger's mesoscopic theory models the per-step population activity as
    n ~ Binomial(N, Abar) / N,
so the standard deviation of the population activity A(t) around its mean should
scale as ~ 1/sqrt(N). This script measures that scaling directly: it runs the
meso simulation at several network sizes, computes the steady-state std of A(t)
per population, and fits std vs N on a log-log axis. A slope near -0.5 confirms
the 1/sqrt(N) finite-size noise law.

This is a genuine *measured* result (not a reproduction of published rates):
it validates the stochastic core of the mesoscopic reduction.

Usage:  python finite_size.py

Karthik Senthil Ganesh, TU Berlin / BCCN internship, 2026.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model_setup import build_params, build_step_stimulus, POP_LABELS
from sim_meso import simulate

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def measure(scalings, T=10.0, dt=1e-4, dt_rec=1e-3, seed=12345, t_settle=1.0):
    """Return dict: per scaling -> (N_per_pop, mean_rate, std_rate)."""
    out = {}
    for s in scalings:
        params, N = build_params(s)
        params = build_step_stimulus(params, T, dt_rec, B=0.0)
        rate, A = simulate(T, dt, dt_rec, params, 8, seed)
        n0 = int(t_settle / dt_rec)
        A_ss = A[:, n0:]
        out[s] = {
            'N': np.asarray(N, float),
            'mean': A_ss.mean(axis=1),
            'std': A_ss.std(axis=1),
        }
        print(f"  scaling={s:<5g} N0={int(N[0])} std(L23e)={out[s]['std'][0]:.4f}")
    return out


def main():
    scalings = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
    print("=== finite-size fluctuation scaling ===")
    data = measure(scalings)

    # For each population, fit log(std) vs log(N): slope ~ -0.5 expected.
    # std of A around mean ~ sqrt(Abar(1-Abar)/N)/dt_rec -> slope -0.5 in N.
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    slopes = {}
    report = {'scalings': scalings, 'per_pop': []}
    for k, lab in enumerate(POP_LABELS):
        N_k = np.array([data[s]['N'][k] for s in scalings])
        std_k = np.array([data[s]['std'][k] for s in scalings])
        good = (std_k > 0) & (N_k > 0)
        logN, logS = np.log10(N_k[good]), np.log10(std_k[good])
        slope, intercept = np.polyfit(logN, logS, 1)
        slopes[lab] = float(slope)
        report['per_pop'].append({'pop': lab, 'slope': float(slope),
                                  'N': [float(x) for x in N_k],
                                  'std': [float(x) for x in std_k]})
        ax.plot(N_k[good], std_k[good], 'o-', lw=1, ms=4,
                label=f"{lab} (slope {slope:.2f})")

    # reference 1/sqrt(N) line anchored to the first population's first point
    N_ref = np.array([data[s]['N'][0] for s in scalings])
    s0 = data[scalings[0]]['std'][0]
    ax.plot(N_ref, s0 * np.sqrt(N_ref[0] / N_ref), 'k--', lw=2,
            label=r"reference $\propto 1/\sqrt{N}$")

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel("population size N")
    ax.set_ylabel("std of population activity A(t) [Hz]")
    mean_slope = np.mean(list(slopes.values()))
    ax.set_title(f"Finite-size fluctuation scaling (mean slope "
                 f"{mean_slope:.2f}, theory -0.5)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    out = FIGURES / "finite_size_scaling.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)

    report['mean_slope'] = float(mean_slope)
    report['theory_slope'] = -0.5
    (RESULTS / "finite_size_latest.json").write_text(json.dumps(report, indent=2))

    print(f"\nmean fitted slope: {mean_slope:.3f}  (theory: -0.5)")
    print(f"wrote {out.name}, finite_size_latest.json")


if __name__ == "__main__":
    main()
