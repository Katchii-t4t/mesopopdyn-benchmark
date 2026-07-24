"""
test_meso_vs_micro.py -- rigorous per-population comparison of meso vs micro.

Runs BOTH models over independent seeds at full scale, then for each population
tests whether the mesoscopic mean rate differs significantly from the
microscopic (sparse, real-connectivity) mean rate.

  * z = (mean_meso - mean_micro) / sqrt(SE_meso^2 + SE_micro^2),  SE = std/sqrt(n)
  * |z| > 2  -> statistically significant difference
  * effect size = |mean_meso - mean_micro| / mean_micro  (relative)

The point: distinguish a real approximation bias (significant AND large effect,
e.g. L4i) from tiny differences that are statistically significant but
practically negligible.

Runs INSIDE WSL, 'nest' env (needs the built C library for micro).

Usage:
    python test_meso_vs_micro.py --meso_seeds 10 --micro_seeds 5 --T_micro 1

Karthik Senthil Ganesh, TU Berlin / BCCN internship, 2026.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
UPSTREAM_C = ROOT / "upstream_reference" / "mesopopdyngif_C_v18_Py3"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(UPSTREAM_C))

from model_setup import build_params, build_step_stimulus, POP_LABELS
from sim_meso import simulate as meso_sim
from benchmark_nest import build_nest_params, POP_LABELS_INTERLEAVED

ORDER = POP_LABELS_INTERLEAVED  # canonical plotting order


def meso_rates(seed, T=10.0, dt=1e-4, dt_rec=1e-3):
    p, _ = build_params(1.0)
    p = build_step_stimulus(p, T, dt_rec, B=0.0)
    _, A = meso_sim(T, dt, dt_rec, p, 8, seed)
    n0 = int(0.5 / dt_rec)
    r = A[:, n0:].mean(axis=1)                       # E-first order
    return {lab: float(x) for lab, x in zip(POP_LABELS, r)}


def micro_rates(mp, seed, T=1.0, dt=1e-4, dt_rec=1e-3):
    kw, K = build_nest_params(1.0, meanfield_connectivity=False)  # sparse
    K = len(kw['N'])
    step = np.zeros((K, 2)); tstep = np.array([[dt, T - dt]] * K)
    p = mp.MultiPop(dt=dt, **kw); p.dt_rec = dt_rec
    p.build_network_tilo_neurons()
    p.simulate(T, step=step, tstep=tstep, seed=seed, ForceSim=True)
    r = np.asarray(p.get_firingrates())              # interleaved order
    return {lab: float(x) for lab, x in zip(POP_LABELS_INTERLEAVED, r)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--meso_seeds', type=int, default=10)
    ap.add_argument('--micro_seeds', type=int, default=5)
    ap.add_argument('--T_micro', type=float, default=1.0)
    args = ap.parse_args()

    import multipop18 as mp   # real C wrapper

    print(f"meso: {args.meso_seeds} seeds ...")
    M = []
    for s in range(args.meso_seeds):
        d = meso_rates(1000 + s)                 # one sim per seed
        M.append([d[l] for l in ORDER])
    M = np.array(M)
    print(f"micro (sparse, full scale): {args.micro_seeds} seeds "
          f"(~expensive) ...")
    U = []
    for s in range(args.micro_seeds):
        d = micro_rates(mp, 200 + s, T=args.T_micro)  # one sim per seed
        U.append([d[l] for l in ORDER])
        print(f"  micro seed {s+1}/{args.micro_seeds} done")
    U = np.array(U)

    m_mean, m_std = M.mean(0), M.std(0, ddof=1)
    u_mean, u_std = U.mean(0), U.std(0, ddof=1)
    se = np.sqrt(m_std**2 / args.meso_seeds + u_std**2 / args.micro_seeds)
    z = (m_mean - u_mean) / np.where(se > 0, se, np.nan)
    eff = np.abs(m_mean - u_mean) / u_mean            # relative effect size

    print("\npop     meso(mean±std)   micro(mean±std)     z      effect  verdict")
    rows = []
    for i, lab in enumerate(ORDER):
        sig = abs(z[i]) > 2
        big = eff[i] > 0.03
        verdict = ("REAL BIAS" if sig and big else
                   "sig, tiny" if sig else "equivalent")
        print(f"{lab:5s} {m_mean[i]:6.3f}±{m_std[i]:.3f}   "
              f"{u_mean[i]:6.3f}±{u_std[i]:.3f}  {z[i]:6.1f}  "
              f"{eff[i]*100:5.1f}%  {verdict}")
        rows.append({'pop': lab, 'meso_mean': float(m_mean[i]),
                     'meso_std': float(m_std[i]), 'micro_mean': float(u_mean[i]),
                     'micro_std': float(u_std[i]), 'z': float(z[i]),
                     'effect_rel': float(eff[i]), 'verdict': verdict})

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "meso_vs_micro_test.json").write_text(json.dumps(
        {'meso_seeds': args.meso_seeds, 'micro_seeds': args.micro_seeds,
         'T_micro': args.T_micro, 'per_pop': rows}, indent=2))

    # figure: means with std error bars + significance stars
    x = np.arange(len(ORDER)); w = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - w/2, m_mean, w, yerr=m_std, capsize=3, label="mesoscopic",
           color="#2c7fb8")
    ax.bar(x + w/2, u_mean, w, yerr=u_std, capsize=3, label="microscopic (sparse)",
           color="#d95f0e")
    for i in range(len(ORDER)):
        if abs(z[i]) > 2 and eff[i] > 0.03:
            ax.text(x[i], max(m_mean[i], u_mean[i]) + 0.3, "*",
                    ha='center', fontsize=16, color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels(ORDER)
    ax.set_ylabel("firing rate [Hz]  (mean ± std over seeds)")
    ax.set_title("Meso vs micro, per population — * = significant real bias "
                 "(|z|>2 and >3%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "meso_vs_micro_test.png", dpi=130)
    plt.close(fig)
    print("\nwrote meso_vs_micro_test.png, meso_vs_micro_test.json")


if __name__ == "__main__":
    main()
