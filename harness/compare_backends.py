"""
compare_backends.py -- merge and compare results across backends.

Reads:
  results/benchmark_latest.json       (meso_numba, meso_python)
  results/benchmark_nest_latest.json  (meso_nest, micro_nest)

Produces:
  results/cross_backend_summary.json  (merged table)
  figures/cross_backend.png           (runtime + rate-agreement comparison)
  prints a comparison table to stdout.

The two scientific points this makes:
  1. Cross-implementation agreement: meso_numba vs meso_nest should give the
     same full-scale rates (validates both implementations against each other).
  2. micro vs meso speedup: at the same network size, the mesoscopic backend
     should be far faster than the microscopic one while giving the same rates
     (reproducing the ~150x of Schwalger et al. 2017 with our own numbers).

Karthik Senthil Ganesh, TU Berlin / BCCN internship, 2026.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def load(name):
    p = RESULTS / name
    if not p.exists():
        return []
    return json.loads(p.read_text()).get('results', [])


def main():
    rows = []
    for f in ("benchmark_latest.json", "benchmark_nest_latest.json"):
        for r in load(f):
            if r.get('status') == 'ok':
                rows.append(r)

    if not rows:
        print("no results found")
        return

    # ---- table ----
    print(f"{'backend':12s} {'scaling':>7s} {'N':>7s} {'min[s]':>8s} "
          f"{'rt':>6s} {'rate_err[Hz]':>12s}")
    for r in sorted(rows, key=lambda r: (r['backend'], r['scaling'])):
        print(f"{r['backend']:12s} {r['scaling']:7g} {r['N_total']:7d} "
              f"{r['time_min_s']:8.2f} {r['realtime_factor']:6.2f} "
              f"{r['mean_abs_rate_err_hz']:12.3f}")

    # ---- cross-implementation agreement at full scale (meso) ----
    def full(backend):
        cand = [r for r in rows if r['backend'] == backend and
                abs(r['scaling'] - 1.0) < 1e-9]
        return cand[0] if cand else None

    summary = {'rows': rows, 'comparisons': {}}
    py = full('meso_numba')
    ne = full('meso_nest')
    if py and ne:
        # rates are in different population orders; compare by sorted magnitude
        rp = np.sort(np.array(py['mean_rate_hz']))
        rn = np.sort(np.array(ne['mean_rate_hz']))
        agree = float(np.abs(rp - rn).mean())
        summary['comparisons']['meso_py_vs_nest_full'] = {
            'mean_abs_rate_diff_hz': agree,
            'meso_numba_min_s': py['time_min_s'],
            'meso_nest_min_s': ne['time_min_s'],
        }
        print(f"\nmeso_numba vs meso_nest @ full scale: "
              f"rate diff {agree:.3f} Hz (sorted), "
              f"time {py['time_min_s']:.2f}s vs {ne['time_min_s']:.2f}s")

    # ---- micro vs meso speedup at the same (reduced) scale ----
    micro = [r for r in rows if r['backend'] == 'micro_nest']
    for m in micro:
        s = m['scaling']
        meso_same = [r for r in rows if r['backend'] in ('meso_nest', 'meso_numba')
                     and abs(r['scaling'] - s) < 1e-9]
        for ms in meso_same:
            # normalise by T (micro may use a shorter T)
            micro_per_s = m['time_min_s'] / m['T']
            meso_per_s = ms['time_min_s'] / ms['T']
            speedup = micro_per_s / meso_per_s
            key = f"micro_vs_{ms['backend']}_scaling{s}"
            summary['comparisons'][key] = {
                'scaling': s, 'N_total': m['N_total'],
                'micro_s_per_biosec': micro_per_s,
                'meso_s_per_biosec': meso_per_s,
                'speedup': speedup,
            }
            print(f"micro_nest vs {ms['backend']} @ scaling {s} "
                  f"(N={m['N_total']}): meso is {speedup:.1f}x faster "
                  f"per bio-second")

    (RESULTS / "cross_backend_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- figure: runtime per bio-second vs N, all backends ----
    FIGURES.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    backends = sorted(set(r['backend'] for r in rows))
    colors = {'meso_numba': '#2c7fb8', 'meso_nest': '#31a354',
              'meso_python': '#999999', 'micro_nest': '#d95f0e'}
    for b in backends:
        pts = sorted([r for r in rows if r['backend'] == b],
                     key=lambda r: r['N_total'])
        N = [r['N_total'] for r in pts]
        tps = [r['time_min_s'] / r['T'] for r in pts]
        ax.plot(N, tps, 'o-', label=b, color=colors.get(b))
    ax.axhline(1.0, color='grey', ls=':', label='realtime')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel("network size (# neurons)")
    ax.set_ylabel("wall-clock seconds per biological second")
    ax.set_title("Cross-backend cost: micro vs meso (lower = faster)")
    ax.legend()
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    out = FIGURES / "cross_backend.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nwrote {out.name}, cross_backend_summary.json")


if __name__ == "__main__":
    main()
