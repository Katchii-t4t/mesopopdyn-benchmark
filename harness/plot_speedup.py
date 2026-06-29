"""
plot_speedup.py -- the headline figure: mesoscopic vs the real (sparse) micro.

Cost per biological second vs network size, log-log, for the mesoscopic backend
and the sparse (real PD14 quenched-random) microscopic backend. The mesoscopic
cost is ~flat in N; the sparse micro grows steeply, so the speedup grows with N
and reaches ~100-160x at full scale (N=77,169) -- a *measured* counterpart to
the ~150x of Schwalger et al. (2017), against the real network (not all-to-all).

Usage:  python plot_speedup.py

Karthik Senthil Ganesh, TU Berlin / BCCN internship, 2026.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_backends import load_all

FIGURES = Path(__file__).resolve().parent.parent / "figures"


def series(rows, backend):
    pts = sorted([r for r in rows if r['backend'] == backend],
                 key=lambda r: r['N_total'])
    N = np.array([r['N_total'] for r in pts], float)
    cost = np.array([r['time_min_s'] / r['T'] for r in pts])
    return N, cost


def main():
    rows = load_all()
    Nm, cm = series(rows, 'meso_c')           # representative meso (compiled)
    Ns, cs = series(rows, 'micro_c_sparse')   # real sparse micro

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(Nm, cm, 'o-', color='#d95f0e', lw=2, ms=7, label='mesoscopic (C)')
    ax.plot(Ns, cs, 's-', color='#252525', lw=2, ms=7,
            label='microscopic, sparse (C)')
    ax.axhline(1.0, color='grey', ls=':', lw=1, label='realtime')

    # annotate speedup where both exist
    common = sorted(set(np.round(Nm).astype(int)) & set(np.round(Ns).astype(int)))
    for N in common:
        cmi = cm[np.argmin(np.abs(Nm - N))]
        csi = cs[np.argmin(np.abs(Ns - N))]
        sp = csi / cmi
        ax.annotate(f"{sp:.0f}× faster", xy=(N, np.sqrt(cmi * csi)),
                    ha='center', fontsize=10, fontweight='bold', color='#d95f0e')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel("network size (# neurons)")
    ax.set_ylabel("wall-clock seconds per biological second")
    ax.set_title("Mesoscopic vs real (sparse) microscopic PD14\n"
                 "speedup grows with N, ~130× at full scale")
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    out = FIGURES / "speedup_vs_sparse_micro.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out.name}")
    for N in common:
        cmi = cm[np.argmin(np.abs(Nm - N))]
        csi = cs[np.argmin(np.abs(Ns - N))]
        print(f"  N={N:6d}: meso {cmi:.2f} vs sparse micro {csi:.1f} "
              f"s/bio-s  ->  {csi/cmi:.0f}x")


if __name__ == "__main__":
    main()
