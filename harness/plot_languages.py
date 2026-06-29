"""
plot_languages.py -- compare the four mesoscopic implementations side by side.

Two panels at full network scale (scaling = 1.0):
  (left)  speed: wall-clock seconds per biological second (log scale)
  (right) correctness: mean absolute rate error vs published PD14 rates

The story in one figure: all four backends agree on the answer (~0.05-0.07 Hz
error), but the cost differs by ~70x between the optimised (Numba/C/NEST) and
the unoptimised upstream Julia implementation.

Usage:  python plot_languages.py

Karthik Senthil Ganesh, TU Berlin / BCCN internship, 2026.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_backends import load_all

FIGURES = Path(__file__).resolve().parent.parent / "figures"

LABELS = {
    'meso_numba': 'Python/Numba',
    'meso_c': 'C (Schwalger)',
    'meso_nest': 'NEST (C++)',
    'meso_julia': 'Julia',
}
ORDER = ['meso_numba', 'meso_c', 'meso_nest', 'meso_julia']
COLORS = {'meso_numba': '#2c7fb8', 'meso_c': '#d95f0e',
          'meso_nest': '#31a354', 'meso_julia': '#756bb1'}


def main():
    rows = load_all()
    # full-scale meso row per backend
    full = {}
    for r in rows:
        if r['backend'] in LABELS and abs(r['scaling'] - 1.0) < 1e-9:
            full[r['backend']] = r

    backs = [b for b in ORDER if b in full]
    names = [LABELS[b] for b in backs]
    cost = [full[b]['time_min_s'] / full[b]['T'] for b in backs]   # s / bio-s
    err = [full[b]['mean_abs_rate_err_hz'] for b in backs]
    cols = [COLORS[b] for b in backs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    # --- speed (log) ---
    bars1 = ax1.bar(names, cost, color=cols)
    ax1.set_yscale('log')
    ax1.axhline(1.0, color='grey', ls=':', lw=1, label='realtime')
    ax1.set_ylabel("wall-clock seconds per biological second")
    ax1.set_title("Speed (lower = faster)")
    ax1.legend()
    for b, c in zip(bars1, cost):
        ax1.text(b.get_x() + b.get_width()/2, c, f"{c:.2g}",
                 ha='center', va='bottom', fontsize=9)
    ax1.tick_params(axis='x', rotation=15)

    # --- correctness ---
    bars2 = ax2.bar(names, err, color=cols)
    ax2.set_ylabel("mean abs rate error vs published [Hz]")
    ax2.set_title("Correctness (all agree, lower = better)")
    for b, e in zip(bars2, err):
        ax2.text(b.get_x() + b.get_width()/2, e, f"{e:.3f}",
                 ha='center', va='bottom', fontsize=9)
    ax2.tick_params(axis='x', rotation=15)
    ax2.set_ylim(0, max(err) * 1.3)

    fig.suptitle("Mesoscopic PD14: four implementations, full scale "
                 "(N=77,169) — same answer, very different speed")
    fig.tight_layout()
    out = FIGURES / "language_comparison.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out.name}")
    print("\nfull-scale meso summary:")
    for b in backs:
        print(f"  {LABELS[b]:14s} {full[b]['time_min_s']/full[b]['T']:7.2f} s/bio-s"
              f"   err {full[b]['mean_abs_rate_err_hz']:.3f} Hz")


if __name__ == "__main__":
    main()
