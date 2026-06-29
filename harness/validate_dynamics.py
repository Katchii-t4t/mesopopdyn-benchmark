"""
validate_dynamics.py -- dynamics-level validation: does meso reproduce micro
beyond the mean rate?

Mean firing rate is necessary but not sufficient. This compares the temporal
structure of the two models via:
  * power spectral density (PSD) of each population's activity A(t)
    -- captures oscillations / fluctuation structure (van-Albada style),
  * coefficient of variation (CV) of the inter-spike interval, for the
    microscopic network (a single-neuron variability measure).

Uses Schwalger's MultiPop machinery (get_psd_* / get_isih_* in the C library),
so it must run INSIDE WSL in the 'nest' env with the C library built.

Compares mesoscopic (population equations, C) vs microscopic (sparse, real
PD14 connectivity, C) at a reduced scale where many trials are affordable.

Usage (in WSL, nest env):
    python validate_dynamics.py --scaling 0.2 --Ntrials 20

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

from benchmark_nest import build_nest_params, POP_LABELS_INTERLEAVED


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scaling', type=float, default=0.2)
    ap.add_argument('--Ntrials', type=int, default=20)
    ap.add_argument('--df', type=float, default=1.0)        # Hz freq resolution
    ap.add_argument('--dt', type=float, default=1e-4)
    ap.add_argument('--dt_rec', type=float, default=1e-3)
    args = ap.parse_args()

    import multipop18 as mp

    def make(meanfield):
        kw, N = build_nest_params(args.scaling, meanfield_connectivity=meanfield)
        p = mp.MultiPop(dt=args.dt, **kw)
        p.dt_rec = args.dt_rec
        return p, N

    # --- mesoscopic PSD ---
    print("computing mesoscopic PSD ...")
    pm, N = make(True)
    pm.build_network_tilo_populations()
    pm.get_psd(df=args.df, dt_sample=args.dt_rec, Ntrials=args.Ntrials)
    f_meso, psd_meso = pm.freq, pm.psd          # psd: (Nfreq, K)

    # --- microscopic (sparse) PSD + ISI CV ---
    print("computing microscopic (sparse) PSD + ISI CV ...")
    pn, _ = make(False)
    pn.build_network_tilo_neurons()
    pn.get_psd(df=args.df, dt_sample=args.dt_rec, Ntrials=args.Ntrials)
    f_micro, psd_micro = pn.freq, pn.psd
    pn.get_isistat(tmax=2.0, Nbin=200, Nspikes=20000)
    cv_micro = pn.cv

    # --- figure: PSD overlay, 8 panels ---
    fig, axes = plt.subplots(4, 2, figsize=(11, 10), sharex=True)
    order = [(0,0,0),(1,0,2),(2,0,4),(3,0,6),(0,1,1),(1,1,3),(2,1,5),(3,1,7)]
    for row, col, k in order:
        ax = axes[row, col]
        ax.loglog(f_micro, psd_micro[:, k], color="#252525", lw=1.0, label="micro (sparse)")
        ax.loglog(f_meso, psd_meso[:, k], color="#d95f0e", lw=1.2, label="meso")
        ax.set_ylabel(POP_LABELS_INTERLEAVED[k])
        if row == 0 and col == 0:
            ax.legend(fontsize=8)
    axes[3, 0].set_xlabel("frequency [Hz]")
    axes[3, 1].set_xlabel("frequency [Hz]")
    fig.suptitle(f"Power spectra: meso vs sparse micro (scaling={args.scaling}, "
                 f"N={int(N.sum())}, {args.Ntrials} trials)")
    fig.tight_layout()
    fig.savefig(FIGURES / "dynamics_psd.png", dpi=120)
    plt.close(fig)

    # --- quantitative PSD agreement (log-space RMS over freq) ---
    psd_err = []
    for k in range(8):
        a = np.log10(np.maximum(psd_meso[:, k], 1e-12))
        b = np.log10(np.maximum(psd_micro[:, k], 1e-12))
        m = min(len(a), len(b))
        psd_err.append(float(np.sqrt(np.mean((a[:m] - b[:m]) ** 2))))

    report = {
        'scaling': args.scaling, 'N_total': int(N.sum()),
        'Ntrials': args.Ntrials, 'pop_order': POP_LABELS_INTERLEAVED,
        'cv_isi_micro': [float(x) for x in np.asarray(cv_micro)],
        'psd_log_rms_meso_vs_micro': psd_err,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "validation_dynamics_latest.json").write_text(json.dumps(report, indent=2))

    print("\npop     CV-ISI(micro)   PSD log-RMS(meso vs micro)")
    for k, lab in enumerate(POP_LABELS_INTERLEAVED):
        print(f"{lab:6s} {float(np.asarray(cv_micro)[k]):10.3f}      {psd_err[k]:8.3f}")
    print("\nwrote dynamics_psd.png, validation_dynamics_latest.json")


if __name__ == "__main__":
    main()
