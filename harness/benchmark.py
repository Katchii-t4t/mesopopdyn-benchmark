"""
benchmark.py -- unified benchmarking harness for the mesoscopic
Potjans-Diesmann (PD14) simulation.

Goals (addressing weaknesses of the original Schwalger repo):
  * Single entry point instead of four separate per-language scripts.
  * Correct timing: Numba JIT compilation is excluded via an explicit warmup
    run (the original code timed the FIRST call and so reported compile time
    as run time).
  * Multiple repeats with mean / std / min statistics.
  * Structured, machine-readable output (JSON + CSV) instead of a manually
    maintained .ods spreadsheet.
  * A pluggable backend registry so the C / NEST / Julia backends can be
    slotted in later behind the same interface.

Runnable on this machine: the 'meso_numba' and 'meso_python' backends
(pure NumPy/Numba, no external simulator). The 'meso_c', 'meso_nest',
'meso_julia' backends are registered as NOT-AVAILABLE stubs documenting how
to wire them in once a NEST/gcc/Julia environment exists.

Usage:
    python benchmark.py --scalings 0.1 0.5 1.0 --T 10 --repeats 3
    python benchmark.py --quick          # fast smoke configuration

Karthik Senthil Ganesh, TU Berlin / BCCN internship, 2026.
"""

import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from model_setup import (
    build_params, build_step_stimulus, PD14_TARGET_RATES, POP_LABELS,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# --------------------------------------------------------------------------
# Backend registry
# --------------------------------------------------------------------------
# Each backend is a callable(T, dt, dt_rec, params, Nrecord, seed) -> (rate, A)
# plus an 'available' flag and a 'needs_warmup' flag.

class Backend:
    def __init__(self, name, run_fn, available, needs_warmup=False, note=""):
        self.name = name
        self.run_fn = run_fn
        self.available = available
        self.needs_warmup = needs_warmup
        self.note = note


def _make_backends():
    backends = {}

    # --- meso, Numba-accelerated (runnable) ---
    try:
        from sim_meso import simulate as _sim

        def run_numba(T, dt, dt_rec, params, Nrecord, seed):
            return _sim(T, dt, dt_rec, params, Nrecord, seed, with_numba=True)

        def run_python(T, dt, dt_rec, params, Nrecord, seed):
            return _sim(T, dt, dt_rec, params, Nrecord, seed, with_numba=False)

        backends['meso_numba'] = Backend(
            'meso_numba', run_numba, available=True, needs_warmup=True,
            note="Numba-JIT mesoscopic sim; warmup excludes compile time.")
        backends['meso_python'] = Backend(
            'meso_python', run_python, available=True, needs_warmup=False,
            note="Pure-Python (numba.njit.py_func) mesoscopic sim; very slow.")
    except Exception as exc:  # pragma: no cover
        backends['meso_numba'] = Backend(
            'meso_numba', None, available=False, note=f"import failed: {exc}")

    # --- stubs for the heavy backends (need external toolchains) ---
    backends['meso_c'] = Backend(
        'meso_c', None, available=False,
        note="Custom C library (glm_popdyn). Needs gcc + GSL + FFTW; "
             "build via upstream_reference/mesopopdyngif_C_v18_Py3/Makefile.")
    backends['meso_nest'] = Backend(
        'meso_nest', None, available=False,
        note="NEST gif_pop_psc_exp. Needs a NEST install (WSL/conda/Docker; "
             "NEST has no native Windows build).")
    backends['meso_julia'] = Backend(
        'meso_julia', None, available=False,
        note="Julia sim5.jl. Needs a Julia install + PyPlot/MAT.")
    return backends


# --------------------------------------------------------------------------
# Core timing routine
# --------------------------------------------------------------------------

def time_backend(backend, scaling, T, dt, dt_rec, repeats, seed=12345):
    """Run one backend at one scaling factor; return a result record."""
    params, N = build_params(scaling)
    params = build_step_stimulus(params, T, dt_rec, B=0.0)

    record = {
        'backend': backend.name,
        'scaling': scaling,
        'N_total': int(N.sum()),
        'N_per_pop': [int(x) for x in N],
        'T': T, 'dt': dt, 'dt_rec': dt_rec,
        'repeats': repeats,
        'available': backend.available,
    }

    if not backend.available:
        record['status'] = 'skipped'
        record['note'] = backend.note
        return record

    # warmup (compile / caches) -- NOT timed
    warmup_time = None
    if backend.needs_warmup:
        wp, _ = build_params(scaling)
        wp = build_step_stimulus(wp, min(0.05, T), dt_rec, B=0.0)
        t0 = time.perf_counter()
        backend.run_fn(min(0.05, T), dt, dt_rec, wp, 8, 1)
        warmup_time = time.perf_counter() - t0

    times = []
    rate = A = None
    for _ in range(repeats):
        # fresh params each repeat (sim mutates internal state, not params,
        # but rebuild is cheap and keeps repeats independent)
        p, _ = build_params(scaling)
        p = build_step_stimulus(p, T, dt_rec, B=0.0)
        t0 = time.perf_counter()
        rate, A = backend.run_fn(T, dt, dt_rec, p, 8, seed)
        times.append(time.perf_counter() - t0)

    times = np.array(times)
    # correctness cross-check: mean firing rate over t>0.5s (or t>T/2)
    n0 = min(int(0.5 / dt_rec), A.shape[1] // 2)
    mean_rate = A[:, n0:].mean(axis=1)
    rate_err = float(np.abs(mean_rate - PD14_TARGET_RATES).mean())

    record.update({
        'status': 'ok',
        'warmup_time_s': warmup_time,
        'time_mean_s': float(times.mean()),
        'time_std_s': float(times.std()),
        'time_min_s': float(times.min()),
        'times_s': [float(x) for x in times],
        'realtime_factor': float(T / times.min()),  # >1 = faster than realtime
        'mean_rate_hz': [float(x) for x in mean_rate],
        'mean_abs_rate_err_hz': rate_err,
    })
    return record


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def run_suite(scalings, T, dt, dt_rec, repeats, backends_to_run):
    all_backends = _make_backends()
    results = []
    for bname in backends_to_run:
        backend = all_backends.get(bname)
        if backend is None:
            print(f"[skip] unknown backend '{bname}'")
            continue
        for scaling in scalings:
            print(f"[run ] {bname:12s} scaling={scaling:<4g} T={T}s ... ",
                  end="", flush=True)
            rec = time_backend(backend, scaling, T, dt, dt_rec, repeats)
            if rec['status'] == 'ok':
                print(f"min={rec['time_min_s']:.3f}s "
                      f"(rt x{rec['realtime_factor']:.2f}) "
                      f"rate_err={rec['mean_abs_rate_err_hz']:.3f}Hz")
            else:
                print(rec['status'])
            results.append(rec)
    return results


def save_results(results, meta):
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {'meta': meta, 'results': results}

    json_path = RESULTS_DIR / f"benchmark_{stamp}.json"
    json_path.write_text(json.dumps(payload, indent=2))

    # flat CSV of the 'ok' rows
    csv_path = RESULTS_DIR / f"benchmark_{stamp}.csv"
    cols = ['backend', 'scaling', 'N_total', 'T', 'repeats',
            'warmup_time_s', 'time_mean_s', 'time_std_s', 'time_min_s',
            'realtime_factor', 'mean_abs_rate_err_hz', 'status']
    lines = [",".join(cols)]
    for r in results:
        lines.append(",".join(str(r.get(c, "")) for c in cols))
    csv_path.write_text("\n".join(lines))

    # stable "latest" copies for downstream plotting/docs
    (RESULTS_DIR / "benchmark_latest.json").write_text(json.dumps(payload, indent=2))
    (RESULTS_DIR / "benchmark_latest.csv").write_text("\n".join(lines))

    print(f"\nsaved: {json_path.name}, {csv_path.name} (+ *_latest.*)")
    return json_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scalings', type=float, nargs='+',
                    default=[0.1, 0.5, 1.0])
    ap.add_argument('--T', type=float, default=10.0,
                    help="biological simulation time (s)")
    ap.add_argument('--dt', type=float, default=0.0001)
    ap.add_argument('--dt_rec', type=float, default=0.001)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--backends', nargs='+',
                    default=['meso_numba'])
    ap.add_argument('--quick', action='store_true',
                    help="fast config: scalings 0.1 1.0, T=1s, repeats 2")
    args = ap.parse_args()

    if args.quick:
        args.scalings = [0.1, 1.0]
        args.T = 1.0
        args.repeats = 2

    meta = {
        'timestamp': datetime.now().isoformat(),
        'platform': platform.platform(),
        'python': platform.python_version(),
        'numpy': np.__version__,
        'args': vars(args),
        'pop_labels': POP_LABELS,
        'target_rates_hz': [float(x) for x in PD14_TARGET_RATES],
    }
    try:
        import numba
        meta['numba'] = numba.__version__
    except Exception:
        meta['numba'] = None

    print(f"=== mesopopdyn benchmark ===")
    print(f"scalings={args.scalings} T={args.T}s repeats={args.repeats} "
          f"backends={args.backends}\n")

    results = run_suite(args.scalings, args.T, args.dt, args.dt_rec,
                        args.repeats, args.backends)
    save_results(results, meta)


if __name__ == "__main__":
    main()
