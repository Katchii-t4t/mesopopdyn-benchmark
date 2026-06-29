"""
benchmark_c.py -- C-library backends for the mesoscopic PD14 benchmark.

Runs INSIDE WSL/Linux in the 'nest' conda env, AFTER building Schwalger's C
library (see docs/NEST_SETUP.md "Optional: the custom C backends"):
    apt-get install build-essential libgsl-dev libfftw3-dev
    cd upstream_reference/mesopopdyngif_C_v18_Py3 && make

Backends (both via the tested upstream MultiPop, using the REAL multipop_c_18
ctypes wrapper around glm_popdyn_18.so / glm_netw_sim_18.so):
  * meso_c  -> build_network_tilo_populations()  (population equations in C)
  * micro_c -> build_network_tilo_neurons()      (individual GIF neurons in C)

Output JSON matches benchmark.py / benchmark_nest.py so all backends compare
directly via compare_backends.py.

Two MultiPop footguns handled here:
  - the C path caches trajectories to data/ and reloads them -> pass ForceSim=True
    so every repeat actually simulates.
  - simulate() expects step/tstep arrays -> build a zero-stimulus (B=0) pair.

Population ordering is upstream interleaved [L23e, L23i, L4e, L4i, ...].

Karthik Senthil Ganesh, TU Berlin / BCCN internship, 2026.
"""

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS_DIR = ROOT / "results"
UPSTREAM_C = ROOT / "upstream_reference" / "mesopopdyngif_C_v18_Py3"

sys.path.insert(0, str(HERE))        # potjans.py
sys.path.insert(0, str(UPSTREAM_C))  # multipop18 + multipop_c_18 + the .so

from benchmark_nest import (  # reuse the verified param construction + labels
    build_nest_params, POP_LABELS_INTERLEAVED, PD14_TARGET_INTERLEAVED,
)


def _zero_stim(K, tstart, duration):
    step = np.zeros((K, 2))                       # B = 0 -> no stimulus
    tstep = np.array([[tstart, tstart + duration] for _ in range(K)])
    return step, tstep


def run_one(mp, backend, scaling, T, dt, dt_rec, repeats, seed=123,
            meanfield=True):
    kwargs, N = build_nest_params(scaling, meanfield_connectivity=meanfield)
    K = len(N)
    label = backend if meanfield else backend + "_sparse"
    rec = {
        'backend': label, 'scaling': scaling, 'N_total': int(N.sum()),
        'N_per_pop': [int(x) for x in N], 'T': T, 'dt': dt, 'dt_rec': dt_rec,
        'repeats': repeats, 'available': True,
        'connectivity': 'meanfield' if meanfield else 'sparse',
    }
    step, tstep = _zero_stim(K, dt, T - dt)

    def make():
        p = mp.MultiPop(dt=dt, **kwargs)
        p.dt_rec = dt_rec
        if backend == 'meso_c':
            p.build_network_tilo_populations()
        elif backend == 'micro_c':
            p.build_network_tilo_neurons()
        else:
            raise ValueError(backend)
        return p

    # warmup (lib load / first JIT-free run is cheap, but keep parity)
    p = make()
    t0 = time.perf_counter()
    p.simulate(min(0.1, T), step=step, tstep=tstep, seed=seed, ForceSim=True)
    warmup = time.perf_counter() - t0

    times = []
    p_last = None
    for _ in range(repeats):
        p = make()
        t0 = time.perf_counter()
        p.simulate(T, step=step, tstep=tstep, seed=seed, ForceSim=True)
        times.append(time.perf_counter() - t0)
        p_last = p

    times = np.array(times)
    rate = np.asarray(p_last.get_firingrates())
    abs_err = float(np.abs(rate - PD14_TARGET_INTERLEAVED).mean())
    rec.update({
        'status': 'ok',
        'warmup_time_s': warmup,
        'time_mean_s': float(times.mean()),
        'time_std_s': float(times.std()),
        'time_min_s': float(times.min()),
        'times_s': [float(x) for x in times],
        'realtime_factor': float(T / times.min()),
        'mean_rate_hz': [float(x) for x in rate],
        'mean_abs_rate_err_hz': abs_err,
        'pop_order': POP_LABELS_INTERLEAVED,
    })
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backends', nargs='+', default=['meso_c'],
                    choices=['meso_c', 'micro_c'])
    ap.add_argument('--scalings', type=float, nargs='+', default=[0.1, 0.5, 1.0])
    ap.add_argument('--T', type=float, default=10.0)
    ap.add_argument('--dt', type=float, default=0.0001)
    ap.add_argument('--dt_rec', type=float, default=0.001)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--connectivity', choices=['meanfield', 'sparse'],
                    default='meanfield')
    args = ap.parse_args()

    import multipop18 as mp  # real multipop_c_18 (NOT stubbed) + the .so

    meta = {
        'timestamp': datetime.now().isoformat(),
        'platform': platform.platform(),
        'python': platform.python_version(),
        'numpy': np.__version__,
        'args': vars(args),
        'pop_labels': POP_LABELS_INTERLEAVED,
        'target_rates_hz': [float(x) for x in PD14_TARGET_INTERLEAVED],
    }

    print("=== mesopopdyn C benchmark ===")
    results = []
    for backend in args.backends:
        for scaling in args.scalings:
            print(f"[run ] {backend:9s} scaling={scaling:<4g} T={args.T}s ... ",
                  end="", flush=True)
            try:
                rec = run_one(mp, backend, scaling, args.T, args.dt,
                              args.dt_rec, args.repeats,
                              meanfield=(args.connectivity == 'meanfield'))
                print(f"min={rec['time_min_s']:.3f}s "
                      f"(rt x{rec['realtime_factor']:.2f}) "
                      f"rate_err={rec['mean_abs_rate_err_hz']:.3f}Hz")
            except Exception as exc:
                rec = {'backend': backend, 'scaling': scaling,
                       'status': 'error', 'error': repr(exc)}
                print(f"ERROR {exc!r}")
            results.append(rec)

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {'meta': meta, 'results': results}
    (RESULTS_DIR / f"benchmark_c_{stamp}.json").write_text(json.dumps(payload, indent=2))
    (RESULTS_DIR / "benchmark_c_latest.json").write_text(json.dumps(payload, indent=2))
    print(f"\nsaved: benchmark_c_{stamp}.json (+ benchmark_c_latest.json)")


if __name__ == "__main__":
    main()
