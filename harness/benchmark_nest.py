"""
benchmark_nest.py -- NEST backends for the mesoscopic PD14 benchmark.

Runs INSIDE WSL/Linux in the 'nest' conda env (see setup_nest_wsl.sh).
Produces JSON in the SAME schema as benchmark.py so the meso-Python results and
the NEST (meso + micro) results can be compared directly / merged.

Design choice: rather than re-derive the NEST parameter mapping by hand
(error-prone, and untestable without NEST), this reuses Tilo's *tested*
`multipop18.MultiPop` from the upstream repo:
  * build_network_populations()  -> mesoscopic NEST  (gif_pop_psc_exp)
  * build_network_neurons()      -> microscopic NEST (gif_psc_exp)
The custom C library is NOT required for the NEST paths, so we install a stub
'multipop_c_18' module before importing multipop18 (the C module is only used by
the C/'tilo' backends, which we don't run here).

Parameter construction is copied verbatim from the upstream
benchmarking/PDmodel_C-code-tilo_NEST.py to guarantee identical physics; the
only additions are warmup, repeats, timing and JSON output.

Population ordering here is the upstream interleaved order:
  [L23e, L23i, L4e, L4i, L5e, L5i, L6e, L6i].

Usage (inside WSL, 'nest' env):
    python benchmark_nest.py --backends meso_nest --scalings 0.1 0.5 --T 10
    python benchmark_nest.py --backends meso_nest micro_nest --scalings 0.1 --T 2

Karthik Senthil Ganesh, TU Berlin / BCCN internship, 2026.
"""

import argparse
import json
import platform
import sys
import time
import types
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS_DIR = ROOT / "results"
UPSTREAM_C = ROOT / "upstream_reference" / "mesopopdyngif_C_v18_Py3"

# population labels in the upstream interleaved ordering
POP_LABELS_INTERLEAVED = ['L23e', 'L23i', 'L4e', 'L4i',
                          'L5e', 'L5i', 'L6e', 'L6i']
# published PD14 rates, same interleaved order
PD14_TARGET_INTERLEAVED = np.array([0.971, 2.868, 4.746, 5.396,
                                    8.142, 9.078, 0.991, 7.523])


def _import_multipop():
    """Import upstream multipop18 with a stub C module so NEST-only works."""
    sys.path.insert(0, str(UPSTREAM_C))
    if 'multipop_c_18' not in sys.modules:
        stub = types.ModuleType('multipop_c_18')
        class _Unavailable:
            def __init__(self, *a, **k):
                raise RuntimeError(
                    "C backend not built; benchmark_nest.py only runs the NEST "
                    "backends. Build the C lib (make) to use the C/'tilo' paths.")
        stub.Multipop = _Unavailable
        sys.modules['multipop_c_18'] = stub
    import multipop18 as mp  # noqa: E402
    return mp


def build_nest_params(scaling, meanfield_connectivity=True):
    """Construct the MultiPop kwargs, copied from upstream NEST driver."""
    from potjans import (
        neuron_params, weight_matrix, conn_probs, d_mean, N_full,
    )
    K = len(conn_probs)
    DeltaV = 5. * np.ones(K)
    # interleaved order
    mu = np.array([19.14942021, 20.36247192, 30.80455243, 28.06868188,
                   29.43680143, 29.32954448, 34.93225564, 32.08123423])
    tau_theta = ([[0.1], [0.1]] * 4)
    J_theta = np.array([[0.], [0.]] * 4)
    c = np.ones(K) * 10.
    Vreset = [neuron_params['v_reset'] - neuron_params['v_rest']] * K
    Vth = [neuron_params['v_thresh'] - neuron_params['v_rest']] * K
    delay = d_mean['E'] * 1e-3
    t_ref = neuron_params['tau_refrac'] * 1e-3
    N = np.array([N_full['L23']['E'], N_full['L23']['I'],
                  N_full['L4']['E'], N_full['L4']['I'],
                  N_full['L5']['E'], N_full['L5']['I'],
                  N_full['L6']['E'], N_full['L6']['I']])

    Js = weight_matrix.copy()
    for col in range(0, K, 2):   # excitatory source columns
        Js[:, col] *= (neuron_params['tau_syn_E'] * 1e-3) / neuron_params['cm'] * 1e3
    for col in range(1, K, 2):   # inhibitory source columns
        Js[:, col] *= (neuron_params['tau_syn_I'] * 1e-3) / neuron_params['cm'] * 1e3

    N = np.round(N * scaling).astype(int)
    Js = Js / scaling

    taus1_ = [neuron_params['tau_syn_E'] * 1e-3,
              neuron_params['tau_syn_I'] * 1e-3] * 3
    taus1 = [taus1_ for _ in range(K)]

    pconn = np.array(conn_probs)
    if meanfield_connectivity:
        Js = Js * pconn
        pconn = np.ones((K, K))

    taum = np.array([neuron_params['tau_m'] * 1e-3] * K)

    return dict(
        N=N, rho_0=c, tau_m=taum, tau_sfa=tau_theta, J_syn=Js, taus1=taus1,
        delay=np.ones((K, K)) * delay, t_ref=np.ones(K) * t_ref,
        V_reset=np.array(Vreset), J_a=J_theta, pconn=pconn, mu=np.array(mu),
        delta_u=DeltaV * np.ones(K), V_th=np.array(Vth) * np.ones(K),
        sigma=np.zeros(K), mode='glif',
    ), N


def run_one(mp, backend, scaling, T, dt, dt_rec, repeats, seed=123):
    kwargs, N = build_nest_params(scaling)
    rec = {
        'backend': backend, 'scaling': scaling, 'N_total': int(N.sum()),
        'N_per_pop': [int(x) for x in N], 'T': T, 'dt': dt, 'dt_rec': dt_rec,
        'repeats': repeats, 'available': True,
    }

    def make():
        p = mp.MultiPop(dt=dt, **kwargs)
        p.dt_rec = dt_rec
        if backend == 'meso_nest':
            p.build_network_populations()
        elif backend == 'micro_nest':
            p.local_num_threads = 1
            p.build_network_neurons()
        else:
            raise ValueError(backend)
        return p

    # warmup (NEST kernel build, import caches) -- not timed
    t0 = time.perf_counter()
    p = make(); p.simulate(min(0.1, T), seed=seed)
    warmup = time.perf_counter() - t0

    times = []
    p_last = None
    for _ in range(repeats):
        p = make()
        t0 = time.perf_counter()
        p.simulate(T, seed=seed)
        times.append(time.perf_counter() - t0)
        p_last = p

    times = np.array(times)
    rate = p_last.get_firingrates()  # mean per population
    abs_err = float(np.abs(np.asarray(rate) - PD14_TARGET_INTERLEAVED).mean())
    rec.update({
        'status': 'ok',
        'warmup_time_s': warmup,
        'time_mean_s': float(times.mean()),
        'time_std_s': float(times.std()),
        'time_min_s': float(times.min()),
        'times_s': [float(x) for x in times],
        'realtime_factor': float(T / times.min()),
        'mean_rate_hz': [float(x) for x in np.asarray(rate)],
        'mean_abs_rate_err_hz': abs_err,
        'pop_order': POP_LABELS_INTERLEAVED,
    })
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backends', nargs='+', default=['meso_nest'],
                    choices=['meso_nest', 'micro_nest'])
    ap.add_argument('--scalings', type=float, nargs='+', default=[0.1, 0.5, 1.0])
    ap.add_argument('--T', type=float, default=10.0)
    ap.add_argument('--dt', type=float, default=0.0001)
    ap.add_argument('--dt_rec', type=float, default=0.001)
    ap.add_argument('--repeats', type=int, default=3)
    args = ap.parse_args()

    sys.path.insert(0, str(HERE))  # for potjans.py
    mp = _import_multipop()

    meta = {
        'timestamp': datetime.now().isoformat(),
        'platform': platform.platform(),
        'python': platform.python_version(),
        'numpy': np.__version__,
        'args': vars(args),
        'pop_labels': POP_LABELS_INTERLEAVED,
        'target_rates_hz': [float(x) for x in PD14_TARGET_INTERLEAVED],
    }
    try:
        import nest
        meta['nest'] = nest.__version__
    except Exception as exc:
        print(f"FATAL: cannot import nest ({exc}). Run inside the 'nest' env.")
        sys.exit(1)

    print(f"=== mesopopdyn NEST benchmark (NEST {meta['nest']}) ===")
    results = []
    for backend in args.backends:
        for scaling in args.scalings:
            print(f"[run ] {backend:11s} scaling={scaling:<4g} T={args.T}s ... ",
                  end="", flush=True)
            try:
                rec = run_one(mp, backend, scaling, args.T, args.dt,
                              args.dt_rec, args.repeats)
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
    out = RESULTS_DIR / f"benchmark_nest_{stamp}.json"
    out.write_text(json.dumps(payload, indent=2))
    (RESULTS_DIR / "benchmark_nest_latest.json").write_text(json.dumps(payload, indent=2))
    print(f"\nsaved: {out.name} (+ benchmark_nest_latest.json)")


if __name__ == "__main__":
    main()
