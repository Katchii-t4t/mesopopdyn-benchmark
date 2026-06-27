"""
test_harness.py -- fast reproducibility tests for the mesoscopic PD14 harness.

Run:  python -m pytest test_harness.py -q   (or: python test_harness.py)

These lock in the core scientific claims so refactors can't silently break them:
  * full-scale rates match published PD14 within tolerance,
  * the simulation is deterministic given a seed,
  * mesoscopic runtime is ~independent of network size,
  * finite-size fluctuations shrink with N.

Kept short (small T) so the whole suite runs in well under a minute.

Karthik Senthil Ganesh, TU Berlin / BCCN internship, 2026.
"""

import time

import numpy as np

from model_setup import (
    build_params, build_step_stimulus, PD14_TARGET_RATES,
)
from sim_meso import simulate


def _run(scaling, T, seed=12345, dt=1e-4, dt_rec=1e-3):
    params, N = build_params(scaling)
    params = build_step_stimulus(params, T, dt_rec, B=0.0)
    rate, A = simulate(T, dt, dt_rec, params, 8, seed)
    return rate, A, N


def test_full_scale_rates_match_published():
    _, A, _ = _run(1.0, T=10.0)
    n0 = int(0.5 / 1e-3)
    mean_rate = A[:, n0:].mean(axis=1)
    err = np.abs(mean_rate - PD14_TARGET_RATES)
    assert err.mean() < 0.5, f"mean abs rate error too high: {err.mean():.3f} Hz"
    assert np.all(err / np.maximum(PD14_TARGET_RATES, 1e-9) < 0.15), \
        "a population deviates >15% from published rate"


def test_deterministic_given_seed():
    _, A1, _ = _run(0.5, T=1.0, seed=7)
    _, A2, _ = _run(0.5, T=1.0, seed=7)
    assert np.array_equal(A1, A2), "same seed must give identical output"


def test_different_seeds_differ():
    _, A1, _ = _run(0.5, T=1.0, seed=1)
    _, A2, _ = _run(0.5, T=1.0, seed=2)
    assert not np.array_equal(A1, A2), "different seeds should differ"


def test_runtime_roughly_independent_of_N():
    # warmup (compile) excluded
    _run(0.1, T=0.05)
    T = 1.0
    t0 = time.perf_counter(); _run(0.1, T=T); t_small = time.perf_counter() - t0
    t0 = time.perf_counter(); _run(1.0, T=T); t_big = time.perf_counter() - t0
    # 10x neurons should NOT cost ~10x; allow generous 2x band for noise
    assert t_big < 2.0 * t_small + 0.5, \
        f"runtime scaled with N unexpectedly: {t_small:.3f}s vs {t_big:.3f}s"


def test_finite_size_fluctuations_shrink_with_N():
    _, A_small, _ = _run(0.1, T=5.0)
    _, A_big, _ = _run(1.0, T=5.0)
    n0 = int(1.0 / 1e-3)
    std_small = A_small[0, n0:].std()
    std_big = A_big[0, n0:].std()
    assert std_big < std_small, \
        f"fluctuations should shrink with N: {std_small:.3f} -> {std_big:.3f}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
