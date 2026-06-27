# Mesoscopic Potjans–Diesmann — benchmarking & validation harness

A clean, reproducible harness around the **mesoscopic (population-rate)**
Potjans–Diesmann (2014) cortical microcircuit, built on top of
[`schwalger/mesopopdyn_microcircuit`](https://github.com/schwalger/mesopopdyn_microcircuit)
and the mesoscopic theory of Schwalger, Deger & Gerstner, *PLoS Comput Biol*
(2017).

TU Berlin / BCCN internship (Schwalger group), Karthik Senthil Ganesh, 2026.
See [`docs/FINDINGS.md`](docs/FINDINGS.md) for results and
[`docs/PLAN.md`](docs/PLAN.md) for scope/next steps.

## Layout

```
harness/
  model_setup.py   single source of truth for the PD14 meso parameters
  sim_meso.py      the Numba/pure-Python mesoscopic simulator (from upstream)
  potjans.py       PD14 network parameters (from upstream)
  benchmark.py     unified timing harness (warmup, repeats, JSON/CSV out)
  benchmark_nest.py NEST backends (meso/micro), run on WSL/Linux
  validate.py      quantitative correctness vs published PD14 rates
  finite_size.py   empirical 1/sqrt(N) fluctuation-scaling test
  test_harness.py  reproducibility tests (pytest)
  plot.py          figures -> ../figures/
  setup_nest_wsl.sh one-command NEST install inside WSL
results/           benchmark + validation outputs (JSON/CSV)
figures/           generated PNGs
docs/              FINDINGS.md, METHODS_RESULTS.md, PLAN.md, NEST_SETUP.md
upstream_reference/  vendored copy of schwalger/mesopopdyn_microcircuit
```

## Requirements

Python 3.11+ with NumPy, Numba, SciPy, matplotlib (`pip install -r requirements.txt`).
The C / NEST / Julia backends need extra toolchains and are not required for the
Python harness — see `docs/PLAN.md`.

## Quick start

```bash
cd harness
python validate.py  --scaling 1.0 --T 10          # correctness check (PASS/FAIL)
python benchmark.py --scalings 0.1 0.5 1.0 --T 10  # timing sweep -> results/
python finite_size.py                              # 1/sqrt(N) fluctuation test
python plot.py                                     # figures -> figures/
python test_harness.py                             # reproducibility tests
```

`python benchmark.py --quick` runs a fast smoke configuration.
For the NEST backends (Linux/WSL only) see `docs/NEST_SETUP.md`.

## Key results (this machine: Win11, Py3.13, NumPy 2.3, Numba 0.62)

- Mesoscopic rates match published PD14 to **0.059 Hz** mean abs error at full
  scale (all 8 populations < 5 % rel err).
- Mesoscopic runtime is **flat (~7 s / 10 s bio) from 7.7k to 77k neurons** —
  cost scales with #populations, not #neurons.
- Numba vs pure Python: **368×**.
- Down-scaling distorts mean rates (20 Hz err at 10 % size → 0.06 Hz at full
  size); benchmark for *timing* at reduced size, validate *correctness* at full.
- Finite-size fluctuations of A(t) shrink with N at a fitted slope ≈ **−0.75**
  (steeper than the naive 1/√N), reflecting the operating-point shift at reduced
  size. See `docs/METHODS_RESULTS.md` for the full write-up.
