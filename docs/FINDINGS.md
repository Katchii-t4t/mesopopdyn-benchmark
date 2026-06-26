# Findings & improvements — mesoscopic Potjans–Diesmann benchmarking

**Author:** Karthik Senthil Ganesh — TU Berlin / BCCN (Schwalger group), 2026
**Scope:** review of `schwalger/mesopopdyn_microcircuit` and a first clean,
runnable benchmarking + validation harness around the mesoscopic
(population-level) PD14 simulation.

---

## 1. What the original repository does

`mesopopdyn_microcircuit` implements the no-adaptation Potjans–Diesmann (2014)
cortical microcircuit, both **microscopically** (individual GIF neurons) and
**mesoscopically** (population-rate dynamics from Schwalger et al. 2017), across
**four backends**: a custom C library, NEST (`gif_pop_psc_exp` /
`gif_psc_exp`), Python+Numba, and Julia. Three folders: `single_run/`
(reproduce Fig. 1/8 of the 2017 paper), `benchmarking/` (10 s of biological
time for timing), `mesopopdyngif_C_v18_Py3/` (the C sources).

It works and is scientifically sound. The weaknesses are in **engineering and
benchmarking methodology**, which is exactly what Tilo flagged.

---

## 2. Concrete problems found (with evidence)

### 2.1 Numba JIT compilation is counted as run time  *(real benchmarking bug)*
In `benchmarking/PDmodel_mesopython.py` the Numba `simulate` is timed on its
**first** call:

```python
start_time = time.time()
rate, A = simulate(T, dt, dt_record, params, Nrecord, seed)   # first call -> JIT compiles here
end_time = time.time()
print("execution time MESO NUMBA: %s seconds" % (end_time - start_time))
```

Numba compiles on first call, so the reported time includes compilation.

**Measured on this machine:** warmup/compile ≈ **6.0–7.0 s**, while the actual
10 s-bio simulation is ≈ **7.0 s**. At short bio-times the distortion is
enormous: for `T = 0.2 s`, the compiled run is **0.15 s** but a naive
first-call timing would report **~7 s** — a **~45× over-report**.

**Fix (implemented):** `benchmark.py` always does an untimed warmup run before
timing, and reports mean/std/min over several repeats.

### 2.2 Fragmentation / duplication
- Four separate driver scripts, one per language, with **the same parameter
  construction copy-pasted** into each (and into the Julia `.mat` exporter).
  Parameter drift between backends is a real risk.
- Timings are printed to stdout and transcribed by hand into
  `benchmarking/run_times.ods`. Not machine-readable, not reproducible.
- The Julia path needs a separate manual step (`save_params_for_julia.py`) to
  regenerate `params_PDmodel.mat`.

**Fix (implemented):** one `model_setup.py` is the single source of truth for
the parameter dictionary; `benchmark.py` writes structured JSON + CSV.

### 2.3 No built-in correctness validation
Validation is by eye (matplotlib traces). The field standard for PD14
(van Albada et al. 2018) is quantitative: firing-rate distributions, CV of ISI,
correlation coefficients.

**Fix (implemented):** `validate.py` checks per-population mean rates against the
published PD14 values with an automatic PASS/FAIL verdict and a fluctuation
(CV) proxy.

### 2.4 Minor latent bug — off-by-one in the recording guard
`sim4.py` line ~149: `if (i <= Nrecord)` records population index `Nrecord`
when `Nrecord < M` (should be `i < Nrecord`). Harmless when `Nrecord == M == 8`
(the only configuration shipped), but wrong if anyone records a subset.

### 2.5 Reproducibility hygiene
No `requirements.txt`, no environment/version capture, no recorded seeds in the
output. **Fix:** `requirements.txt` added; `benchmark.py` records platform,
Python/NumPy/Numba versions, args and seeds in every results file.

---

## 3. Results obtained on this machine

Environment: Windows 11, Python 3.13.9, NumPy 2.3.5, Numba 0.62.1.
Only the Numba/pure-Python backends are runnable here (NEST has no native
Windows build; gcc and Julia are not installed — see `PLAN.md`).

### 3.1 Correctness — mesoscopic vs published PD14 rates (full scale, T = 10 s)

| pop  | sim [Hz] | target [Hz] | rel err |
|------|----------|-------------|---------|
| L23e | 0.968    | 0.971       | 0.3 %   |
| L4e  | 4.673    | 4.746       | 1.5 %   |
| L5e  | 8.194    | 8.142       | 0.6 %   |
| L6e  | 0.997    | 0.991       | 0.6 %   |
| L23i | 2.858    | 2.868       | 0.3 %   |
| L4i  | 5.659    | 5.396       | 4.9 %   |
| L5i  | 9.033    | 9.078       | 0.5 %   |
| L6i  | 7.542    | 7.523       | 0.3 %   |

Mean absolute error **0.059 Hz**; verdict **PASS** (tol 10 %). See
`figures/validation_rates.png`, `figures/activity_traces.png`.

### 3.2 Runtime is independent of network size  *(the core mesoscopic property)*

| scaling | # neurons | min wall [s] / 10 s bio | realtime factor |
|---------|-----------|--------------------------|-----------------|
| 0.10    | 7 717     | 7.59                     | 1.32×           |
| 0.25    | 19 292    | 7.05                     | 1.42×           |
| 0.50    | 38 586    | 6.93                     | 1.44×           |
| 1.00    | 77 169    | 6.99                     | 1.43×           |

Wall time is **flat (~7 s) across a 10× change in neuron count**, because the
mesoscopic cost depends on the number of *populations* (M = 8), not the number
of *neurons*. This is precisely the property that gives the ~150× speedup over
microscopic simulation reported in Schwalger et al. (2017).
See `figures/benchmark_scaling.png`.

### 3.3 Numba vs pure Python: **368× speedup**
Same algorithm, `T = 0.2 s`: Numba **0.15 s** vs pure Python **55.2 s**
(`results/numba_vs_python.json`). Motivates keeping the compiled/C/NEST paths
for any real parameter sweep.

### 3.4 Down-scaling distorts mean rates  *(a real neuroscience caveat)*
Mean absolute rate error vs network size: **20.1 Hz** (0.1) → **9.5** (0.25) →
**3.7** (0.5) → **0.06 Hz** (1.0). Down-scaling the column is *not* rate-neutral
(consistent with van Albada et al. 2015) — benchmarking at reduced size is fine
for *timing* but not for *correctness*. The harness therefore only treats
full-scale runs as authoritative for pass/fail.

---

## 4. Recommendations for the joint project

1. **One model definition, many backends.** Promote `model_setup.py` to the
   canonical parameter source and have C/NEST/Julia drivers consume the same
   numbers (the C/NEST drivers already accept a params object via `MultiPop`).
2. **Benchmark methodology as a first-class artifact:** warmup, repeats,
   machine metadata, JSON output, and a fixed correctness gate — done here for
   Python; replicate for the other backends.
3. **Add the van-Albada correctness triplet** (rate, CV-ISI, correlation) for
   the *microscopic* runs so micro↔meso agreement is quantified, not eyeballed.
4. **Make the micro→meso reduction the research deliverable.** The clean,
   reusable harness is the platform; the scientific contribution is showing the
   reduction generalizes (and where it breaks — e.g. the original Poisson-driven
   PD14 needs the hazard-rate approximation of Schwalger 2021).
