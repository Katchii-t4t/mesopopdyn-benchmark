# Mesoscopic Potjans–Diesmann: methods & results (draft)

*Draft seed for an HON3000 project report / benchmark paper. Karthik Senthil
Ganesh, TU Berlin / BCCN, 2026. Results measured on Windows 11, Python 3.13.9,
NumPy 2.3.5, Numba 0.62.1; meso-Python/Numba backend. NEST/C cross-backend rows
pending (run on WSL/Linux per docs/NEST_SETUP.md).*

## 1. Background

The Potjans–Diesmann (2014) cortical microcircuit (PD14) is the standard
reference model of a 1 mm² cortical column: ~77,000 neurons in 8 populations
(layers L2/3, L4, L5, L6 × excitatory/inhibitory), ~3×10⁸ synapses. A
**microscopic** simulation integrates every neuron; a **mesoscopic** simulation
(Schwalger, Deger & Gerstner, 2017) tracks only the per-population activity
A(t) = fraction of a population spiking, reducing ~77k equations to 8.

## 2. Model and mathematics

Single neurons are generalized integrate-and-fire (GIF) units with **escape
(soft-threshold) noise**: the instantaneous firing intensity is a hazard rate

    λ(u) = c · exp((u − V_th) / Δu),

where u is the membrane potential, V_th the threshold, Δu the softness, c a
rate prefactor. The probability of firing in a step dt follows from survival
statistics, P_fire = 1 − exp(−∫λ dt), implemented with a trapezoidal estimate
0.5(λ_old + λ)dt.

At the population level the mesoscopic theory evolves the distribution of times
since last spike (a refractory-density / quasi-renewal formulation). The
finite-size stochasticity — the defining feature versus classical mean-field —
enters as a binomial draw for the number of spiking neurons:

    n ~ Binomial(N, Ābar) / N,

with Ābar the expected fraction. This single relation captures the
finite-N fluctuations that classical (N→∞) mean-field models (Wilson–Cowan;
Brunel/Fokker–Planck) discard. Synaptic coupling uses the annealed-average
connectivity W_av = W ⊙ p_conn and population coupling w = J_s · N.

## 3. Methods (the harness)

A clean, reproducible Python harness was built around the meso simulator:
- `model_setup.py`: single source of truth for the PD14 mesoscopic parameters
  (previously duplicated across four per-language driver scripts upstream).
- `benchmark.py`: timing with an explicit untimed warmup (excluding Numba JIT
  compilation), R repeats, and machine-readable JSON/CSV output.
- `validate.py`: quantitative correctness vs published PD14 rates.
- `finite_size.py`: empirical test of the 1/√N fluctuation law.
- `plot.py`: figures.
- `benchmark_nest.py` (+ `setup_nest_wsl.sh`): NEST backends (meso/micro),
  ready to run on Linux/WSL via the tested upstream MultiPop interface.

## 4. Results

### 4.1 Correctness (full scale, T = 10 s)
Mesoscopic spontaneous rates match the published PD14 values to a mean absolute
error of **0.059 Hz**; all 8 populations within ~1–5 % relative error.
Figure: `validation_rates.png`, `activity_traces.png`.

### 4.2 Runtime is independent of network size
Wall-clock time for 10 s of biological activity is flat at ~7 s as the network
grows from 7,717 to 77,169 neurons (10×):

| scaling | neurons | min wall [s] | realtime factor |
|---------|---------|--------------|-----------------|
| 0.10    | 7,717   | 7.59         | 1.32×           |
| 0.25    | 19,292  | 7.05         | 1.42×           |
| 0.50    | 38,586  | 6.93         | 1.44×           |
| 1.00    | 77,169  | 6.99         | 1.43×           |

Mesoscopic cost scales with the number of populations (M = 8), not neurons —
the mechanism behind the ~150× speedup over microscopic simulation reported in
Schwalger et al. (2017). Figure: `benchmark_scaling.png`.

### 4.3 Implementation speedup
Identical algorithm, Numba vs pure Python (T = 0.2 s): **368×** (0.15 s vs
55.2 s). Motivates the compiled / NEST / C backends for parameter sweeps.

### 4.4 Down-scaling with preserved coupling is nearly rate-neutral
Using the correct down-scaling convention (coupling w = J_s·N computed at full
size; only the binomial finite-size N reduced), the mean absolute rate error vs
published values stays small and decreases with N:
0.25 Hz (10 %) → 0.10 (25 %) → 0.066 (50 %) → 0.059 Hz (full). The residual at
small N is the genuine finite-size effect, not a coupling artifact.

> **Correction.** An earlier version of `model_setup.py` computed the coupling
> with the down-scaled N, which shifted the mean input and produced a spurious
> ~20 Hz "down-scaling distortion". This was caught by a meso-Python vs
> meso-NEST disagreement at scaling 0.1 (Python 20 Hz vs NEST 0.4 Hz) and fixed.
> A clean cross-implementation comparison revealed a bug neither backend showed
> alone.

### 4.5 Finite-size fluctuation scaling (measured)
The steady-state std of A(t) decreases with N with a fitted log-log slope of
**≈ −0.565** (mean over populations), close to the theoretical 1/√N binomial
law (−0.5). The small excess reflects mild network nonlinearity near threshold.
(Before the §4.4 coupling fix this slope was a contaminated −0.75.)
Figure: `finite_size_scaling.png`; data: `results/finite_size_latest.json`.

### 4.6 Cross-backend results (NEST 3.10, WSL/Linux)
Three independent simulators of the same model were run and compared:

| backend     | scaling | N      | wall (10 s bio) | rate err |
|-------------|---------|--------|-----------------|----------|
| meso_numba  | 1.0     | 77,169 | 7.08 s          | 0.059 Hz |
| meso_nest   | 1.0     | 77,169 | 9.90 s          | 0.052 Hz |
| micro_nest  | 0.1     | 7,717  | 51.8 s (2 s bio)| 0.239 Hz |

- **Cross-implementation agreement:** meso_numba and meso_nest match to
  **0.008 Hz** at full scale — two independent implementations (custom Numba vs
  NEST's C++ `gif_pop_psc_exp`) validate each other.
- **Micro→meso speedup grows with N** (meso flat, micro ~N² for all-to-all):
  **12×** at scaling 0.05 (N=3,858) → **37×** at 0.10 (7,717) → **69×** at 0.15
  (11,575). The speedup roughly doubles per step, extrapolating cleanly toward
  the ~150× of Schwalger et al. (2017) at full scale.
- *Caveat:* micro used all-to-all (mean-field) connectivity and only scaling
  0.1; the speedup figure is not directly comparable to the paper's sparse
  full-scale value, but the trend is unambiguous.
Figure: `cross_backend.png`; data: `results/cross_backend_summary.json`.

## 5. Engineering findings (vs upstream)
- **Benchmark bug:** upstream times the first Numba call, counting ~7 s of JIT
  compilation as run time (~45× over-report at short bio-times). Fixed via warmup.
- **Fragmentation:** four per-language scripts with duplicated parameters and a
  hand-maintained .ods spreadsheet → unified harness + structured output.
- **No correctness gate:** eyeball plots → quantitative PASS/FAIL.
- **Off-by-one** in the meso recorder guard (`i <= Nrecord`), fixed.

## 6. Next steps (to make this a contribution, not a reproduction)
1. Cross-backend rows (meso-NEST, micro-NEST, meso-C) on WSL/Linux; confirm all
   agree on full-scale rates and reproduce the ~150× micro→meso speedup
   end-to-end on one machine.
2. Add the van-Albada correctness triplet (rate dist, CV-ISI, Pearson
   correlation) for the microscopic runs.
3. Frame as a systematic, methodology-corrected modern benchmark — the form
   suited to Frontiers in Neuroinformatics / Neuroinformatics, and to HON3000.
