# Project plan — mesoscopic PD14 benchmarking & micro→meso reduction

**Internship:** TU Berlin / BCCN, Schwalger group (with Iago Ungera Fraga).
**Framing:** *proof-of-concept + clean benchmarking platform* for the week,
laying the groundwork for the larger micro→meso reduction project.

> Honest scoping (see also the conversation): a *full* generalized micro→meso
> reduction pipeline is a multi-month research project — the mathematical core
> (hazard-rate function, finite-size corrections) is Schwalger's own 2017–2021
> work. What is achievable and valuable in one week is the **benchmarking +
> validation platform** plus a concrete demonstration on the meso path, which
> is exactly the gap Tilo identified.

---

## Status (done autonomously, night 1)

- [x] Reviewed paper (Plesser et al. 2025) + all four repos/links.
- [x] Single source-of-truth parameter builder (`harness/model_setup.py`).
- [x] Unified benchmarking harness with warmup/repeats/JSON (`harness/benchmark.py`).
- [x] Quantitative validation vs published PD14 rates (`harness/validate.py`).
- [x] Figures (`harness/plot.py` → `figures/`).
- [x] Full benchmark sweep (scaling 0.1/0.25/0.5/1.0, T=10 s) — runtime is
      flat in N; rates match to 0.06 Hz at full scale.
- [x] Findings written up (`docs/FINDINGS.md`).

## Priority list (continues from here)

1. **NEST environment** *(blocked on this machine — needs setup)*
   NEST has no native Windows build. Options, easiest first:
   - **WSL2 + conda:** `conda create -n nest -c conda-forge nest-simulator python=3.11`
   - Docker image `nest/nest-simulator`.
   - On the TU Berlin office workstation/cluster NEST is likely already available.
   Then run `upstream_reference/benchmarking/PDmodel_C-code-tilo_NEST.py` and
   wire the `meso_nest` / `meso_c` backends in `benchmark.py` (stubs already in
   place with notes).

2. **C library build** *(needs gcc + GSL + FFTW)*
   `cd upstream_reference/mesopopdyngif_C_v18_Py3 && make` then expose
   `multipop18.MultiPop.build_network_tilo_populations` through the harness.

3. **Cross-backend correctness:** confirm meso-Python, meso-C and meso-NEST
   agree to numerical tolerance on the full-scale rates (extend `validate.py`
   to accept a backend argument).

4. **van-Albada correctness triplet for the micro runs** (rate dist, CV-ISI,
   Pearson correlation) to quantify micro↔meso agreement and reproduce the
   ~150× speedup claim end-to-end on one machine.

5. **Write-up:** either a short methods/benchmark report or a project page on
   Tilo's / personal site. `docs/FINDINGS.md` + the figures are the seed.

## Open questions for the Wednesday meeting (Tilo + Iago)

- Is the intended deliverable (a) a *cleaner benchmarking + reproduction
  platform*, or (b) a step toward a *generalized* micro→meso reduction tool in
  NEST? They scope very differently.
- Target network: stick with no-adaptation PD14, or include adaptation
  (`gif_pop_psc_exp` supports `q_sfa`/`tau_sfa`)?
- Original Poisson-driven PD14 vs the GIF variant — do we want the
  hazard-rate approximation (Schwalger 2021) so the *original* model also gets
  a mesoscopic description?
- Where does this live and who co-authors (report vs personal/Tilo page)?
