# Running the NEST backends (WSL2)

NEST has no native Windows build, so the NEST (and C/Julia) backends run inside
WSL2 (Ubuntu). The meso-Python harness (`benchmark.py`, `validate.py`) runs fine
on Windows directly; this file is only for the NEST backends.

## One-time setup

1. **Install WSL2** (admin PowerShell, then reboot):
   ```powershell
   wsl --install
   ```
   Reboot when prompted. On first launch Ubuntu asks for a username/password.

2. **Install NEST** (inside WSL):
   ```bash
   wsl
   cd /mnt/c/Users/Karthik/Documents/mesopopdyn-internship/harness
   bash setup_nest_wsl.sh
   ```
   This installs Miniforge (if needed), creates a `nest` conda env with
   `nest-simulator` + deps, and verifies `import nest` and the
   `gif_pop_psc_exp` / `gif_psc_exp` models.

## Running

```bash
wsl
conda activate nest
cd /mnt/c/Users/Karthik/Documents/mesopopdyn-internship/harness

# mesoscopic NEST, timing sweep
python benchmark_nest.py --backends meso_nest --scalings 0.1 0.5 1.0 --T 10

# microscopic NEST (slow at full scale; start small)
python benchmark_nest.py --backends micro_nest --scalings 0.1 --T 2
```

Output lands in `../results/benchmark_nest_*.json` in the **same schema** as the
Python harness, so meso-Python, meso-NEST and micro-NEST results compare
directly. Population order in the NEST output is interleaved
(`L23e, L23i, L4e, L4i, ...`); the meso-Python output is E-first
(`L23e, L4e, L5e, L6e, L23i, ...`) — match by label, not index.

## What this unlocks (the real contribution)

Once both run, you can show, with your own measurements on one machine:
- meso-NEST vs meso-Python agree on the rates (cross-implementation check);
- micro-NEST vs meso-NEST agree on the rates but meso is far faster
  (reproduce the ~150× speedup of Schwalger et al. 2017 end-to-end).

That micro↔meso comparison is what turns "I reproduced the model" into
"I benchmarked the mesoscopic reduction" — the deliverable Tilo asked for.

## Optional: the custom C backends

To also benchmark Tilo's C library (meso-C, micro-C), build it inside WSL:
```bash
conda activate nest
conda install -c conda-forge gsl fftw c-compiler
cd ../upstream_reference/mesopopdyngif_C_v18_Py3
make CC=$CC CFLAGS="-I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -lm -lgsl -lgslcblas -lfftw3 -O3 -w"
```
Then the upstream `PDmodel_C-code-tilo_NEST.py` runs all four backends; the
`benchmark_nest.py` stub trick is no longer needed.
