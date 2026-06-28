"""
model_setup.py -- single source of truth for building the mesoscopic
Potjans-Diesmann (PD14) parameter dictionary used by the meso-Python/Numba
simulation (sim_meso.simulate).

This factors out the parameter-construction logic that was previously
*duplicated* across PDmodel_mesopython.py, PDmodel_mesojulia.jl and
PDmodel_C-code-tilo_NEST.py in Schwalger's repository, so that every backend
can be fed from one verified definition.

Reference: Schwalger, Deger & Gerstner, PLoS Comput Biol (2017),
            no-adaptation variant of the Potjans-Diesmann (2014) microcircuit.

Karthik Senthil Ganesh, TU Berlin / BCCN internship, 2026.
"""

import numpy as np
from potjans import (
    N_full, neuron_params, weight_matrix, conn_probs, d_mean, tha_drive_vec,
)

# Population ordering used by the meso-Python sim: all excitatory first, then
# all inhibitory:  [L23e, L4e, L5e, L6e, L23i, L4i, L5i, L6i]
M = 8
Me = 4  # number of excitatory populations

# Calibrated background drive (mu) for the no-adaptation variant, taken from
# Schwalger's corrected potjans_noadap_trajec3_mdopt.py. Order matches the
# E-first / I-second population ordering above.
MU_DEFAULT = np.array([19.14942021, 30.80455243, 29.43680143, 34.93225564,
                       20.36247192, 28.06868188, 29.32954448, 32.08123423])

# Published full-scale spontaneous mean firing rates (Hz), same population
# ordering. Used by validate.py as the ground-truth target.
#   L23e, L4e, L5e, L6e, L23i, L4i, L5i, L6i
PD14_TARGET_RATES = np.array([0.971, 4.746, 8.142, 0.991,
                              2.868, 5.396, 9.078, 7.523])

POP_LABELS = ['L23e', 'L4e', 'L5e', 'L6e', 'L23i', 'L4i', 'L5i', 'L6i']


def build_params(scaling_factor=1.0, mu=None, hazard_c=10.0, DeltaV=5.0):
    """Build the params dict consumed by sim_meso.simulate.

    Parameters
    ----------
    scaling_factor : float
        Fraction of the full network size (1.0 = full ~77k-neuron column).
    mu : array-like or None
        Background drive per population. Defaults to MU_DEFAULT.
    hazard_c : float
        Hazard rate prefactor c (Hz).
    DeltaV : float
        Soft-threshold width Delta_u (mV), broadcast over populations.

    Returns
    -------
    params : dict
        Ready to pass to sim_meso.simulate, *except* for the 'Iext' field,
        which must be added by the caller (build_step_stimulus).
    N : np.ndarray
        Population sizes after scaling (int).
    """
    if mu is None:
        mu = MU_DEFAULT.copy()
    mu = np.asarray(mu, dtype=float)

    DeltaV_vec = DeltaV * np.ones(M)

    # population sizes, E-first ordering
    N = np.array([N_full['L23']['E'], N_full['L4']['E'],
                  N_full['L5']['E'], N_full['L6']['E'],
                  N_full['L23']['I'], N_full['L4']['I'],
                  N_full['L5']['I'], N_full['L6']['I']])

    # --- annealed-average connectivity: W_av = W * p_conn ---
    W = weight_matrix
    W_av = W * conn_probs

    # The potjans.py weight/conn matrices use an interleaved E/I ordering
    # (L23e, L23i, L4e, L4i, ...). Re-order into E-first / I-second.
    src_perm = [0, 2, 4, 6, 1, 3, 5, 7]  # columns: source populations
    Js_tmp = W_av[:, src_perm]
    Js = Js_tmp[src_perm, :]             # rows: target populations

    # turn PSC amplitudes into PSP amplitudes
    Js[:, :Me] *= (neuron_params['tau_syn_E'] * 1e-3) / neuron_params['cm'] * 1e3
    Js[:, Me:] *= (neuron_params['tau_syn_I'] * 1e-3) / neuron_params['cm'] * 1e3

    # Coupling strength w = Js * N is computed at FULL size, so that the mean
    # synaptic input is preserved under down-scaling (the standard convention;
    # cf. original PDmodel_mesopython.py and van Albada et al. 2015). Only the
    # population size N that enters the finite-size binomial noise is reduced.
    # (Earlier this used the down-scaled N for w, which wrongly shifted the mean
    # input and inflated the apparent "down-scaling rate distortion"; the bug
    # was caught by meso-Python vs meso-NEST disagreement at scaling=0.1.)
    w = Js * N[np.newaxis, :]
    N = np.round(N * scaling_factor).astype(int)

    params = {
        'mu': mu,
        'hazard_c': hazard_c,
        'hazard_Delta_u': DeltaV_vec,
        'vth': neuron_params['v_thresh'] - neuron_params['v_rest'],
        'vreset': neuron_params['v_reset'] - neuron_params['v_rest'],
        'taum_e': neuron_params['tau_m'] * 1e-3,
        'taum_i': neuron_params['tau_m'] * 1e-3,
        'delay': d_mean['E'] * 1e-3,
        'tref': neuron_params['tau_refrac'] * 1e-3,
        'taus_e': neuron_params['tau_syn_E'] * 1e-3,
        'taus_i': neuron_params['tau_syn_I'] * 1e-3,
        'M': M,
        'Me': Me,
        'N': N,
        'weights': w,
    }
    return params, N


def build_step_stimulus(params, T, dt_record, B=0.0, tstart=0.001,
                        duration=None):
    """Add an external step-current stimulus ('Iext') to params.

    B = 0.0 gives pure spontaneous activity (the benchmarking default).
    A non-zero B injects a thalamic-like step into L4 and L6 populations.
    """
    M_ = params['M']
    Nbin = round(T / dt_record)
    if duration is None:
        duration = T

    step = np.zeros(M_)
    # thalamic drive targets (E-first ordering): L4e=1, L4i=5, L6e=3, L6i=7
    step[1] = tha_drive_vec[2]
    step[5] = tha_drive_vec[3]
    step[3] = tha_drive_vec[6]
    step[7] = tha_drive_vec[7]
    if step.max() > 0:
        step = step / step.max() * B
    else:
        step = step * 0.0

    Iext = np.zeros((Nbin, M_), dtype=np.float32)
    k1 = int(tstart / dt_record)
    k2 = min(int((tstart + duration) / dt_record), Nbin)
    assert k1 <= k2
    for m in range(M_):
        Iext[k1:k2, m] = step[m]
    params['Iext'] = Iext
    return params
