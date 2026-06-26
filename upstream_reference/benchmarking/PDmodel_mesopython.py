# simulation single realization of Potjans-Diesmann model
# using Python implementation in sim4.py
# setting as in Fig.1 and 8 of Schwalger et al. PLoS Comput Biol (2017) but without adaptation
#
# copyright Tilo Schwalger, July 2024, tilo.schwalger@bccn-berlin.de

from sim4 import simulate
import matplotlib.pyplot as plt
import numpy as np
import time

#load parameters for potjans column
from potjans import *

with_numba = True
with_python = False # Pure Python simulation without Numba acceleration

scaling_factor = 1.0 # set to 1 for original network size

#simulation time
t0 = 0.0 #time for relaxation of initial transient
T = t0 + 10.0 # simulation time (seconds)

#set parameters of step stimulation
B=0.
tstart = t0 + 0.001
duration = T

M = 8       # number populations
Nrecord = 8 # number populations to be recorded

dt_record = 0.001  #bin size of population activity in seconds
dt = 0.0001        #simulation time step in seconds

#parameter values taken from corrected potjans_noadap_trajec3_mdopt.py
DeltaV = 5. * np.ones(M)
#              2/3e         4e           5e           6e           2/3i         4i           5i           6i   
mu = np.array([19.14942021, 30.80455243, 29.43680143, 34.93225564, 20.36247192, 28.06868188, 29.32954448, 32.08123423])


N = np.array([ N_full['L23']['E'], N_full['L4']['E'], N_full['L5']['E'], N_full['L6']['E'],\
               N_full['L23']['I'], N_full['L4']['I'], N_full['L5']['I'], N_full['L6']['I'],\
              ])

params = {'mu': mu,
          'hazard_c': 10.0,      #Hz
          'hazard_Delta_u': DeltaV, #mV
          'vth': neuron_params['v_thresh'] - neuron_params['v_rest'],    #mV
          'vreset': neuron_params['v_reset'] - neuron_params['v_rest'],
          'taum_e': neuron_params['tau_m']*1e-3,  # (s)
          'taum_i': neuron_params['tau_m']*1e-3,  # (s)
          'delay': d_mean['E'] * 1e-3,
          'tref': neuron_params['tau_refrac'] *1e-3,
          'taus_e': neuron_params['tau_syn_E'] * 1e-3, # (s)
          'taus_i': neuron_params['tau_syn_I'] * 1e-3, # (s)
          'M': M,
          'Me': 4,
          'N': N}



W = weight_matrix

# annealed average of quenched connectivity
W_av = W * conn_probs

# matrices W based on a different ordering of excitatory and inhibitory populations
 
#re-order columns
Js_tmp = np.zeros((M, M))
Js_tmp[:,0] = W_av[:,0] # from 2/3e
Js_tmp[:,1] = W_av[:,2] # from 4e
Js_tmp[:,2] = W_av[:,4] # from 5e
Js_tmp[:,3] = W_av[:,6] # from 6e
Js_tmp[:,4] = W_av[:,1] # from 2/3i
Js_tmp[:,5] = W_av[:,3] # from 4i
Js_tmp[:,6] = W_av[:,5] # from 5i
Js_tmp[:,7] = W_av[:,7] # from 6i

# re-order rows
Js=np.zeros((M, M))
Js[0,:]=Js_tmp[0,:] # to 2/3e
Js[1,:]=Js_tmp[2,:] # to 4e
Js[2,:]=Js_tmp[4,:] # to 4e
Js[3,:]=Js_tmp[6,:] # to 4e
Js[4,:]=Js_tmp[1,:] # to 4e
Js[5,:]=Js_tmp[3,:] # to 4e
Js[6,:]=Js_tmp[5,:] # to 4e
Js[7,:]=Js_tmp[7,:] # to 4e

# turn psc amplitudes into psp amplitudes
Js[:,:4] *= (neuron_params['tau_syn_E'] * 1e-3) / neuron_params['cm'] * 1e3
Js[:,4:] *= (neuron_params['tau_syn_I'] * 1e-3)/ neuron_params['cm'] * 1e3

#coupling strength w= Js * N
w = np.zeros((M, M))
for j in range(M):
    w[:,j] = Js[:,j] * N[j]
params['weights'] = w

#rescale network size as in microscopic simulation (../c_sim/)
N = np.round(N * scaling_factor).astype(int)  #decrease network size by scaling_factor
params['N'] = N

Nbin = round(T/dt_record)
tt= dt_record * np.arange(Nbin)

# creating step stimulus
step=np.zeros(M)
step[1] = tha_drive_vec[2]  # input to 4e
step[5] = tha_drive_vec[3]  # input to 4i
step[3] = tha_drive_vec[6]  # input to 6e
step[7] = tha_drive_vec[7]  # input to 6i

step = step / step.max() * B
Iext=np.zeros((Nbin,M), dtype=np.float32)   # external current R*Iext in (mV)
kstep1=int(tstart/dt_record)
kstep2=min(int((tstart + duration) / dt_record), Nbin)
assert kstep1 <= kstep2
for m in range(M):
    Iext[kstep1:kstep2,m] = step[m]
params['Iext'] = Iext

seed=12345          #seed for finite-size noise

if with_numba:
    start_time=time.time()
    rate, A = simulate(T, dt, dt_record, params, Nrecord, seed)
    end_time=time.time()
    print("execution time MESO NUMBA: %s seconds"%(end_time-start_time))

if with_python:
    start_time=time.time()
    rate, A = simulate(T, dt, dt_record, params, Nrecord, seed, with_numba=False)
    end_time=time.time()
    print("execution time MESO PYTHON: %s seconds"%(end_time-start_time))

