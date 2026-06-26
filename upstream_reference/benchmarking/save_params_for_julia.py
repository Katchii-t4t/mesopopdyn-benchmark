# save parameter dictionary of Potjans-Diesmann model for use in Julia
# corresponds to parameters in PDmodel_mesopython.py
# copyright Tilo Schwalger, September 2024, tilo.schwalger@bccn-berlin.de

import numpy as np
from scipy.io import savemat

#load parameters for potjans column
from potjans import *

M = 8       # number populations

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

# creating step stimulus
step=np.zeros(M)
step[1] = tha_drive_vec[2]  # input to 4e
step[5] = tha_drive_vec[3]  # input to 4i
step[3] = tha_drive_vec[6]  # input to 6e
step[7] = tha_drive_vec[7]  # input to 6i

params["steps_normalized"] = step / step.max()
savemat('params_PDmodel.mat', params)





