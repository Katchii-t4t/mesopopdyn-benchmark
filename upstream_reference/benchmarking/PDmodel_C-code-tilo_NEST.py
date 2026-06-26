# simulate single realization of potjans column
# using custom C library and/or NEST simulator
# setting like Fig.1 and Fig. 8 of Schwalger et al. PLoS Comput Biol (2017) but without adaptation
#
# July 2024, Tilo Schwalger, tilo.schwalger@bccn-berlin.de


import matplotlib.pyplot as plt
import sys
from numpy.fft import fft

path = '../mesopopdyngif_C_v18_Py3'
sys.path.append(path)
import multipop18 as mp
import multiprocessing as multproc
import time
import os

with_nest = True
with_C = True
with_micro = True
meanfield_connectivity = True
scaling_factor = 0.1 # set to 1 for original network size 
nproc = 1 # number of local threads for NEST simulation (microscopic NEST sim with nproc=1 most likely fails for original network size, scaling_factor=1. To simulate with NEST in this case, increase nproc.)

#load parameters for potjans column
from potjans import *

#simulation time
t0 = 0.0 #time for relaxation of initial transient
tend = t0 + 10.0

#set parameters of step stimulation
B = 0.0
tstart = t0 + 0.001
duration= tend -0.001

dt=0.0001
dtpop=0.0001
dtbin=0.001
dtbinpop=dtbin
K = len(conn_probs)

#parameter values taken from corrected potjans_noadap_trajec3_mdopt.py
DeltaV = 5. * np.ones(K)
mu = np.array([ 19.14942021,  20.36247192,  30.80455243,  28.06868188, 29.43680143,  29.32954448,  34.93225564,  32.08123423])
rates = np.array([ 0.9736411 ,  2.86068612,  4.67324687,  5.64983048,  8.1409982 , 9.01281818,  0.98813263,  7.53034027])

#adaptation set to zero
tau_theta = [[0.1],[0.1]]  # irrelevant values, set to positive number as required for NEST
tau_theta = tau_theta + tau_theta + tau_theta + tau_theta
J_theta =  [[0.],[0.]] #
J_theta = np.array(J_theta + J_theta + J_theta + J_theta)


c = np.ones(K, dtype=float) * 10.
Vreset = [neuron_params['v_reset'] - neuron_params['v_rest'] for k in range(K)]
Vth = [neuron_params['v_thresh'] - neuron_params['v_rest'] for k in range(K)]
delay = d_mean['E'] * 1e-3
t_ref = neuron_params['tau_refrac'] *1e-3
N = np.array([ N_full['L23']['E'], N_full['L23']['I'],\
               N_full['L4']['E'], N_full['L4']['I'],\
               N_full['L5']['E'], N_full['L5']['I'],\
               N_full['L6']['E'], N_full['L6']['I'],\
              ])


W = weight_matrix
# turn psc amplitudes into psp amplitudes
Js = W.copy()   
Js[:,0] *= (neuron_params['tau_syn_E'] * 1e-3) / neuron_params['cm'] * 1e3
Js[:,2] *= (neuron_params['tau_syn_E'] * 1e-3) / neuron_params['cm'] * 1e3  
Js[:,4] *= (neuron_params['tau_syn_E'] * 1e-3) / neuron_params['cm'] * 1e3
Js[:,6] *= (neuron_params['tau_syn_E'] * 1e-3) / neuron_params['cm'] * 1e3
Js[:,1] *= (neuron_params['tau_syn_I'] * 1e-3)/ neuron_params['cm'] * 1e3
Js[:,3] *= (neuron_params['tau_syn_I'] * 1e-3)  / neuron_params['cm'] * 1e3
Js[:,5] *= (neuron_params['tau_syn_I'] * 1e-3) / neuron_params['cm'] * 1e3
Js[:,7] *= (neuron_params['tau_syn_I'] * 1e-3) / neuron_params['cm'] * 1e3

N = np.round(N * scaling_factor).astype(int)  #decrease network size by factor scaling_factor
Js /= scaling_factor

taus1_ = [neuron_params['tau_syn_E'] * 1e-3, neuron_params['tau_syn_I'] * 1e-3]
taus1_ = taus1_ + taus1_ + taus1_
taus1 = [taus1_ for k in range(K)]

pconn = np.array(conn_probs)
if meanfield_connectivity:
    Js *= pconn
    pconn=np.ones((K,K))

mode='glif'
taum = np.array([neuron_params['tau_m']*1e-3 for k in range(K)])

i0=int(t0/dtbin)
step_ = np.hstack(( np.reshape( tha_drive_vec, (8,1) ), np.zeros((8,1)) ))
step_ = step_ / step_.max() * B
tstep_=np.array([[tstart,tstart+duration] for k in range(K)])

# #repeated step stimulation used in Nest simulation
# tstep=np.hstack([tstep_ + n * tend for n in range(Ntrials)])
# step=np.hstack([step_ for n in range(Ntrials)])

step=step_
tstep=tstep_


#######################################################################


seed=123


# --------------------------------------------------------------------------------
# microscopic simulation
# --------------------------------------------------------------------------------

if with_C and with_micro:
    print('Build microscopic network with C library ...')
    p1 = mp.MultiPop(dt=dt, N=N, rho_0= c, tau_m = taum, tau_sfa=tau_theta, J_syn=Js, taus1=taus1,delay=np.ones((K,K))*delay, t_ref= np.ones(K)*t_ref, V_reset= np.array(Vreset), J_a=J_theta, pconn=pconn, mu= np.array(mu), delta_u= DeltaV*np.ones(K), V_th= Vth*np.ones(K),sigma=np.zeros(K), mode=mode)
    p1.dt_rec=dtbin
    p1.build_network_tilo_neurons()
    print('Simulate microscopic network with C library ...')
    start_time = time.time()
    p1.simulate(tend,step=step,tstep=tstep, seed=seed)
    end_time = time.time()
    print("execution time MICRO C: %s seconds"%(end_time-start_time))
    print("------------------------------------------------------------------------")

if with_nest and with_micro:
    print('Build microscopic network with NEST ...')
    p2 = mp.MultiPop(dt=dt, N=N, rho_0= c, tau_m = taum, tau_sfa=tau_theta, J_syn=Js, taus1=taus1,delay=np.ones((K,K))*delay, t_ref= np.ones(K)*t_ref, V_reset= np.array(Vreset), J_a=J_theta, pconn=pconn, mu= np.array(mu), delta_u= DeltaV*np.ones(K), V_th= Vth*np.ones(K),sigma=np.zeros(K), mode=mode)
    p2.dt_rec=dtbin
    p2.local_num_threads = nproc
    p2.build_network_neurons()
    print('Simulate microscopic network with NEST ...')
    start_time = time.time()
    p2.simulate(tend,step=step,tstep=tstep, seed=seed)
    end_time = time.time()
    print("execution time MICRO NEST: %s seconds"%(end_time-start_time))
    print("------------------------------------------------------------------------")

# --------------------------------------------------------------------------------
# mesoscopic simulation
# --------------------------------------------------------------------------------

if with_C:
    print('Simulate mesoscopic populations with C library ...')
    p3 = mp.MultiPop(dt=dtpop, N=N, rho_0= c, tau_m = taum, tau_sfa=tau_theta, J_syn=Js, taus1=taus1, delay=np.ones((K,K))*delay, t_ref= np.ones(K)*t_ref, V_reset= np.array(Vreset), J_a=J_theta, pconn=pconn, mu= np.array(mu), delta_u= DeltaV*np.ones(K), V_th= Vth*np.ones(K),sigma=np.zeros(K), mode='glif')
    p3.dt_rec=dtbinpop
    p3.build_network_tilo_populations()
    start_time = time.time()
    p3.simulate(tend,step=step,tstep=tstep, seed=seed)
    end_time = time.time()
    print("execution time MESO C: %s seconds"%(end_time-start_time))
    print("------------------------------------------------------------------------")

if with_nest:
    print('Simulate mesoscopic populations with NEST ...')
    p4 = mp.MultiPop(dt=dtpop, N=N, rho_0= c, tau_m = taum, tau_sfa=tau_theta, J_syn=Js, taus1=taus1, delay=np.ones((K,K))*delay, t_ref= np.ones(K)*t_ref, V_reset= np.array(Vreset), J_a=J_theta, pconn=pconn, mu= np.array(mu), delta_u= DeltaV*np.ones(K), V_th= Vth*np.ones(K),sigma=np.zeros(K), mode='glif')
    p4.dt_rec=dtbinpop
    p4.build_network_populations()
    start_time = time.time()
    p4.simulate(tend,step=step,tstep=tstep, seed=seed)    
    end_time = time.time()
    print("execution time MESO NEST: %s seconds"%(end_time-start_time))
    print("------------------------------------------------------------------------")


if with_C and with_micro:
    plt.figure(10)
    plt.clf()
    for k in range(K):
        ax=plt.subplot(4,2,k+1)
        plt.plot(p1.sim_t[i0:]-p1.sim_t[i0], p1.sim_A[i0:,k],color='0.7',label=r'$A_N(t)$')
        if k%2==0:
            plt.ylabel("rate [Hz]")
    plt.subplot(4,2,7)
    plt.xlabel("time [s]")
    plt.subplot(4,2,8)
    plt.xlabel("time [s]")
    plt.subplots_adjust(top=0.995, bottom=0.1,left=0.12,right=0.97,hspace=0.32,wspace=0.26)

if with_nest and with_micro:
    plt.figure(11)
    plt.clf()
    for k in range(K):
        ax=plt.subplot(4,2,k+1)
        plt.plot(p2.sim_t[i0:]-p2.sim_t[i0], p2.sim_A[i0:,k],color='0.7',label=r'$A_N(t)$')
        if k%2==0:
            plt.ylabel("rate [Hz]")
    plt.subplot(4,2,7)
    plt.xlabel("time [s]")
    plt.subplot(4,2,8)
    plt.xlabel("time [s]")
    plt.subplots_adjust(top=0.995, bottom=0.1,left=0.12,right=0.97,hspace=0.32,wspace=0.26)

if with_C:
    plt.figure(12)
    plt.clf()
    for k in range(K):
        ax=plt.subplot(4,2,k+1)
        plt.plot(p3.sim_t[i0:]-p3.sim_t[i0], p3.sim_A[i0:,k],color='0.7',label=r'$A_N(t)$')
        plt.plot(p3.sim_t[i0:]-p3.sim_t[i0], p3.sim_a[i0:,k],label=r'$\bar A(t)$')
        if k%2==0:
            plt.ylabel("rate [Hz]")
    plt.subplot(4,2,7)
    plt.xlabel("time [s]")
    plt.subplot(4,2,8)
    plt.xlabel("time [s]")
    plt.subplots_adjust(top=0.995, bottom=0.1,left=0.12,right=0.97,hspace=0.32,wspace=0.26)

if with_nest:
    plt.figure(13)
    plt.clf()
    for k in range(K):
        ax=plt.subplot(4,2,k+1)
        plt.plot(p4.sim_t[i0:]-p4.sim_t[i0], p4.sim_A[i0:,k],color='0.7',label=r'$A_N(t)$')
        plt.plot(p4.sim_t[i0:]-p4.sim_t[i0], p4.sim_a[i0:,k],label=r'$\bar A(t)$')
        if k%2==0:
            plt.ylabel("rate [Hz]")
    plt.subplot(4,2,7)
    plt.xlabel("time [s]")
    plt.subplot(4,2,8)
    plt.xlabel("time [s]")
    plt.subplots_adjust(top=0.995, bottom=0.1,left=0.12,right=0.97,hspace=0.32,wspace=0.26)

plt.subplots_adjust(top=0.995, bottom=0.1,left=0.12,right=0.97,hspace=0.32,wspace=0.26)
plt.show()




