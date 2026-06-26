# simulation of mesoscopic PD model like in PDmodel_mesopython.py
# use parameters set generated in save_params_for_julia.py (needs to be run first!)
# 2024 Tilo Schwalger


using PyPlot, MAT

include("sim5.jl")

scaling_factor = 0.1 # set to 1 for original network size

#simulation time
t0 = 1.0 #time for relaxation of initial transient
T = t0 + 0.15 # simulation time (seconds)

# external stimulus
B = 15.0 # stimulus strength
tstart = t0 + 0.06
duration = 0.03

M = 8
Nrecord = 8 # number populations to be recorded

dt_record = 0.001  #bin size of population activity in s
dt = 0.0001        #simulation time step in s

Nbin = round(Int, T/dt_record)
tt= dt_record * collect(1:Nbin)

seed=30          #seed for finite-size noise

file = matopen("params_PDmodel.mat")
params = read(file)
close(file)

#rescale network size as in microscopic simulation (../c_sim/)
params["N"] = round.(Int, params["N"] * scaling_factor)  #decrease network size by scaling_factor

# Julia sim function requires weights per spike (PSP)
J=zeros(M,M)
for i=1:M
    for j=1:M
        J[i,j] = params["weights"][i,j] / params["N"][j]
    end
end
params["weights"] = J
    
step = params["steps_normalized"] * B
Iext=zeros(Float32,(Nbin,M))   # external current R*Iext in (mV)
kstep1=round(Int, tstart/dt_record)
kstep2=min(round(Int, (tstart + duration) / dt_record), Nbin)
for m in 1:M
    Iext[kstep1:kstep2,m] .= step[m]
end
params["Iext"] = Iext

@time begin
    rate, A = sim(T, dt, dt_record, params, Nrecord, seed)
end

i0 = round(Int,t0/dt_record) + 1
figure(1)
clf()

# plot excitatory populations    
for i=1:4
    subplot(4,2,2*i-1)
    plot(tt[i0:end],rate[i,i0:end])
    ylabel("rate [Hz]")
end
xlabel("time [s]")

# plot excitatory populations    
for i=5:8
    subplot(4,2,2*(i-4))
    plot(tt[i0:end],rate[i,i0:end])
    ylabel("rate [Hz]")
end
xlabel("time [s]")
show()
