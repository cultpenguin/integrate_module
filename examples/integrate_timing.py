#!/usr/bin/env python
# %% [markdown]
# # INTEGRATE timing
# This notebook compares CPU time using for both forward mdoeling and inversion 
#
# # Force use of single CPU with numpy
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
#


# %% Imports
try:
    # Check if the code is running in an IPython kernel (which includes Jupyter notebooks)
    get_ipython()
    # If the above line doesn't raise an error, it means we are in a Jupyter environment
    # Execute the magic commands using IPython's run_line_magic function
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except:
    # If get_ipython() raises an error, we are not in a Jupyter environment
    # # # # # # # # # # #%load_ext autoreload
    # # # # # # # # # # #%autoreload 2
    pass

# %%
import integrate as ig
# check if parallel computations can be performed
parallel = ig.use_parallel(showInfo=1)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import loglog
import time
import h5py
# get name of CPU
import os
import socket

# Get hostanme and number of processors
hostname = socket.gethostname()
Ncpu_total = os.cpu_count()
print("Hostname: %s" % hostname)
print("Number of processors: %d" % Ncpu_total)


# %% [markdown]
# ## Get the default data set

# %% SELECT THE CASE TO CONSIDER AND DOWNLOAD THE DATA
files = ig.get_case_data()
f_data_h5 = files[0]
file_gex= ig.get_gex_file_from_data(f_data_h5)

print("Using data file: %s" % f_data_h5)
print("Using GEX file: %s" % file_gex)

with h5py.File(f_data_h5, 'r') as f:
    nobs = f['D1/d_obs'].shape[0]


# %% [markdown]
# ## Setup the timing test


# %% TIMING
#### Set the size of the data sets to test
N_arr = np.array([100,500,1000,5000,10000,50000,100000, 500000, 1000000])

# Set the number of cores to test
Nproc_arr=2**(np.double(np.arange(1+int(np.log2(Ncpu_total)))))

useAltTest=True
if useAltTest:
    N_arr = np.array([100,500,1000,5000,10000])
    #N_arr = np.array([100,1000,2000,5000,10000,50000,100000,500000])
    
    skip_proc = 0
    Nproc_arr=2**(np.double(skip_proc+np.arange(1+int(np.log2(Ncpu_total)))));
    Nproc_arr=np.int8(np.ceil(np.linspace(1,Ncpu_total,5)))
    Nproc_arr=1+np.arange(Ncpu_total)
    #Nproc_arr ois 1:4:Ncpu_total
    Nproc_arr=np.arange(2, 9, 2)
    
    ##Nproc_arr=np.arange(2, Ncpu_total/2+1, 2)
    Nproc_arr=np.append(Nproc_arr,np.arange(10,1+Ncpu_total/2,2))
    Nproc_arr=np.append(Nproc_arr,Ncpu_total)
    Nproc_arr=np.array([1,4,8])

    N_arr = np.array([1000,5000])
    Nproc_arr=np.array([8])

n1 = len(N_arr)
n2 = len(Nproc_arr)


print("Testing on %d data sets of sizes" % n1)
print(N_arr)
print("Testing on %d sets of cores" % n2)
print(Nproc_arr)


file_out  = 'timing_%s-%d_Nproc%d_N%d' % (hostname,Ncpu_total,len(Nproc_arr), len(N_arr))
print("Writing results to %s " % file_out)

# %% [markdown]
# ## Run INTEGRATE workflow using different data sizes and number of CPUS

# %%

loadFromFile=False
if loadFromFile:
    file_out = 'timing_3990X-64_Nproc7_N9'  
    file_out = 'timing_3990X-64_Nproc35_N8' 
    
    #file_out = 'timing_d52534-32_Nproc13_N9'
    #file_out = 'timing_d52534-32_Nproc6_N9'  
    file_out = 'timing_d52534-32_Nproc17_N9'
    file_out = 'timing_d52534-32_Nproc13_N9-issue16'
    
    #file_out = 'timing_Z13-16_Nproc8_N7'
    
    data = np.load('%s.npz' % file_out)
    T_prior = data['T_prior']
    T_forward = data['T_forward']
    T_rejection = data['T_rejection']
    T_poststat = data['T_poststat']

    T_total = T_prior + T_forward + T_rejection + T_poststat

    N_arr = data['N_arr']
    Nproc_arr = data['Nproc_arr']
    try:
        nobs=data['nobs']
    except:
        nobs=11693

else:

    showInfo = 0

    T_prior = np.zeros((n1,n2))*np.nan
    T_forward = np.zeros((n1,n2))*np.nan
    T_rejection = np.zeros((n1,n2))*np.nan
    T_poststat = np.zeros((n1,n2))*np.nan

    testRejection = True
    testPostStat = True  
                
    for j in np.arange(n2):
        Ncpu = int(Nproc_arr[j])
        
        t_prior = []
        t_forward  = []
        t_rejection = []
        t_poststat = []

        for i in np.arange(len(N_arr)):
            N=int(N_arr[i])
            Ncpu_min = int(np.floor(2**(np.log10(N)-4)))
            
            #print('=====================================================')
            print('N=%d, Ncpu=%d, Ncpu_min=%d'%(N,Ncpu,Ncpu_min))

            RHO_min = 1
            RHO_max = 800
            z_max = 50 
            useP = 1
            
            if (Ncpu>=Ncpu_min):
                    
                t0_prior = time.time()
                if useP ==1:
                    ## Layered model    
                    f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='chi2', NLAY_deg=5, z_max = z_max, RHO_dist='log-uniform', RHO_min=RHO_min, RHO_max=RHO_max, showInfo=showInfo)
                    #f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='uniform', z_max = z_max, NLAY_min=1, NLAY_max=3, rho_dist='log-uniform', RHO_min=RHO_min, RHO_max=RHO_max)
                    #f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='uniform', z_max = z_max, NLAY_min=1, NLAY_max=8, rho_dist='log-uniform', RHO_min=RHO_min, RHO_max=RHO_max)
                else: 
                    ## N layer model with increasing thickness
                    f_prior_h5 = ig.prior_model_workbench(N=N, z_max = 30, nlayers=20, rho_min = RHO_min, rho_max = RHO_max, showInfo=showInfo)
                #t_prior.append(time.time()-t0_prior)
                T_prior[i,j] = time.time()-t0_prior

            
                #ig.plot_prior_stats(f_prior_h5)
                #% A2. Compute prior DATA
                t0_forward = time.time()
                f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, Ncpu=Ncpu, showInfo=showInfo)
                T_forward[i,j]=time.time()-t0_forward

                #% READY FOR INVERSION
                N_use = 1000000
                t0_rejection = time.time()
                if testRejection:
                    f_post_h5 = ig.integrate_rejection(f_prior_data_h5, f_data_h5, N_use = N_use, parallel=1, updatePostStat=False,  Ncpu=Ncpu, showInfo=showInfo)
                T_rejection[i,j]=time.time()-t0_rejection

                #% Compute some generic statistic of the posterior distribution (Mean, Median, Std)
                t0_poststat = time.time()
                if testPostStat and testRejection:
                    ig.integrate_posterior_stats(f_post_h5,showInfo=showInfo)
                    T_poststat[i,j]=time.time()-t0_poststat

            T_total = T_prior + T_forward + T_rejection + T_poststat
            np.savez(file_out, T_total=T_total, T_prior=T_prior, T_forward=T_forward, T_rejection=T_rejection, T_poststat=T_poststat, N_arr=N_arr, Nproc_arr=Nproc_arr, nobs=nobs)



# %%

# %% Plot
# LSQ, Assumed time, in seconds, for least squares inversion of a single sounding
t_lsq = 2.0
# SAMPLING, Assumed time, in seconds, for an McMC inversion of a single sounding
t_mcmc = 10.0*60.0 

total_lsq = np.array([nobs*t_lsq, nobs*t_lsq/Nproc_arr[-1]])
total_mcmc = np.array([nobs*t_mcmc, nobs*t_mcmc/Nproc_arr[-1]])


# %% loglog(T_total.T)
plt.figure(figsize=(6,6))    
plt.loglog(Nproc_arr, T_total.T, 'o-',  label=N_arr)
plt.ylabel(r'Total time - $[s]$')
plt.xlabel('Number of processors')
plt.grid()
total_lsq = np.array([nobs*t_lsq, nobs*t_lsq/Nproc_arr[-1]])
plt.plot([Nproc_arr[0], Nproc_arr[-1]], total_lsq, 'k--', label='LSQ')
plt.plot([Nproc_arr[0], Nproc_arr[-1]], total_mcmc, 'r--', label='MCMC')
plt.legend(loc='upper right')
plt.xticks(ticks=Nproc_arr, labels=[str(int(x)) for x in Nproc_arr])
plt.ylim(1,1e+8)
plt.tight_layout()
plt.savefig('%s_total_sec' % file_out)

plt.figure(figsize=(6,6)) 
plt.loglog(N_arr, T_total, 'o-', label=[f'{int(x)}' for x in Nproc_arr])
plt.ylabel(r'Total time - $[s]$')
plt.xlabel('N-prior')
plt.grid()
plt.tight_layout()
plt.plot([N_arr[0], N_arr[-1]], [nobs*t_lsq, nobs*t_lsq], 'k--', label='LSQ')
plt.plot([N_arr[0], N_arr[-1]], [nobs*t_mcmc, nobs*t_mcmc], 'r--', label='MCMC')
plt.legend(loc='upper left')
#plt.xticks(ticks=N_arr, labels=[str(int(x)) for x in Nproc_arr])
plt.ylim(1,1e+8)
plt.savefig('%s_total_sec' % file_out)



# %% Plot timing results for forward modeling - GAAEM

# Average timer per sounding 
T_forward_sounding = T_forward/N_arr[:,np.newaxis]
T_forward_sounding_per_sec = N_arr[:,np.newaxis]/T_forward
T_forward_sounding_per_sec_per_cpu = T_forward_sounding_per_sec/Nproc_arr[np.newaxis,:]
T_forward_sounding_speedup = T_forward_sounding_per_sec/T_forward_sounding_per_sec[0,0]


plt.figure(figsize=(6,6))    
#plt.plot(Nproc_arr, T_forward_sounding.T, 'o-')
#plt.plot(Nproc_arr, T_forward.T, 'o-')
plt.loglog(Nproc_arr, T_forward.T, 'o-', label='A')
# plot line 
plt.ylabel(r'TIME - $[s]$')
plt.xlabel('Number of processors')
plt.title('Forward calculation')
plt.grid()
plt.legend(N_arr, loc='upper left')
plt.ylim(1e-1, 1e+4)
plt.tight_layout()
plt.savefig('%s_forward_sec' % file_out)

plt.figure(figsize=(6,6))    
plt.plot(Nproc_arr, T_forward_sounding_per_sec.T, 'o-')
# plot line 
plt.ylabel(r'Forward computations per second - $[s^{-1}]$')
plt.xlabel('Number of processors')
plt.title('Forward calculation')
plt.grid()
plt.legend(N_arr)
plt.tight_layout()
plt.savefig('%s_forward_sounding_per_sec' % file_out)

plt.figure(figsize=(6,6))    
plt.plot(Nproc_arr, T_forward_sounding_per_sec_per_cpu.T, 'o-')
plt.ylabel('Forward computations per second per cpu')
plt.xlabel('Number of processors')
plt.title('Forward calculation')
plt.grid()
plt.legend(N_arr)
plt.savefig('%s_forward_sounding_per_sec_per_cpu' % file_out)

plt.figure(figsize=(6,6))    
plt.plot(Nproc_arr, T_forward_sounding_speedup.T, 'o-')
# plot a line from 0,0 tp Nproc_arr[-1], Nproc_arr[-1]
plt.plot([0, Nproc_arr[-1]], [0, Nproc_arr[-1]], 'k--')
# set xlim to 1, Nproc_arr[-1]
plt.xlim(.8, Nproc_arr[-1])
plt.ylim(.8, Nproc_arr[-1])
plt.ylabel('gatdaem - speedup compared to 1 processor')
plt.xlabel('Number of processors')
plt.grid()
plt.legend(N_arr)
plt.savefig('%s_forward_speedup' % file_out)



# %% STATS FOR REJECTION SAMPLING
# Average timer per sounding
T_rejection_sounding = T_rejection/N_arr[:,np.newaxis]
T_rejection_sounding_per_sec = N_arr[:,np.newaxis]/T_rejection
T_rejection_sounding_per_sec_per_cpu = T_rejection_sounding_per_sec/Nproc_arr[np.newaxis,:]
T_rejection_sounding_speedup = T_rejection_sounding_per_sec/T_rejection_sounding_per_sec[0,0]
T_rejection_sounding_speedup = T_rejection_sounding_per_sec*0

T_rejection_per_data = nobs/T_rejection

for i in range(len(N_arr)):
    # find index of first valiue in T_rejection_sounding_per_sec[i,:] that is not nan
    idx = np.where(~np.isnan(T_rejection_sounding_per_sec[i,:]))[0][0]
    T_rejection_sounding_speedup[i,:] = T_rejection_sounding_per_sec[i,:]/(T_rejection_sounding_per_sec[i,idx]/Nproc_arr[idx]) 

plt.figure(figsize=(6,6))
plt.loglog(Nproc_arr, T_rejection_per_data.T, 'o-', label=N_arr)
plt.plot([Nproc_arr[0], Nproc_arr[-1]], [1./t_lsq, 1./t_lsq], 'k--', label='LSQ')
plt.plot([Nproc_arr[0], Nproc_arr[-1]], [1./t_mcmc, 1./t_mcmc], 'r--', label='MCMC')
plt.ylabel('Rejection sampling - number of soundings per second - $s^{-1}$')
plt.xlabel('Number of processors')
plt.grid()
plt.legend(loc='lower left')
plt.tight_layout()
plt.ylim(1e-3, 1e+5)
plt.savefig('%s_rejection_sound_per_sec' % file_out)

plt.figure(figsize=(6,6))
plt.semilogy(Nproc_arr, 1./T_rejection_per_data.T, 'o-', label=N_arr)
#plt.plot(Nproc_arr, 1./T_rejection_per_data.T, 'o-', label=N_arr)
plt.plot([Nproc_arr[0], Nproc_arr[-1]], [t_lsq, t_lsq], 'k--', label='LSQ')
plt.plot([Nproc_arr[0], Nproc_arr[-1]], [t_mcmc, t_mcmc], 'r--', label='MCMC')
plt.ylabel('Rejection sampling - seconds per sounding - $s$')
plt.xlabel('Number of processors')
plt.grid()
plt.legend(loc='upper right')
plt.tight_layout()
plt.ylim(1e-5, 1e+3)
plt.savefig('%s_rejection_sec_per_sound' % file_out)


# %%
plt.figure(figsize=(6,6))
#plt.loglog(Nproc_arr, T_rejection.T, 'o-')
plt.semilogy(Nproc_arr, T_rejection.T, 'o-')
plt.ylabel('Rejection sampling - total time - $[s]$')
plt.xlabel('Number of processors')
plt.grid()
plt.legend(N_arr)
plt.tight_layout()
plt.ylim(1e-1, 2e+3)
plt.savefig('%s_rejection_time' % file_out)

# %%
plt.figure(figsize=(6,6))
plt.plot(Nproc_arr, T_rejection_sounding_speedup.T, 'o-')
# plot a line from 0,0 tp Nproc_arr[-1], Nproc_arr[-1]
plt.plot([0, Nproc_arr[-1]], [0, Nproc_arr[-1]], 'k--')
# set xlim to 1, Nproc_arr[-1]
plt.xlim(.8, Nproc_arr[-1])
plt.ylim(.8, Nproc_arr[-1])
plt.ylabel('Rejection sampling - speedup compared to 1 processor')
plt.xlabel('Number of processors')
plt.grid()
plt.legend(N_arr)
plt.savefig('%s_rejection_speedup' % file_out)


# %% STATS FOR POSTERIOR STATISTICS
# Average timer per sounding
T_poststat_sounding = T_poststat/N_arr[:,np.newaxis]
T_poststat_sounding_per_sec = N_arr[:,np.newaxis]/T_poststat
T_poststat_sounding_per_sec_per_cpu = T_poststat_sounding_per_sec/Nproc_arr[np.newaxis,:]
T_poststat_sounding_speedup = T_poststat_sounding_per_sec/T_poststat_sounding_per_sec[0,0]

plt.figure(figsize=(6,6))
plt.plot(Nproc_arr, T_poststat_sounding_per_sec.T, 'o-')
plt.ylabel('Soundings per second - $[s^{-1}]$')
plt.xlabel('Number of processors')
plt.grid()
plt.legend(N_arr)
plt.tight_layout()
plt.savefig('%s_poststat_sounding_per_sec' % file_out)

plt.figure(figsize=(6,6))
plt.plot(Nproc_arr, T_poststat_sounding_speedup.T, 'o-')
# plot a line from 0,0 tp Nproc_arr[-1], Nproc_arr[-1]
plt.plot([0, Nproc_arr[-1]], [0, Nproc_arr[-1]], 'k--')
# set xlim to 1, Nproc_arr[-1]
plt.xlim(.8, Nproc_arr[-1])
plt.ylim(.8, Nproc_arr[-1])
plt.ylabel('Posterior statistics - speedup compared to 1 processor')
plt.xlabel('Number of processors')
plt.grid()
plt.legend(N_arr)
plt.savefig('%s_poststat_speedup' % file_out)

# %%

# %%

# %%
i_proc = len(Nproc_arr)-1
T_total[:,i_proc]
T_forward[:,i_proc]

# %%
T = [T_prior[:,i_proc],T_forward[:,i_proc],T_rejection[:,i_proc],T_poststat[:,i_proc]]
T=np.array(T)
cumT=np.cumsum(T, axis=0)
plt.semilogx(N_arr,cumT.T,'*')
plt.plot(N_arr,T_total[:,i_proc],'k--')
plt.xlabel('$N_{lookup}$')
plt.ylabel('Time [$s$]')


# %%
N_arr


# %%
cumT

# %%
cumT

# %%
plt.plot(T.cumsum())

# %%
