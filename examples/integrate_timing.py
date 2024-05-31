#!/usr/bin/env python
# %% [markdown]
# # INTEGRATE timing


# %%
import integrate as ig
import numpy as np
import matplotlib.pyplot as plt
import time

# get name of CPU
import os
import socket
hostname = socket.gethostname()
# Get number of processors
Ncpu = os.cpu_count()
print("Hostname: %s" % hostname)
print("Number of processors: %d" % Ncpu)

# %% Choose the GEX file used for forward modeling. THis should be stored in the data file.
f_data_h5 = 'Fra20200930_202001001_1_AVG_export.h5'
file_gex ='fraastad_ttem.gex'
print("Using GEX file: %s" % file_gex)


# %% TIMING
# N_arr = logspace form 100 to 1000000 in ns stesp
N_arr = np.array([100,500,1000,5000,10000,50000,100000, 500000, 1000000])
#N_arr = np.array([100,1000,10000,50000])
Nproc_arr=[1, 2, 4, 8, 16,  32]
Nproc_arr=(1+np.arange(int(Ncpu**(0.5))))**2

n1 = len(N_arr)
n2 = len(Nproc_arr)

T_prior = np.zeros((n1,n2))*np.nan
T_forward = np.zeros((n1,n2))*np.nan
T_rejection = np.zeros((n1,n2))*np.nan
T_poststat = np.zeros((n1,n2))*np.nan

for j in np.arange(n2):
    Nproc = Nproc_arr[j]
    
    t_prior = []
    t_forward  = []
    t_rejection = []
    t_poststat = []

    for i in np.arange(len(N_arr)):
        N=N_arr[i]

        RHO_min = 1
        RHO_max = 800
        z_max = 50 
        useP = 1

        t0_prior = time.time()
        if useP ==1:
            ## Layered model    
            f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='chi2', NLAY_deg=5, z_max = z_max, rho_dist='log-uniform', RHO_min=RHO_min, RHO_max=RHO_max)
            #f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='uniform', z_max = z_max, NLAY_min=1, NLAY_max=3, rho_dist='log-uniform', RHO_min=RHO_min, RHO_max=RHO_max)
            #f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='uniform', z_max = z_max, NLAY_min=1, NLAY_max=8, rho_dist='log-uniform', RHO_min=RHO_min, RHO_max=RHO_max)
        else: 
            ## N layer model with increasing thickness
            f_prior_h5 = ig.prior_model_workbench(N=N, z_max = 30, nlayers=20, rho_min = RHO_min, rho_max = RHO_max)
        #t_prior.append(time.time()-t0_prior)
        T_prior[i,j] = time.time()-t0_prior

        if (Nproc<5 and N>9000)| (Nproc<7 and N>40000):
            pass
        else:   

            #ig.plot_prior_stats(f_prior_h5)
            #% A2. Compute prior DATA
            t0_forward = time.time()
            f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, Nproc=Nproc)
            T_forward[i,j]=time.time()-t0_forward

            #% READY FOR INVERSION
            N_use = 1000000
            t0_rejection = time.time()
            f_post_h5 = ig.integrate_rejection(f_prior_data_h5, f_data_h5, N_use = N_use, parallel=1, updatePostStat=False, showInfo=1, Nproc=Nproc)
            T_rejection[i,j]=time.time()-t0_rejection

            #% Compute some generic statistic of the posterior distribution (Mean, Median, Std)
            t0_poststat = time.time()
            ig.integrate_posterior_stats(f_post_h5)
            T_poststat[i,j]=time.time()-t0_poststat


# %% Save T_prior, N_arr, Nproc_arr in one file


file_out  = 'timing_%s-%d_Nproc%d_N%d' % (hostname,Ncpu,len(Nproc_arr), len(N_arr))
print(file_out)
np.savez(file_out, T_prior=T_prior, T_forward=T_forward, T_rejection=T_rejection, T_poststat=T_poststat, N_arr=N_arr, Nproc_arr=Nproc_arr)


#%%
ax, fig = plt.subplots(1,1, figsize=(8,8))
plt.loglog(N_arr, T_prior, 'k-*',label='Prior model')
plt.plot(N_arr, T_forward, 'r-*', label='Forward model')
plt.plot(N_arr, T_rejection, 'b-*', label='Rejection sampling')
plt.plot(N_arr, T_poststat, 'g-*', label='Posterior statistics')
plt.xlabel('Number of realizations')
plt.ylabel('Time [s]')
plt.legend()
plt.grid()
plt.savefig('%s_Narr' % file_out)
plt.show()


ax, fig = plt.subplots(1,1, figsize=(8,8))
plt.loglog(Nproc_arr, T_prior.T, 'k-*',label='Prior model')
plt.plot(Nproc_arr, T_forward.T, 'r-*', label='Forward model')
plt.plot(Nproc_arr, T_rejection.T, 'b-*', label='Rejection sampling')
plt.plot(Nproc_arr, T_poststat.T, 'g-*', label='Posterior statistics')
plt.xlabel('Number of processors')
plt.ylabel('Time [s]')
plt.legend()
plt.grid()
plt.savefig('%s_Nproc' % file_out)
plt.show()



# %%
dlw = 0.1
ax, fig = plt.subplots(2,2, figsize=(8,8))
plt.subplot(2,2,1)
for i in range(len(Nproc_arr)):
    plt.loglog(N_arr, T_prior[:,i], 'k-*',label='Np=%d' % Nproc_arr[i], linewidth=2+(2*(i*dlw)))
plt.xlabel('Number of realizations')
plt.ylabel('Time [s]')
plt.title('Prior')
plt.legend()
plt.grid()

plt.subplot(2,2,2)
for i in range(len(Nproc_arr)):
    plt.loglog(N_arr, T_forward[:,i], 'r-*',label='Np=%d' % Nproc_arr[i], linewidth=2+(2*(i*dlw)))
plt.xlabel('Number of realizations')
plt.ylabel('Time [s]')
plt.title('Forward')
plt.legend()
plt.grid()

plt.subplot(2,2,3)
for i in range(len(Nproc_arr)):
    plt.plot(N_arr, T_rejection[:,i], 'b-*',label='Np=%d' % Nproc_arr[i], linewidth=2+(2*(i*dlw)))
plt.xlabel('Number of realizations')
plt.ylabel('Time [s]')
plt.title('Rejection sampling')
plt.legend()
plt.grid()

plt.subplot(2,2,4)
for i in range(len(Nproc_arr)):
    plt.plot(N_arr, T_poststat[:,i], 'g-*',label='Np=%d' % Nproc_arr[i], linewidth=2+(2*(i*dlw)))
plt.xlabel('Number of realizations')
plt.ylabel('Time [s]')
plt.title('Posterior statistics')
plt.legend()
plt.grid()
ymin, ymax = plt.ylim()
plt.ylim(ymin*.9, ymax*1.1)

plt.tight_layout()
plt.savefig('%s_N_arr_sp' % file_out)
plt.show()



# %%
dlw = 0.4
ax, fig = plt.subplots(2,2, figsize=(8,8))
plt.subplot(2,2,1)
for i in range(len(N_arr)):
    plt.loglog(Nproc_arr, T_prior[i,:].T, 'k-*',label='N=%d' % N_arr[i], linewidth=1+(2*(i*dlw)))
plt.xlabel('Number of processors')
plt.ylabel('Time [s]')
plt.title('Prior')
plt.legend()
plt.grid()
ymin, ymax = plt.ylim()
plt.ylim(ymin*.9, ymax*1.1)

plt.subplot(2,2,2)
for i in range(len(N_arr)):
    plt.loglog(Nproc_arr, T_forward[i,:].T, 'r-*',label='N=%d' % N_arr[i], linewidth=1+(2*(i*dlw)))
plt.xlabel('Number of processors')
plt.ylabel('Time [m]')
plt.title('Forward')
plt.legend()
plt.grid()
ymin, ymax = plt.ylim()
plt.ylim(ymin*.9, ymax*1.1)

plt.subplot(2,2,3)
for i in range(len(N_arr)):
    plt.loglog(Nproc_arr, T_rejection[i,:].T, 'b-*',label='N=%d' % N_arr[i], linewidth=1+(2*(i*dlw)))
plt.xlabel('Number of processors')
plt.ylabel('Time [s]')
plt.title('Rejection sampling')
plt.legend()
plt.grid()
ymin, ymax = plt.ylim()
plt.ylim(ymin*.9, ymax*1.1)

plt.subplot(2,2,4)
for i in range(len(N_arr)):
    plt.semilogx(Nproc_arr, T_poststat[i,:].T, 'g-*',label='N=%d' % N_arr[i], linewidth=1+(2*(i*dlw)))
plt.xlabel('Number of processors')
plt.ylabel('Time [s]')
plt.title('Posterior statistics')
plt.legend()
plt.grid()
# get yaxis limits
ymin, ymax = plt.ylim()
plt.ylim(ymin*.9, ymax*1.1)

plt.tight_layout()

plt.savefig('%s_N_proc_sp' % file_out)
plt.show()


# %%


# %% PER SOUNDING
dlw = 0.4
ax, fig = plt.subplots(2,2, figsize=(8,8))
plt.subplot(2,2,1)
for i in range(len(N_arr)):
    plt.loglog(Nproc_arr, 1000*T_prior[i,:].T/N_arr[i], 'k-*',label='N=%d' % N_arr[i], linewidth=1+(2*(i*dlw)))
plt.xlabel('Number of processors')
plt.ylabel('Time/sonding [ms]')
plt.title('Prior')
plt.legend()
plt.grid()
ymin, ymax = plt.ylim()
plt.ylim(ymin*.9, ymax*1.1)

plt.subplot(2,2,2)
for i in range(len(N_arr)):
    plt.loglog(Nproc_arr, 1000*T_forward[i,:].T/N_arr[i], 'r-*',label='N=%d' % N_arr[i], linewidth=1+(2*(i*dlw)))
plt.xlabel('Number of processors')
plt.ylabel('Time/sonding [ms]')
plt.title('Forward')
plt.legend()
plt.grid()
ymin, ymax = plt.ylim()
plt.ylim(ymin*.9, ymax*1.1)

plt.subplot(2,2,3)
for i in range(len(N_arr)):
    plt.loglog(Nproc_arr, 1000*T_rejection[i,:].T/N_arr[i], 'b-*',label='N=%d' % N_arr[i], linewidth=1+(2*(i*dlw)))
plt.xlabel('Number of processors')
plt.ylabel('Time/sonding [ms]')
plt.title('Rejection sampling')
plt.legend()
plt.grid()
ymin, ymax = plt.ylim()
plt.ylim(ymin*.9, ymax*1.1)

plt.subplot(2,2,4)
for i in range(len(N_arr)):
    plt.semilogx(Nproc_arr, 1000*T_poststat[i,:].T/N_arr[i], 'g-*',label='N=%d' % N_arr[i], linewidth=1+(2*(i*dlw)))
plt.xlabel('Number of processors')
plt.ylabel('Time/sonding [ms]')
plt.title('Posterior statistics')
plt.legend()
plt.grid()
# get yaxis limits
ymin, ymax = plt.ylim()
plt.ylim(ymin*.9, ymax*1.1)

plt.tight_layout()

plt.savefig('%s_N_proc_sp_per_sounding' % file_out)
plt.show()


# %%
