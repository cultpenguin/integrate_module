#!/usr/bin/env python
# %% [markdown]
# # INTEGRATE timing


# %%
import integrate as ig
import numpy as np
import matplotlib.pyplot as plt
import time

# %% Choose the GEX file used for forward modeling. THis should be stored in the data file.
f_data_h5 = 'Fra20200930_202001001_1_AVG_export.h5'
file_gex ='fraastad_ttem.gex'
print("Using GEX file: %s" % file_gex)


# %% TIMING

N_arr = np.array([100,200,300])
Nproc_arr=[8, 16]

n1 = len(N_arr)
n2 = len(Nproc_arr)

T_prior = np.zeros((n1,n2))
T_forward = np.zeros((n1,n2))
T_rejection = np.zeros((n1,n2))
T_poststat = np.zeros((n1,n2))

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
        t_prior.append(time.time()-t0_prior)

        #ig.plot_prior_stats(f_prior_h5)
        #% A2. Compute prior DATA
        t0_forward = time.time()
        f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, Nproc=Nproc)
        t_forward.append(time.time()-t0_forward)

        #% READY FOR INVERSION
        N_use = 1000000
        t0_rejection = time.time()
        f_post_h5 = ig.integrate_rejection(f_prior_data_h5, f_data_h5, N_use = N_use, parallel=1, updatePostStat=False, showInfo=1, Nproc=Nproc)
        t_rejection.append(time.time()-t0_rejection)

        #% Compute some generic statistic of the posterior distribution (Mean, Median, Std)
        t0_poststat = time.time()
        ig.integrate_posterior_stats(f_post_h5)
        t_poststat.append(time.time()-t0_poststat)

    T_prior[:,j]=t_prior
    T_forward[:,j]=t_forward
    T_rejection[:,j]=t_rejection
    T_poststat[:,j]=t_poststat


# Save T_prior, N_arr, Nproc_arr in one file
np.savez('timing_Nproc%d_N%d.npz' % (len(Nproc_arr), len(N_arr)), T_prior=T_prior, T_forward=T_forward, T_rejection=T_rejection, T_poststat=T_poststat, N_arr=N_arr, Nproc_arr=Nproc_arr)


#%% 
ax, fig = plt.subplots(1,1, figsize=(10,5))
plt.loglog(N_arr, t_prior, '-*',label='Prior model')
plt.plot(N_arr, t_forward, '-*', label='Forward model')
plt.plot(N_arr, t_rejection, '-*', label='Rejection sampling')
plt.plot(N_arr, t_poststat, '-*', label='Posterior statistics')
plt.xlabel('Number of samples')
plt.ylabel('Time [s]')
plt.legend()
plt.show()



# %%
