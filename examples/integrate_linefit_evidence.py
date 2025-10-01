#!/usr/bin/env python
# %% [markdown]
# # Daugaard Case Study with three lithology-resistivity prior models.
#
# This notebook contains an example of inverison of the DAUGAARD tTEM data using three different lithology-resistivity prior models

# %%
try:
    # Check if the code is running in an IPython kernel (which includes Jupyter notebooks)
    get_ipython()
    # If the above line doesn't raise an error, it means we are in a Jupyter environment
    # Execute the magic commands using IPython's run_line_magic function
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except:
    # If get_ipython() raises an error, we are not in a Jupyter environment
    # # # # # # # #%load_ext autoreload
    # # # # # # # #%autoreload 2
    pass

import integrate as ig
# Check if parallel computations can be performed
parallel = ig.use_parallel(showInfo=1)
import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt
import h5py

from integrate.integrate_io import copy_prior
hardcopy=True

autoT=False
T_base=1

# %% load mat file
f_mat = 'linefit_evidence_N100000_std10.mat'
# load matlab file using scipy
mat = scipy.io.loadmat(f_mat)
d_obs  = np.atleast_2d(mat['d_obs']).T
d_std = np.atleast_2d(mat['d_std']).T

EV_org = mat['EV']
EV_mix_org = mat['EV_mix']

P_org = mat['P_hyp'].flatten()
P_mix_org = mat['P_hyp_mix'].flatten()


f_data_h5='d_linefit.h5'
ig.write_data_gaussian(d_obs, D_std=d_std, f_data_h5=f_data_h5, id=1, showInfo=2)

# %% First we use a single prior for which 
# that is a mixture prior of the 4 hypothesis

M1 = mat['d_sim_mix']
M2 = mat['hypothesis_mix']
D1 = mat['d_sim_obs_mix']
f_prior_data_mix_h5='linefit_prior_data_mix.h5'
ig.save_prior_model(f_prior_data_mix_h5, M1, im=1,delete_if_exist=True)
ig.save_prior_model(f_prior_data_mix_h5, M2, im=2)
ig.save_prior_data(f_prior_data_mix_h5, D1)


f_post_h5 = 'post_mix_linefit.h5'
f_post_h5 = ig.integrate_rejection(f_prior_data_mix_h5, 
                                f_data_h5, 
                                f_post_h5=f_post_h5, 
                                showInfo=1, 
                                parallel=True, 
                                autoT=autoT,
                                T_base=T_base,
                                updatePostStat=True)

with h5py.File(f_post_h5,'r') as f:
    EV_mix = f['/EV'][:]
    T_mix = f['/T'][:]
    LOGL_mean_mix = f['/LOGL_mean'][:]
    N_UNIQUE_mix = f['/N_UNIQUE'][:]
    i_use_mix = f['/i_use'][:]
    

print("EV_mix from integrate: ", EV_mix)
print("EV_mix from reference: ", EV_mix_org)
if np.sum(np.abs(EV_mix_org-EV_mix)) < 1e-5:
    print("EV match reference within 1e-5")
else:
    print("!!!!!!!!!!!!!!! EV do not match reference within 1e-5")

M_mix, idx = ig.load_prior_model(f_prior_data_mix_h5)


post_cat = M_mix[1][i_use_mix]
# find number of poast_cat = 1,2,3,4
N_cat = np.array([np.sum(post_cat==i) for i in [1,2,3,4]])
P_mix = N_cat/np.sum(N_cat)
print("Posterior probability of hypothesis from MIXTURE: ", P_mix)
print("Posterior probability of hypothesis from MIXTURE reference: ", P_mix_org)


max_err = np.max(np.abs(P_mix_org - P_mix))
if max_err < 1e-5:
    print("P match reference within 1e-5")
else:
    print("!!!!!!!!!!!!!!! P do not match reference within 1e-5")   
    print("max error: ", max_err    )


#%% Then we combine the 4 priors into a singkle mixture prior 
f_prior_data_h5_list=[]
f_post_h5_list=[]
EV=[]
for i in [0,1,2,3]:
    M1_single = mat['d_sim'][0][i]
    D1_single = mat['d_sim_obs'][0][i]
    f_prior_data_h5='linefit_prior_data_%d.h5' % (i+1)
    ig.save_prior_model(f_prior_data_h5, M1_single, im=1,delete_if_exist=True)
    ig.save_prior_data(f_prior_data_h5, D1_single)
    f_prior_data_h5_list.append(f_prior_data_h5)



    f_post_h5_single = 'post_linefit_P%d.h5' % (i+1)
    f_post_h5_single = ig.integrate_rejection(f_prior_data_h5, 
                                    f_data_h5, 
                                    f_post_h5=f_post_h5_single, 
                                    showInfo=0, 
                                    parallel=False, 
                                    autoT=autoT,
                                    T_base=T_base,
                                    updatePostStat=True)
    
    with h5py.File(f_post_h5_single,'r') as f:
        EV.append(f['/EV'][:])


    f_post_h5_list.append(f_post_h5_single)

EV = np.array(EV).flatten()

print("EV from integrate: ", EV)
print("EV from reference: ", EV_org)
if np.sum(np.abs(EV_org-EV)) < 1e-5:
    print("EV match reference within 1e-5")
else:
    print("!!!!!!!!!!!!!!! EV do not match reference within 1e-5")


P = np.exp(EV)/np.sum(np.exp(EV))
## % Compare posterior probability of hypothesis from the two methods

print("Posterior probability of hypothesis from integrate: ", P)
print("Posterior probability of hypothesis from reference: ", P_org)

#%% plot bar chart of P*
plt.figure(figsize=(8,6))
width = 0.2
x = np.arange(1,5)
P_org_plot = np.array(P_org).flatten()
P_plot = np.array(P).flatten()
P_mix_plot = np.array(P_mix).flatten()
P_mix_org_plot = np.array(P_mix_org).flatten()
plt.bar(x-width, P_org_plot, width=width, label='Reference (E)', color='C0', alpha=0.5)
plt.bar(x, P_plot, width=width, label='Integrate (E)', color='C1', alpha=0.5)
plt.bar(x+1*width, P_mix_org_plot, width=width, label='Reference (mixture)', color='C3', alpha=0.5)
plt.bar(x+2*width, P_mix_plot, width=width, label='Integrate (mixture)', color='C2', alpha=0.5)
plt.xticks(x, ['P1','P2','P3','P4'])
plt.xlabel('Hypothesis')
plt.ylabel('Posterior Probability')
plt.legend()
plt.title('Posterior Probability of Hypotheses')

