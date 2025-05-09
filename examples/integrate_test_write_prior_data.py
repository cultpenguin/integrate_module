#!/usr/bin/env python
# %% [markdown]
# # Getting started with INTEGRATE
# Example of reading/writing/updating prior data in a prior hdf5 file.

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
    # # #%load_ext autoreload
    # # #%autoreload 2
    pass
# %%
import integrate as ig
import numpy as np
import matplotlib.pyplot as plt
# check if parallel computations can be performed
parallel = ig.use_parallel(showInfo=1)

# %% Get tTEM data from DAUGAARD
case = 'DAUGAARD'
files = ig.get_case_data(case=case)
f_data_h5 = files[0]
f_data_h5 = 'DAUGAARD_AVG.h5'
file_gex= ig.get_gex_file_from_data(f_data_h5)

print("Using data file: %s" % f_data_h5)
print("Using GEX file: %s" % file_gex)

# %% [markdown]
# sample a prior, to compute prior models
N=15000
# Layered model
f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='chi2', NLAY_deg=4, RHO_min=1, RHO_max=3000, f_prior_h5='prior.h5')


# %load prior models
M_prior_arr, idx = ig.load_prior_model(f_prior_h5)
# Find the number of values in each priormodel M_prior_arr[0] with a resitivity below 10.
N_below_10 = np.sum(M_prior_arr[0] < 10, axis=1)

ig.save_prior_model(f_prior_h5,N_below_10, name='N_below_10')



#%%  compute prior data
f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, parallel=parallel, showInfo=0, Ncpu=8)


#%% Make a copy of the prior data
f_prior_data_h5_org = 'f_prior_data_org.h5'
ig.copy_hdf5_file(f_prior_data_h5,f_prior_data_h5_org)
f_prior_data_h5_nn = 'f_prior_data_nn.h5'
ig.copy_hdf5_file(f_prior_data_h5,f_prior_data_h5_nn)


# %% create some new data
D_prior_arr, idx = ig.load_prior_data(f_prior_data_h5)
D_new = np.abs(np.real(10**(np.log10(D_prior_arr[0]) + .04+ np.random.normal(0, .03, size=D_prior_arr[0].shape))))
ig.save_prior_data(f_prior_data_h5_nn, D_new, id=1, force_delete=True)

#ig.plot_data_prior(f_prior_data_h5_org, f_data_h5, id=1)
#ig.plot_data_prior(f_prior_data_h5_nn, f_data_h5, id=1)
#plt.show()
# %% Solve the inverse problem
f_post_h5_org = ig.integrate_rejection(f_prior_data_h5_org, f_data_h5, parallel=parallel, showInfo=0, updatePostStat=False)
f_post_h5_nn = ig.integrate_rejection(f_prior_data_h5_nn, f_data_h5,parallel=parallel, showInfo=0, updatePostStat=False)
ig.integrate_posterior_stats(f_post_h5_org)
ig.integrate_posterior_stats(f_post_h5_nn, showInfo=2)

#%%
ig.plot_profile(f_post_h5_org, i1=800, i2=1000, im=1)
ig.plot_profile(f_post_h5_nn, i1=800, i2=1000, im=1)


# %%
ig.plot_profile(f_post_h5_org, i1=800, i2=1000, im=3)
ig.plot_profile(f_post_h5_nn, i1=800, i2=1000, im=3)
# %%
ig.plot_feature_2d(f_post_h5_org, im=3, key='Median', uselog=False, clim=np.array([1, 20]))
ig.plot_feature_2d(f_post_h5_nn, im=3, key='Median', uselog=False, clim=np.array([1, 20]))
# %%
