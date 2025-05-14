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
showInfo = 1
parallel = ig.use_parallel(showInfo=showInfo)

# %% Get tTEM data from DAUGAARD
case = 'DAUGAARD'
files = ig.get_case_data(case=case, showInfo=showInfo)
f_data_h5 = files[0]
f_data_h5 = 'DAUGAARD_AVG.h5'
file_gex= ig.get_gex_file_from_data(f_data_h5)

print("Using data file: %s" % f_data_h5)
print("Using GEX file: %s" % file_gex)

# %% [markdown]
# sample a prior, to compute prior models
showInfo = 1
N=20000
# Layered model
f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='chi2', NLAY_deg=4, RHO_min=1, RHO_max=3000, f_prior_h5='prior.h5', showInfo=showInfo)

# %load prior models
M_prior_arr, idx = ig.load_prior_model(f_prior_h5)
# Find the number of values in each priormodel M_prior_arr[0] with a resitivity below 10.
N_below_10 = np.sum(M_prior_arr[0] < 10, axis=1)

ig.save_prior_model(f_prior_h5,N_below_10, name='N_below_10', showInfo=showInfo)


ig.integrate_update_prior_attributes(f_prior_h5)
#%%  compute prior data
f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, parallel=parallel, showInfo=0, Ncpu=8)

# %% POST
f_post_h5 = ig.integrate_rejection(f_prior_data_h5, f_data_h5, parallel=parallel, showInfo=0, updatePostStat=False, f_post_h5='post.h5')
ig.integrate_posterior_stats(f_post_h5, showInfo=showInfo)

#%%
ig.plot_feature_2d(f_post_h5, im=3, key='Median', uselog=False, clim=np.array([1, 20]))

# %%
