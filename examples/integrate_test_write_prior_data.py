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
N=500
# Layered model
f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='chi2', NLAY_deg=4, RHO_min=1, RHO_max=3000)

# compute prior data
f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, parallel=parallel, showInfo=0, Ncpu=8)

# plot prior data and observed data
ig.plot_data_prior(f_prior_data_h5, f_data_h5)
# %% create some new data
D_prior_arr, idx = ig.load_prior_data(f_prior_data_h5)
nd = len(D_prior_arr)

D_new = D_prior_arr[0]
# Add gaussian noise
D_new = 10**(np.log10(D_new) + np.random.normal(0, .13, size=D_prior_arr[0].shape))
D_new = np.real(D_new)
D_new = np.abs(D_new)


# %%

ig.save_prior_data(f_prior_data_h5, D_new)
ig.save_prior_data(f_prior_data_h5, 10*D_new)
ig.save_prior_data(f_prior_data_h5, 50*D_new)
ig.save_prior_data(f_prior_data_h5, D_new/10, id=2, force_delete=False)
ig.save_prior_data(f_prior_data_h5, D_new/10, id=2, force_delete=True, method='other', with_noise=1)

# %%
ig.plot_data_prior(f_prior_data_h5, f_data_h5, id=1)
ig.plot_data_prior(f_prior_data_h5, f_data_h5, id=2, id_data=1)
ig.plot_data_prior(f_prior_data_h5, f_data_h5, id=3, id_data=1)
ig.plot_data_prior(f_prior_data_h5, f_data_h5, id=4, id_data=1)
plt.show()
# %%
