#!/usr/bin/env python
# %% [markdown]
# # INTEGRATE - synthetic prior models and prior data
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
    # # # # # # # # # #%load_ext autoreload
    # # # # # # # # # #%autoreload 2
    pass

import integrate as ig
# check if parallel computations can be performed
parallel = ig.use_parallel(showInfo=1)

import numpy as np
import os
import matplotlib.pyplot as plt
hardcopy=True

# %% [markdown]
# ## Download the data for a specific case study
#
# The following case study areas are available: 
#
# * DAUGAARD
# * FANGEL
# * HALD
#

# %% SELECT THE CASE TO CONSIDER AND DOWNLOAD THE DATA
case = 'DAUGAARD'

files = ig.get_case_data(case=case)
f_data_h5 = files[0]
file_gex= ig.get_gex_file_from_data(f_data_h5)
# check that file_gex exists
if not os.path.isfile(file_gex):
    print("file_gex=%s does not exist in the current folder." % file_gex)

print('CASE: %s' % case)
print('Using hdf5 data file %s with gex file %s' % (f_data_h5,file_gex))

# %% [markdown]
# ## Setup up the prior , $\rho(m,d)$
# A lookup table of prior model parameters and corresponding prior data needs to be defined
#
#

# %% [markdown]
# ### Prior model paramegters, $\rho(m)$: Setup the prior for the model parameters
# In principle and arbitrarily complex prior can be used with INTEGRATE, quantifying information about both discrete and continuous model parameters, and modle parameters describing physical parameters, and geo related parameters.
# Here, we consider using a simple generic resistivity only prior.
#
#

# %% SELECT THE PRIOR MODEL
# A1. CONSTRUCT PRIOR MODEL OR USE EXISTING
N=200000
RHO_min = 1
RHO_max = 2500
RHO_dist='log-uniform'
NLAY_min=1 
NLAY_max=9 
z_max = 90
## Layered model
f_prior_h5 = ig.prior_model_layered(N=N,
                                    lay_dist='uniform', z_max = z_max, 
                                    NLAY_min=NLAY_min, NLAY_max=NLAY_max, 
                                    RHO_dist=RHO_dist, RHO_min=RHO_min, RHO_max=RHO_max)
    
ig.plot_prior_stats(f_prior_h5)




# %% Make a few forward realizations
# Compute Prior Data
f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, parallel=parallel, showInfo=0, Ncpu=64)
ig.plot_data_prior(f_prior_data_h5,f_data_h5,nr=1000,alpha=1, ylim=[1e-13,1e-5], hardcopy=hardcopy) 


# %%

# %%
