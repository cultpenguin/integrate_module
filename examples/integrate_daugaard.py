#!/usr/bin/env python
# %% [markdown]
# # INTEGRATE Daugaard Case Study with three eology-resistivity prior models.
#
# This notebook contains an example of inverison of the DAUGAARD tTEM data using three different geology-resistivity prior models

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
    # # # # # # #%load_ext autoreload
    # # # # # # #%autoreload 2
    pass

import integrate as ig
import numpy as np
import os
import matplotlib.pyplot as plt
import h5py
hardcopy=True

# %% [markdown]
# ## Download the data DAUGAARD data including non-trivial prior data


# %% SELECT THE CASE TO CONSIDER AND DOWNLOAD THE DATA
files = ig.get_case_data(case='DAUGAARD', loadType='prior') # Load data and prior realizations
f_data_h5 = files[0]
file_gex= ig.get_gex_file_from_data(f_data_h5)
# check that file_gex exists
if not os.path.isfile(file_gex):
    print("file_gex=%s does not exist in the current folder." % file_gex)

print('Using hdf5 data file %s with gex file %s' % (f_data_h5,file_gex))

# %% SELECT THE PRIOR MODEL
# A1. Compute prior data form existing prior model
f_prior_h5='prior_detailed_general_N2000000_dmax90.h5'
#% plot some 1D statistics of the prior
ig.plot_prior_stats(f_prior_h5)
plt.show

#% Compute prior data
N_use = 50000
f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, N=N_use)


#%%

updatePostStat =True
f_post_h5 = ig.integrate_rejection(f_prior_data_h5, f_data_h5, 
                                    N_use = N_use, 
                                    parallel=1, 
                                    updatePostStat=updatePostStat, 
                                    showInfo=1)


# %% Plot some statistics of the posterior
ig.plot_T_EV(f_post_h5, pl='T', hardcopy=hardcopy)
ig.plot_T_EV(f_post_h5, pl='EV', hardcopy=hardcopy)
#% Plot Profiles
ig.plot_profile(f_post_h5, i1=0, i2=1000, hardcopy=hardcopy)

#% Export to CSV
ig.post_to_csv(f_post_h5)
