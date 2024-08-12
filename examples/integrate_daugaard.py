#!/usr/bin/env python
# %% [markdown]
# # INTEGRATE Case Study example
#
# This notebook contains an examples of the simplest use of INTEGRATE, on which tTEM data from various caswe study areas, will be be inverted using simple generic, resistivity only, prior models.
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
    # # # # # #%load_ext autoreload
    # # # # # #%autoreload 2
    pass

import integrate as ig
import numpy as np
import os
import matplotlib.pyplot as plt
hardcopy=True

# %% [markdown]
# ## Download the data DAUGAARD data including non-trivial prior data


# %% SELECT THE CASE TO CONSIDER AND DOWNLOAD THE DATA
files = ig.get_case_data(case='DAUGAARD')
#files = ig.get_case_data(case='DAUGAARD', loadAll=True)
f_data_h5 = files[0]
file_gex= ig.get_gex_file_from_data(f_data_h5)
# check that file_gex exists
if not os.path.isfile(file_gex):
    print("file_gex=%s does not exist in the current folder." % file_gex)

print('Using hdf5 data file %s with gex file %s' % (f_data_h5,file_gex))

# %% [markdown]
# ### Plot the geometry of the observed data

# %% plot the data
fig = ig.plot_data_xy(f_data_h5)

# %% [markdown]
# ### Plot the observed data

# %% Plot the observed data
ig.plot_data(f_data_h5)


# %% SELECT THE PRIOR MODEL
# A1. CONSTRUCT PRIOR MODEL OR USE EXISTING

N_use = 100000

f_prior_h5_list = []
f_post_h5_list = []
f_prior_h5_list.append('prior_detailed_invalleys_N2000000_dmax90.h5')
f_prior_h5_list.append('prior_detailed_outvalleys_N2000000_dmax90.h5')
f_prior_h5_list.append('prior_detailed_general_N2000000_dmax90.h5')

for f_prior_h5  in f_prior_h5_list:
    print('Using prior model file %s' % f_prior_h5)

    #% plot some 1D statistics of the prior
    ig.plot_prior_stats(f_prior_h5)

    #% Compute prior data
    f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, N=N_use)

    #% READY FOR INVERSION [NOTE: CHANGE 'N_use', to 'N'.]

    N_use = 10000000
    #f_prior_data_h5 = 'gotaelv2_N1000000_fraastad_ttem_Nh280_Nf12.h5'
    updatePostStat =True
    f_post_h5 = ig.integrate_rejection(f_prior_data_h5, f_data_h5, 
                                       N_use = N_use, 
                                       parallel=1, 
                                       updatePostStat=updatePostStat, 
                                       showInfo=1)
    f_post_h5_list.append(f_post_h5)


#%%
for f_post_h5 in f_post_h5_list:

    # %% Posterior analysis
    # Plot the Temperature used for inversion
    ig.plot_T_EV(f_post_h5, pl='T')
    ig.plot_T_EV(f_post_h5, pl='EV')
    ig.plot_T_EV(f_post_h5, pl='ND')


    # %% Plot Profiles
    ig.plot_profile(f_post_h5, i1=0, i2=2000, cmap='jet', hardcopy=hardcopy)

    # %% Export to CSV
    ig.post_to_csv(f_post_h5)
    
