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
    # # # # #%load_ext autoreload
    # # # # #%autoreload 2
    pass

import integrate as ig
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
case = 'FANGEL'
#case = 'HALD'

files = ig.get_case_data(case=case)
f_data_h5 = files[0]
file_gex= ig.get_gex_file_from_data(f_data_h5)
# check that file_gex exists
if not os.path.isfile(file_gex):
    print("file_gex=%s does not exist in the current folder." % file_gex)

print('CASE: %s' % case)
print('Using hdf5 data file %s with gex file %s' % (f_data_h5,file_gex))

# %% [markdown]
# ### Plot the geometry of the observed data

# %% plot the data
fig = ig.plot_data_xy(f_data_h5)

# %% [markdown]
# ### Plot the observed data

# %% Plot the observed data
ig.plot_data(f_data_h5)

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
N=100000
RHO_min = 10
RHO_max = 2500
RHO_max = 500
RHO_dist='log-uniform'
NLAY_min=4 
NLAY_max=4 
z_max = 90

useP=1
if useP==1:
    ## Layered model
    #f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='chi2', NLAY_deg=5, z_max = z_max, RHO_dist='log-uniform', RHO_min=RHO_min, RHO_max=RHO_max)
    #f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='uniform', z_max = z_max, NLAY_min=1, NLAY_max=3, RHO_dist='log-uniform', RHO_min=RHO_min, RHO_max=RHO_max)
    f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='uniform', 
                                        z_max = z_max, 
                                        NLAY_min=NLAY_min, 
                                        NLAY_max=NLAY_max, 
                                        RHO_dist=RHO_dist, 
                                        RHO_min=RHO_min, 
                                        RHO_max=RHO_max)
elif useP==2:
    ## N layer model with increasing thickness
    f_prior_h5 = ig.prior_model_workbench(N=N, 
                                          RHO_mean=45, 
                                          RHO_std=45, 
                                          RHO_dist='log-normal', 
                                          z_max = z_max, 
                                          nlayers=1, 
                                          RHO_min = RHO_min, 
                                          RHO_max = RHO_max)
    #f_prior_h5 = ig.prior_model_workbench(N=N, z_max= 30, nlayers=20, RHO_min = RHO_min, RHO_max = RHO_max)
    f_prior_h5 = ig.prior_model_workbench(N=N, z_max = z_max, nlayers=NLAY_max, RHO_dist=RHO_dist, RHO_min = RHO_min, RHO_max = RHO_max)
    
else:
    f_prior_h5 = 'existing_prior.h5'

# %% plot some 1D statistics of the prior
ig.plot_prior_stats(f_prior_h5)

# %% [markdown]
# ### Prior data, $\rho(d)$
# The prior data, i.e. the forwward response of of the realizations of the prior needs to be computed. Here we use only tTEM data, so onæy on type (tTEM) of data is computed.

# %% Compute prior data
f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex)


# %% [markdown]
# ## Sample the posterior $\sigma(\mathbf{m})$
#
# The posterior distribution is sampling using the extended rejection sampler.

# %% READY FOR INVERSION

N_use = 10000000
#f_prior_data_h5 = 'gotaelv2_N1000000_fraastad_ttem_Nh280_Nf12.h5'
updatePostStat =True
f_post_h5 = ig.integrate_rejection(f_prior_data_h5, f_data_h5, N_use = N_use, parallel=1, updatePostStat=updatePostStat, showInfo=1)




# %% [markdown]
# ## Plot some statistics from $\sigma(\mathbf{m})$

# %% [markdown]
# ### The temperature refer to the annealing temperature used by the extended rejection sampler, in order to get 'enough' realizations.
# T=1, implies no anealing has occurred. Higher values of T implies increasingly difficulty of fitting the data within the noise, suggesting either that the lookup table size is too small and/or that the prior is not consistent with the data.

# %% Posterior analysis
# Plot the Temperature used for inversion
ig.plot_T_EV(f_post_h5, pl='T')
ig.plot_T_EV(f_post_h5, pl='EV')
ig.plot_T_EV(f_post_h5, pl='ND')
#

# %%
import h5py
with h5py.File(f_data_h5,'r') as f_prior:
    nd=f_prior['UTMX'].shape[0]


i1 = np.linspace(0,nd-1,4).astype(int)
for i in i1:
    ig.plot_data_prior_post(f_post_h5, i_plot = i)
    #ig.plot_data_prior_post(f_post_h5, i_plot = 1199)

# %% Plot Profiles
ig.plot_profile(f_post_h5, i1=0, i2=np.min([2000,nd]), cmap='jet', hardcopy=hardcopy)

# %%
for iz in range(0,z_max,5):
    ig.plot_feature_2d(f_post_h5,im=1,iz=iz,key='Mean', title_text = 'XX', cmap='jet', s=12, vmin=10, vmax=100, hardcopy=hardcopy)

# %% Export to CSV
ig.post_to_csv(f_post_h5)
# %%
