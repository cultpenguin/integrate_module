#!/usr/bin/env python
# %% [markdown]
# # Getting started with INTEGRATE
#
# This notebook contains a simple example of geeting started with INTEGRATE

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
    # #%load_ext autoreload
    # #%autoreload 2
    pass
# %%
import integrate as ig

# %% CHECK OS
# If os is not UNIX then set paralell to True
import os
if os.name == 'nt':
    parallel = False
else:
    parallel = True
print("OS=%s, Parallel=%s " % (os.name,parallel))


# %% Get tTEM data from DAUGAARD
case = 'DAUGAARD'

files = ig.get_case_data(case=case)
f_data_h5 = files[0]
f_data_h5 = 'DAUGAARD_AVG.h5'
file_gex= ig.get_gex_file_from_data(f_data_h5)

print("Using data file: %s" % f_data_h5)
print("Using GEX file: %s" % file_gex)


# %%

# %% [markdown]
# ## 1. Setup the prior model ($\rho(\mathbf{m},\mathbf{d})$
#
# In this example a simple layered prior model will be considered

# %% [markdown]
# ### 1a. first, a sample of the prior model parameters, $\rho(\mathbf{m})$, will be generated

# %% A. CONSTRUCT PRIOR MODEL OR USE EXISTING
N=100000
# Layered model
f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='chi2', NLAY_deg=3, RHO_min=1, RHO_max=3000)
# WorkBench type layered model
#f_prior_h5 = ig.prior_model_workbench(N=N,RHO_dist='chi2', nlayers=5)

# Plot some summary statistics of the prior model
ig.plot_prior_stats(f_prior_h5)

# %% [markdown]
# ### 1b. Then, a corresponding sample of $\rho(\mathbf{d})$, will be generated

# %% Compute prior DATA
f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, parallel=parallel)

# %% [markdown]
# ## Sample the posterior $\sigma(\mathbf{m})$
#
# The posterior distribution is sampling using the extended rejection sampler.

# %% READY FOR INVERSION
N_use = N
f_post_h5 = ig.integrate_rejection(f_prior_data_h5, 
                                   f_data_h5, 
                                   N_use = N_use, 
                                   updatePostStat=False, 
                                   showInfo=1, 
                                   parallel=parallel)

# %% Compute some generic statistic of the posterior distribtiuon (Mean, Median, Std)
ig.integrate_posterior_stats(f_post_h5)


# %% [markdown]
# ### Plot some statistic from $\sigma(\mathbf{m})$

# %% Plot prior, posterior, and observed  data
ig.plot_data_prior_post(f_post_h5, i_plot=100)
ig.plot_data_prior_post(f_post_h5, i_plot=0)

# %% Posterior analysis
# Plot the Temperature used for inversion
ig.plot_T_EV(f_post_h5, pl='T')

# %% Plot Profiles
ig.plot_profile(f_post_h5, i1=1000, i2=2000, im=1)
# %%

# Plot a 2D feature: Resistivity in layer 10
ig.plot_feature_2d(f_post_h5,im=1,iz=12, key='Median', uselog=1, cmap='jet', s=10)
#ig.plot_feature_2d(f_post_h5,im=1,iz=80,key='Median')

try:
    # Plot a 2D feature: The number of layers
    ig.plot_feature_2d(f_post_h5,im=2,iz=0,key='Median', title_text = 'Number of layers', uselog=0, clim=[1,6], cmap='jet', s=12)
except:
    pass


# %%
