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
    #%load_ext autoreload
    #%autoreload 2
    pass
# %%
import integrate as ig



# %% Choose the GEX file used for forward modeling. THis should be stored in the data file.
#file_gex= ig.get_gex_file_from_data(f_data_h5, id=id)
f_data_h5 = 'DAUGAARD_AVG.h5'
file_gex ='ttem_example.gex'
print("Using GEX file: %s" % file_gex)


# %% [markdown]
# ## 1. Setup the prior model ($\rho(\mathbf{m},\mathbf{d})$
#
# In this example a simple layered prior model will be considered

# %% [markdown]
# ### 1a. first, a sample of the prior model parameters, $\rho(\mathbf{m})$, will be generated

# %% A. LOAD PRIOR MODEL OR USE EXISTING
#f_prior_h5 = 'PRIOR_Daugaard_N2000000.h5'

# %% [markdown]
# ### 1b. Then, a corresponding sample of $\rho(\mathbf{d})$, will be generated

# %% Compute prior DATA
#f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex)
f_prior_data_h5 = 'PRIOR_Daugaard_N2000000_TX07_20230731_2x4_RC20-33_Nh280_Nf12.h5'

# %% [markdown]
# ## Sample the posterior $\sigma(\mathbf{m})$
#
# The posterior distribution is sampling using the extended rejection sampler.

# %% READY FOR INVERSION
N_use = 100000
f_post_h5 = ig.integrate_rejection(f_prior_data_h5, f_data_h5, N_use = N_use, parallel=1, updatePostStat=False, showInfo=1)

# %% Compute some generic statistic of the posterior distribtiuon (Mean, Median, Std)
ig.integrate_posterior_stats(f_post_h5)


# %% [markdown]
# ### Plot some statistic from $\sigma(\mathbf{m})$

# %% Posterior analysis
# Plot the Temperature used for inversion
ig.plot_T(f_post_h5)

# %% Plot Profiles
ig.plot_profile_continuous(f_post_h5, i1=1000, i2=2000, im=1)
# %%

# Plot a 2D feature: Resistivity in layer 10
ig.plot_feature_2d(f_post_h5,im=1,key='Median', uselog=1, cmap='jet', s=10)
#ig.plot_feature_2d(f_post_h5,im=1,iz=80,key='Median')

try:
    # Plot a 2D feature: The number of layers
    ig.plot_feature_2d(f_post_h5,im=2,iz=0,key='Median', title_text = 'Number of layers', cmap='jet', s=12)
except:
    pass





# %%
