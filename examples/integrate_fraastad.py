#!/usr/bin/env python
# %% [markdown]
# # INTEGRATE Fraastad example

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
import numpy as np
import matplotlib.pyplot as plt

# %% Choose the GEX file used for forward modeling. THis should be stored in the data file.
#file_gex= ig.get_gex_file_from_data(f_data_h5, id=id)
f_data_h5 = 'Fra20200930_202001001_1_AVG_export.h5'
file_gex ='fraastad_ttem.gex'
print("Using GEX file: %s" % file_gex)


# %% [markdown]
# ## 1. Setup the prior model ($\rho(\mathbf{m},\mathbf{d})$
#
# In this example a simple layered prior model will be considered
# %% A1. CONSTRUCT PRIOR MODEL OR USE EXISTING
N=5000000
N=1000000
RHO_min = 1
RHO_max = 300
# Layered model
f_prior_h5 =ig.prior_model_layered(N=N,lay_dist='chi2', NLAY_deg=5, rho_dist='log-uniform', RHO_min=RHO_min, RHO_max=RHO_max)
f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='uniform', NLAY_min=1, NLAY_max=4, rho_dist='log-uniform', RHO_min=1, RHO_max=100)

# %% A2. Compute prior DATA
f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, Nproc=16)

# %% [markdown]
# ## Sample the posterior $\sigma(\mathbf{m})$
#
# The posterior distribution is sampling using the extended rejection sampler.

# %% READY FOR INVERSION
N_use = 10000000
f_post_h5 = ig.integrate_rejection(f_prior_data_h5, f_data_h5, N_use = N_use, parallel=1, updatePostStat=False, showInfo=1)
#convert -delay 10 -loop 0 POST_*Median*feature*.png animation.gif
#ffmpeg -i animation.gif -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" animation.mp4
# %% Compute some generic statistic of the posterior distribtiuon (Mean, Median, Std)
ig.integrate_posterior_stats(f_post_h5)


# %% [markdown]
# ### Plot some statistic from $\sigma(\mathbf{m})$

# %% Posterior analysis
# Plot the Temperature used for inversion
ig.plot_T(f_post_h5)

# %% Plot Profiles
ig.plot_profile_continuous(f_post_h5, i1=1000, i2=2000, im=1, clim=[1,100])
# %%

# Plot a 2D feature: Resistivity in layer 10
ig.plot_feature_2d(f_post_h5,im=1,iz=12, key='Median', uselog=1, cmap='jet', s=10, clim=np.log10([RHO_min,RHO_max]))
#ig.plot_feature_2d(f_post_h5,im=1,iz=80,key='Median')

#%% 
for iz in range(90):
    ig.plot_feature_2d(f_post_h5,im=1,iz=iz, key='Median', uselog=1, cmap='jet', s=10, clim=np.log10([RHO_min,RHO_max]))

#%%

try:
    # Plot a 2D feature: The number of layers
    ig.plot_feature_2d(f_post_h5,im=2,iz=0,key='Median', title_text = 'Number of layers', cmap='jet', s=12)
except:
    pass





# %%
