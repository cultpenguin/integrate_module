#!/usr/bin/env python
# %% [markdown]
# # Getting started with INTEGRATE - Using data from the PriorGeneratorApp

# %%
import integrate as ig
import h5py 
import numpy as np

# check if parallel computations can be performed*
parallel = ig.use_parallel(showInfo=1)
hardcopy = False 
import matplotlib.pyplot as plt
plt.show()
# %% Get tTEM data from DAUGAARD
case = 'DAUGAARD'
files = ig.get_case_data(case=case)
f_data_h5 = files[0]
file_gex= ig.get_gex_file_from_data(f_data_h5)

print("Using data file: %s" % f_data_h5)
print("Using GEX file: %s" % file_gex)

# plot the Geometry
#ig.plot_geometry(f_data_h5, hardcopy=hardcopy)
# Plot the data
#ig.plot_data(f_data_h5, hardcopy=hardcopy)



# %% [markdown]
# ## 1. Choose the prior model ($\rho(\mathbf{m},\mathbf{d})$
# f_prior_h5 = prior_generator_app('excel_file.cls',N=10000, zmax=90, dz=1, hardcopy=hardcopy)
f_prior_h5 = 'prior_faelles_os_Roesnes_N50000_dmax90.h5'
f_prior_h5 = 'prior_faelles_os_Roesnes_N50000_dmax90.h5'
f_prior_h5 = 'prior_faelles_os_Roesnes_01042025_N1000000_dmax90.h5';


with h5py.File(f_prior_h5, 'a') as f:
    x_orig = f['M1'].attrs['x']
    x_len = len(x_orig)
    x = np.arange(x_len)
    if x_orig[0]==1:
        f['M1'].attrs['x'] = x
        f['M1'].attrs['z'] = x

with h5py.File(f_prior_h5, 'r') as f:
    N = f['M1'].shape[0]
    x = f['M1'].attrs['x']
    
print(N)


#%% Make sure all atts are set
ig.integrate_update_prior_attributes(f_prior_h5,showInfo=1)

#%% 
#ig.plot_prior_stats(f_prior_h5, hardcopy=hardcopy)


# %% [markdown]
# ### 1b. Then, a corresponding sample of $\rho(\mathbf{d})$, will be generated

# %% Compute prior DATA
N_use = N
f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, parallel=parallel, showInfo=0, N=N_use)

#ig.plot_data_prior(f_prior_data_h5,f_data_h5,nr=1000,hardcopy=hardcopy)
# %% [markdown]
# ## Sample the posterior $\sigma(\mathbf{m})$
#
# The posterior distribution is sampling using the extended rejection sampler.

# %% READY FOR INVERSION
f_post_h5 = ig.integrate_rejection(f_prior_data_h5, 
                                   f_data_h5, 
                                   N_use = N_use, 
                                   showInfo=1, 
                                   parallel=parallel)


# %% [markdown]
# ### Plot some statistic from $\sigma(\mathbf{m})$

# %% Plot prior, posterior, and observed  data
ig.plot_data_prior_post(f_post_h5, i_plot=100,hardcopy=hardcopy)
ig.plot_data_prior_post(f_post_h5, i_plot=0,hardcopy=hardcopy)

# %% Posterior analysis
# Plot the Temperature used for inversion
ig.plot_T_EV(f_post_h5, pl='T',hardcopy=hardcopy)
# Plot the evidnence (prior likelihood) estimated as part of inversion
ig.plot_T_EV(f_post_h5, pl='EV',hardcopy=hardcopy)

# %% Plot Profiles
ig.plot_profile(f_post_h5, i1=1, i2=2000, im=1, hardcopy=hardcopy)
# %%

# Plot a 2D feature: Resistivity in layer 10
try:
    ig.plot_feature_2d(f_post_h5,im=1,iz=5, key='Median', uselog=1, cmap='jet', s=10,hardcopy=hardcopy)
    plt.show()
except:
    pass

try:
    ig.plot_feature_2d(f_post_h5,im=1,iz=30, key='Median', uselog=1, cmap='jet', s=10,hardcopy=hardcopy)
    plt.show()
except:
    pass

try:
    ig.plot_feature_2d(f_post_h5,im=1,iz=50, key='Median', uselog=1, cmap='jet', s=10,hardcopy=hardcopy)
    plt.show()
except:
    pass




try:
    # Plot a 2D feature: The number of layers
    ig.plot_feature_2d(f_post_h5,im=2,iz=0,key='Median', title_text = 'Number of layers', uselog=0, clim=[1,6], cmap='jet', s=12,hardcopy=hardcopy)
    plt.show()
except:
    pass

# %% Export to CSV
f_csv, f_point_csv = ig.post_to_csv(f_post_h5)
