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
case = 'FANGEL'
case = 'HALD'
#case = 'GRUSGRAV' # NOT YET AVAILABLE

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
#fig = ig.plot_data_xy(f_data_h5)

# %% [markdown]
# ### Plot the observed data

# %% Plot the observed data
#ig.plot_data(f_data_h5)
#ig.plot_data(f_data_h5, plType='plot', hardcopy=hardcopy)

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
N=1000000
RHO_min = 1
RHO_max = 2500
RHO_dist='log-uniform'
NLAY_min=1 
NLAY_max=9 
z_max = 90

useP_arr  = [1,2,3,4,5]
#useP_arr  = [5]
f_prior_h5_arr = []
for useP in useP_arr:

    if useP==1:
        ## Layered model
        f_prior_h5 = ig.prior_model_layered(N=N,
                                            lay_dist='uniform', z_max = z_max, 
                                            NLAY_min=NLAY_min, NLAY_max=NLAY_max, 
                                            RHO_dist=RHO_dist, RHO_min=RHO_min, RHO_max=RHO_max)
        f_prior_h5_arr.append(f_prior_h5)
    elif useP==2:
        ## 20 layer model with increasing thickness
        f_prior_h5 = ig.prior_model_workbench(N=N, z_max = z_max,
                                            RHO_mean=45, RHO_std=45, RHO_dist='log-normal', 
                                            RHO_min = RHO_min, RHO_max = RHO_max)
        f_prior_h5_arr.append(f_prior_h5)
    elif useP==3:
        ## NLAY_max-layer model with increasing thickness
        f_prior_h5 = ig.prior_model_workbench(N=N, z_max = z_max, 
                                              nlayers=NLAY_max, 
                                              RHO_dist=RHO_dist, RHO_min = RHO_min, RHO_max = RHO_max)
        f_prior_h5_arr.append(f_prior_h5)
    elif useP==4:
        ## 10 Layered model
        nlay=4
        f_prior_h5 = ig.prior_model_layered(N=N,
                                            lay_dist='uniform', z_max = z_max, 
                                            NLAY_min=1, NLAY_max=nlay, 
                                            RHO_dist=RHO_dist, RHO_min=10, RHO_max=500)
        f_prior_h5_arr.append(f_prior_h5)
    elif useP==5:
        ## 10 Layered model
        ## 1-9 layer model with increasing thickness
        RHO_dist = 'log-uniform'
        LAY_dist='chi2'
        f_prior_h5 = ig.prior_model_workbench(N=N, dz = 1,
                                            lay_dist='chi2', z_max = z_max, 
                                            NLAY_min=NLAY_min, NLAY_max=NLAY_max, 
                                            RHO_dist=RHO_dist, RHO_min=RHO_min, RHO_max=RHO_max)        
        f_prior_h5_arr.append(f_prior_h5)
    
    ig.plot_prior_stats(f_prior_h5)

# %% Make a few forward realizations

for f_prior_h5 in f_prior_h5_arr:

    f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, Ncpu=0, N=1001)
    ig.plot_data_prior(f_prior_data_h5,f_data_h5,nr=1000,alpha=1, ylim=[1e-13,1e-5], hardcopy=hardcopy) 
    plt.show()

    #f_prior_data_h5_arr.append(f_prior_data_h5)
    
# %% Perform inversion and compare
f_prior_data_h5_arr=[]
f_post_h5_arr=[]
for f_prior_h5 in f_prior_h5_arr:

    f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, Ncpu=0, N=N)
    
    f_post_h5 = ig.integrate_rejection(f_prior_data_h5, f_data_h5, parallel=parallel, Ncpu=8)

    f_post_h5_arr.append(f_post_h5)
    f_prior_data_h5_arr.append(f_prior_data_h5)

    ig.plot_T_EV(f_post_h5, pl='EV', hardcopy=hardcopy)
    ig.plot_T_EV(f_post_h5, pl='T', hardcopy=hardcopy)
    
    ig.plot_feature_2d(f_post_h5,im=1,iz=5, key='Median', uselog=1, cmap='jet', s=1, hardcopy=hardcopy)
    plt.show()

    ig.plot_profile(f_post_h5, i1=0, i2=1000, hardcopy=hardcopy)

    ig.plot_data_prior_post(f_post_h5, i_plot=100, hardcopy=hardcopy)
    ig.plot_data_prior_post(f_post_h5, i_plot=1000, hardcopy=hardcopy)

#%%
for f_post_h5 in f_post_h5_arr:
    ig.plot_feature_2d(f_post_h5,im=1,iz=20, key='Median', uselog=1, cmap='jet', s=1, hardcopy=hardcopy)
    plt.show()



# %%
