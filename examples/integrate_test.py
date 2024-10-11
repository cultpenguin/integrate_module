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
    # # #%load_ext autoreload
    # # #%autoreload 2
    pass
# %%
import integrate as ig
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
# ## 1. Setup the prior model ($\rho(\mathbf{m},\mathbf{d})$
#
# In this example a simple layered prior model will be considered

# %% [markdown]
# ### 1a. first, a sample of the prior model parameters, $\rho(\mathbf{m})$, will be generated

# %% A. CONSTRUCT PRIOR MODEL OR USE EXISTING
N=10000
f_prior_data_h5 = 'PRIOR_CHI2_NF_3_log-uniform_N%d_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5' % (N)

# check if the file exists
import os
if os.path.isfile(f_prior_data_h5):
    print("Using existing prior model: %s" % f_prior_data_h5)
else:   

    f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='chi2', NLAY_deg=3, RHO_min=1, RHO_max=3000)

    # Plot some summary statistics of the prior model
    #ig.plot_prior_stats(f_prior_h5)
    # Compute prior DATA
    f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, parallel=parallel, showInfo=0)


DATA = ig.load_data(f_data_h5)
D, M, idx = ig.load_prior(f_prior_data_h5)


D = ig.load_prior_data(f_prior_data_h5)[0]
M = ig.load_prior_model(f_prior_data_h5)[0]
D, idx = ig.load_prior_data(f_prior_data_h5, N_use = 998, Randomize=True)
M, idx = ig.load_prior_model(f_prior_data_h5, idx=idx)
# %% [markdown]
# ## Sample the posterior $\sigma(\mathbf{m})$
#
# %% READY FOR INVERSION
# N_use = 100
# 9.1 sec 1 CPU, 1214 ite/s
# 2.0 sec 8 CPUs 950-1000 ite/s
#N_use = 10000
#          1 CPU, 220 ite/s  --> NEW 292 ite/s: 33% faster
# 13.1 sec 8 CPUs 116-131 ite/s
f_post_h5 = ig.integrate_rejection(f_prior_data_h5, 
                                   f_data_h5, 
                                   showInfo=1, 
                                   parallel=parallel,
                                   Ncpu=1,
                                   updatePostStat=0,
                                   use_N_best=0,
                                   N_use = 6000
                                   )

#%% 

# %% Compute some generic statistic of the posterior distribution (Mean, Median, Std)
# ### Plot some statistic from $\sigma(\mathbf{m})$
ig.integrate_posterior_stats(f_post_h5)
#ig.plot_data_prior_post(f_post_h5, i_plot=100)
#ig.plot_data_prior_post(f_post_h5, i_plot=0)
#ig.plot_T_EV(f_post_h5, pl='T')
#ig.plot_T_EV(f_post_h5, pl='EV')
ig.plot_profile(f_post_h5, im=1, i1=0, i2=2800)
ig.plot_profile(f_post_h5, im=2, i1=0, i2=2800)


# %%
