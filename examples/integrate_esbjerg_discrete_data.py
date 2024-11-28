#!/usr/bin/env python
# %% [markdown]
# # INTEGRATE on ESBJERG data

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
    # # # # # # #%load_ext autoreload
    # # # # # # #%autoreload 2
    pass
# %%
import integrate as ig
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

# check if parallel computations can be performed
parallel = ig.use_parallel(showInfo=1)
hardcopy = True
# %% Get tTEM data from DAUGAARD
N=5000
case = 'ESBJERG'
files = ig.get_case_data(case=case)
f_data_h5 = files[0]
file_gex= ig.get_gex_file_from_data(f_data_h5)

print("Using data file: %s" % f_data_h5)
print("Using GEX file: %s" % file_gex)

ig.plot_geometry(f_data_h5, pl='LINE')
X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)

f_data_new_h5 = 'DATA.h5'
os.system('cp %s %s' % (f_data_h5,f_data_new_h5))
f_data_h5= f_data_new_h5

# %% [markdown]
# ## 1. Setup the prior model ($\rho(\mathbf{m},\mathbf{d})$
#
# In this example a simple layered prior model will be considered

# %% [markdown]
# ### 1a. first, a sample of the prior model parameters, $\rho(\mathbf{m})$, will be generated

# %% A. CONSTRUCT PRIOR MODEL OR USE EXISTING

# Layered model
f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='chi2', NLAY_deg=3, RHO_min=1, RHO_max=500)
#f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='uniform', NLAY_min=1, NLAY_max=8, RHO_min=1, RHO_max=500)

# Plot some summary statistics of the prior model
#ig.plot_prior_stats(f_prior_h5)

# %% [markdown]
# ### 1b. Then, a corresponding sample of $\rho(\mathbf{d})$, will be generated

# %% Compute prior DATA
f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, parallel=parallel, showInfo=0)


# %% [markdown]
# ## Discrete prior model type:
# Say som knowledge about about a specific claim C, is available at some locations
# C:"In the top 10m, the resistivity is between 10 and 100 Ohm.m"
# Say the observations is the probability that this claim is correct.
#
# In order to use such data, 
#    
#   1* Construct a prior model type that can represent the claim
#   2* Construct the identity of the prior model as a prior data type
#   3* Define a new data observation, that represent P(C|location)
#


#%% Go through all prior models and compute C(mi)
M, idx = ig.load_prior_model(f_prior_h5)

nm= len(M)
# get x/z
with h5py.File(f_prior_h5, 'r') as f_prior:
    z = f_prior['M1'].attrs['x']
            
z_min = 0
z_max = 10
rho_min = 10
rho_max = 100


im = 0
RHO = M[im]        
nmodels = RHO.shape[0]

# go trhough all nm models and compute whether the claim is true
C = np.zeros(nmodels)
for i in range(nmodels):
    rho = RHO[i,:]
    if np.all(rho[0:10] > rho_min) and np.all(rho[0:10] < rho_max):
        C[i] = 1

plt.hist(C)
plt.xlabel('rho(C|RHO)')

M_new = C

# Wrote prior data
#D_obs  = np.zeros((nm,2))
#ig.write_prior_data(D_obs, f_prior_data_h5, im)

# %%


def write_prior_model(f_prior_data_h5, M_new, x_new, class_id=[], class_name=[], delIfExist = True):

    if len(class_id) == 0:
        # get number of unuqie values in M
        class_id = np.sort(M_unique)

    if len(class_name) == 0:
        # set class_id to 'class%d' % i
        class_name = ['class%d' % i for i in class_id]

    M_new = np.atleast_2d(M_new).T
    key = 'M%d' % (nm+1 )

    # Delete key if it exists
    if delIfExist:
        with h5py.File(f_prior_data_h5, 'a') as f:
            if key in f.keys():
                print('Deleting %s' % key)
                del f[key]
        
    # Write DATA
    with h5py.File(f_prior_data_h5, 'a') as f:
        f.create_dataset(key, data=M_new)

    # Write attributes to DATA
    with h5py.File(f_prior_data_h5, 'a') as f:
        f[key].attrs['name'] = 'C:Rho in top 10m between 10 and 100 Ohm.m'
        f[key].attrs['x'] = x_new
        f[key].attrs['class_name'] = class_name
        f[key].attrs['class_id'] = class_id

        print(f.keys())
        

x_new = np.arange(M_new.shape[0])
x_new = np.atleast_2d(x_new).T

# Write prior model - strictly not needed
write_prior_model(f_prior_data_h5, M_new, x_new, class_id=[], class_name=[], delIfExist = True)

# Write prior data - NEEEDED
#write_prior_data()


# %%
