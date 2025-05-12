#!/usr/bin/env python
# %% [markdown]
# # INTEGRATE EM example - Evidence
#
# This notebook demonstrates howto compute an estimate of the evidence using the INTEGRATE package. 
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
import h5py
# check if parallel computations can be performed
parallel = ig.use_parallel(showInfo=1)

import numpy as np
import os
import matplotlib.pyplot as plt
showInfo = 1
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
case = 'HADERUP' # NOT YET AVAILABLE

files = ig.get_case_data(case=case)
f_data_h5 = files[0]
file_gex= ig.get_gex_file_from_data(f_data_h5)
# check that file_gex exists
if not os.path.isfile(file_gex):
    print("file_gex=%s does not exist in the current folder." % file_gex)

print('CASE: %s' % case)
print('Using hdf5 data file %s with gex file %s' % (f_data_h5,file_gex))

# %% [markdown]
# ### Multiple hypotjhsis: 
# Propose multiple hypothesis.
#


# %% SELECT THE PRIOR MODEL
# A1. CONSTRUCT PRIOR MODEL OR USE EXISTING
N=50000
z_max = 80
RHO_min = 1
RHO_max = 1000
RHO_dist='log-uniform'

f_prior_h5_arr = []

## 4 layered model
NLAY_min=4
NLAY_max=4
f_prior_h5 = ig.prior_model_layered(N=N,
                                    lay_dist='uniform', z_max = z_max, 
                                    NLAY_min=NLAY_min, NLAY_max=NLAY_max, 
                                    RHO_dist=RHO_dist, RHO_min=RHO_min, RHO_max=RHO_max, f_prior_h5 = 'prior_1.h5', showInfo=showInfo)
f_prior_h5_arr.append(f_prior_h5)

## 4 layered model
NLAY_deg = 4
f_prior_h5 = ig.prior_model_layered(N=N,
                                    lay_dist='chi2', z_max = z_max, NLAY_deg=NLAY_deg, 
                                    RHO_dist=RHO_dist, RHO_min=RHO_min, RHO_max=RHO_max, f_prior_h5 = 'prior_2.h5', showInfo=showInfo)
f_prior_h5_arr.append(f_prior_h5)

## 4 layered model
NLAY_deg = 4
RHO_min = 10
f_prior_h5 = ig.prior_model_layered(N=N,
                                    lay_dist='chi2', z_max = z_max, NLAY_deg=NLAY_deg, 
                                    RHO_dist=RHO_dist, RHO_min=RHO_min, RHO_max=RHO_max, f_prior_h5 = 'prior_2.h5', showInfo=showInfo)
f_prior_h5_arr.append(f_prior_h5)



for f_prior_h5 in f_prior_h5_arr:
    ig.plot_prior_stats(f_prior_h5)

# %% [markdown]
# ### Genewarte prior data
f_prior_data_h5_arr = []
for f_prior_h5 in f_prior_h5_arr:
    f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, Ncpu=0, N=N)
    f_prior_data_h5_arr.append(f_prior_data_h5)



# %% TEST 
nprior = len(f_prior_data_h5_arr)

# # Narr should an array from 10 to N, in nstep in logspace
nsteps=11
N_use_arr = np.logspace(2, np.log10(N+1), num=nsteps, dtype=int)


EV_all = []
for f_prior_data_h5 in  f_prior_data_h5_arr:
    EV_arr = []
    for N_use in N_use_arr:
        print("N_use=%d" % N_use)
        f_post_h5 = ig.integrate_rejection(f_prior_data_h5, f_data_h5, parallel=parallel, Ncpu=8, N_use = N_use, updatePostStat=False)
        with h5py.File(f_post_h5, 'r') as f:
            EV = f['/EV'][()]
        EV_arr.append(EV)
    EV_all.append(EV_arr)

# %%
EV_arr = np.array(EV_arr)
for i in range(len(N_use_arr)):
    plt.semilogy(-EV_arr[i,:100],'-',label=str(N_use_arr[i]), linewidth=3-i*0.2)
plt.xlabel('Prior model')
#plt.plot(EV_arr[:,::100].T, )
plt.ylabel('-log(Evidence)')
plt.legend()
plt.title('Evidence for different N_use')
plt.show()



# %%
import matplotlib.pyplot as plt
X, Y, LINE, ELEVATION = ig.get_geometry(f_post_h5)

for ih in np.arange(len(f_prior_data_h5_arr)):
    #ih=2;    
    for j in np.arange(len(N_use_arr)):
        EV = np.squeeze(EV_all[:,j,:])
        # subtract the small value on each column form each column
        P  = np.exp(EV - np.max(EV, axis=0))
        # Normalize each column to sum to 1
        P = P / np.sum(P, axis=0)
        #plt.imshow(P[:,0:100], aspect='auto', cmap='viridis', interpolation='nearest')
        # Plot a cumulgtive sum of the probabilities as an area plot
        #plt.fill_between(np.arange(P.shape[1]), np.sum(P, axis=0), alpha=0.5)
        #plt.plot(np.sum(P, axis=0),'-', label='Cumulative sum of probabilities')
        #plt.xlabel('Prior model')
        ##plt.ylabel('Cumulative sum of probabilities')
        #p#lt.title('Cumulative sum of probabilities for different N_use')
        #plt.legend()
        #plt.show()
        # get X, Y coordinates using ig.get_geometry(f_post_h5)
        plt.subplot(3,4,j+1)
        plt.scatter(X, Y, c=P[ih,:], s=1, cmap='hot_r', alpha=0.5, vmin=0, vmax=1)
        plt.axis('equal')
        plt.title('N_use=%d' % N_use_arr[j])
        # set axiss off
        plt.axis('off')
    plt.colorbar()
    plt.suptitle('Posterior probability of hypothesis H%d' %(ih+1))
    plt.tight_layout()
    plt.savefig('posterior_probabilities_hypothesis_%d.png' % (ih+1), dpi=300)
    plt.show()
        


# %%
