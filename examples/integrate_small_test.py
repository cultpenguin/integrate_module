#!/usr/bin/env python
# %% [markdown]
# # INTEGRATE Daugaard Case Study with three eology-resistivity prior models.
#
# This notebook contains an example of inverison of the DAUGAARD tTEM data using three different geology-resistivity prior models

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
    # # # # # # # #%load_ext autoreload
    # # # # # # # #%autoreload 2
    pass

import integrate as ig
import numpy as np
import os
import matplotlib.pyplot as plt
import h5py
hardcopy=True
import time
# %% [markdown]
# ## Download the data DAUGAARD data including non-trivial prior data


#%%
import h5py

# CONSTRUCT NEW PRIOR
MakeNewPrior = True
if MakeNewPrior:
    file1 = 'prior_detailed_invalleys_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5'
    file2 = 'prior_detailed_outvalleys_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5'
    file12 = 'prior_detailed_inout_N4000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5'

    ig.copy_hdf5_file(file1, file12, N=10)

    # Read D1 from f1['D1'], and D2 from f1['D2']. 
    # COmbine D1 and D2 to D12 havig doubvle size of D1 
    # Then replace f12['D1'] with D12
    # Then copy the rest of the data from f1 to f12
    f1 = h5py.File(file1, 'r')
    f2 = h5py.File(file2, 'r')
    f12 =  h5py.File(file12, 'r+')
    N_in = f1['D1'].shape[0]

    print('updating D1')
    D1 = f1['D1'][:]
    D2 = f2['D1'][:]
    D12 = np.concatenate((D1, D2), axis=0)
    del f12['D1']
    f12['D1']=D12

    print('updating M1')
    M1 = f1['M1'][:]
    M2 = f2['M1'][:]
    M12 = np.concatenate((M1, M2), axis=0)
    del f12['M1']
    f12['M1']=M12

    print('updating M2')
    M1 = f1['M2'][:]
    M2 = f2['M2'][:]
    M12 = np.concatenate((M1, M2), axis=0)
    del f12['M2']
    f12['M2']=M12

    print('creating M2')
    D2a = np.zeros(N_in)+1
    D2b = np.zeros(N_in)+1
    D2 = np.concatenate((D2a, D2b), axis=0)
    f12['D2']=D12
    # add  attrubute of 'f5_forward' as 'none' to data set D2
    f12['D2'].attrs['f5_forward'] = 'none'
    f12['D2'].attrs['with_noise'] = 0

    f1.close()
    f2.close()
    f12.close()

    # Make new data 'D2' that is an observation of a specific class
    

#%%


# %% SELECT THE CASE TO CONSIDER AND DOWNLOAD THE DATA
loadData = False
if loadData:
    files = ig.get_case_data(case='DAUGAARD', loadType='prior_data') # Load data and prior+data realizations
    f_data_h5 = files[0]
    file_gex= ig.get_gex_file_from_data(f_data_h5)

    f_prior_h5 = 'prior_detailed_invalleys_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5'


f_data_h5 = 'DAUGAARD_RAW.h5'
f_data_h5 = 'DAUGAARD_AVG.h5'
f_prior_h5 = 'prior_detailed_invalleys_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5'
f_prior_h5 = 'prior_detailed_outvalleys_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5'
f_prior_h5 = 'prior_detailed_inout_N4000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5'


#f_prior_data_h5 = 'gotaelv2_N1000000_fraastad_ttem_Nh280_Nf12.h5'
updatePostStat =False
N_use = 10000
f_post_h5 = ig.integrate_rejection(f_prior_h5, f_data_h5, 
                                N_use = N_use, 
                                parallel=1, 
                                updatePostStat=updatePostStat, 
                                showInfo=1,
                                Nproc = 16)

# %% Likelihood computation
# We need to construct a way to compute the likelihood for a given
# noise model, likelihood_gaussian(d, d_obs, d_std, Cd_inv)
#
# Then we need a general sampling rejection sampler that works
# in either single data points, or a set of data points
# For each data point it should compute the likelihood for all data types. 
# Then apply annealing
# Then combined multiple likelihoos into one.
# Then sample from that likleihood using rejection sampling
# to obtained the indexes..
#

# set which data types to use

def likelihood_gaussian_diagonal(D, d_obs, d_std):
    """
    Compute the likelihood for a single data point
    """
    # Compute the likelihood
    dd = D - d_obs
    # Sequential
    #L = np.zeros(D.shape[0])
    #for i in range(D.shape[0]):
    #    L[i] = -0.5 * np.nansum(dd[i]**2 / d_std**2)
    # Vectorized
    L = -0.5 * np.nansum((D - d_obs)**2 / d_std[0]**2, axis=1)

    return L

def likelihood_gaussian_full(D, d_obs, Cd):
    a = 1
    return 1


N_use = 9000000
id_use = [1,1]
i=0
# the data
with h5py.File(f_data_h5, 'r') as f_data:
    d_obs = f_data['/D1/d_obs'][:]
    d_std = f_data['/D1/d_std'][:]

# load D
D = []
with h5py.File(f_prior_h5, 'r') as f_prior:
    for id in id_use:
        DS = '/D%d' % id
        N = f_prior[DS].shape[0]
        print(f_prior[DS].shape)
        print('Reading %s' % DS)
        if N_use<N:
            doRandom=False
            if doRandom:
                idx = np.random.choice(N, N_use, replace=False)
                print('Reading %s' % DS)
                Dsub = f_prior[DS][np.sort(idx)]
            else:
                Dsub = f_prior[DS][0:N_use]
            D.append(Dsub)
        else:        
            D.append(f_prior[DS][:])

        print(D[i].shape)



L=[]
t0=time.time()
for id in range(len(D)):
    L_single = likelihood_gaussian_diagonal(D[0], d_obs[0], (id+1)*d_std[0])
    L.append(L_single)

t2=time.time()-t0

print('Time for vectorized: %f' % t2)




'''
Cd_inv = 1.0/d_std**2
d_obs = f['d_obs'][id]
print(f.keys())
print(f['d_obs'][id])
print(f['d_std'][id])
print(f['d'][id])
print(f['d_std
'''



# %%
