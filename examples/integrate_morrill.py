#% Morill example

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

# %%
import integrate as ig
# check if parallel computations can be performed
parallel = ig.use_parallel(showInfo=1)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import loglog
import time
import h5py
# get name of CPU
import os
import socket

#%% files
file_in = '1D_P22_NO500_451_ABC5000000_0000_D2_HTX1_1.h5'
file_in_post_h5 = '1D_P22_NO500_451_ABC5000000_0000_D2_HTX1_1_ME0_aT1_CN1.h5'
f_prior_h5 = 'PRIOR_Morill.h5'
f_data_h5 = 'DATA_Morill.h5'

#%% READ HDF5 TRAINING data from Morill AI paper 

# Load D1, D2, M1, M2, M3, M4, M5, M6 from the HDF5 file
print('Reading data from file: ', file_in)
with h5py.File(file_in, 'r') as f:
    print('Reading D1 data from file: ', file_in)
    D1 = f['D1'][:]
    #D2 = f['D2'][:]
    print('Reading M1 data from file: ', file_in)
    M1 = 10**(f['M1'][:])
    print('Reading M2 data from file: ', file_in)
    M2 = f['M2'][:]
    print('Reading M3 data from file: ', file_in)
    M3 = f['M3'][:]
    #print('Reading M4 data from file: ', file_in)
    #M4 = f['M4'][:]
    print('Reading M5 data from file: ', file_in)
    M5 = f['M5'][:]
    print('Reading M6 data from file: ', file_in)
    M6 = f['M6'][:]
    print('DONE Read from file: %s' %(file_in) )
    
#%% WRITE PRIOR TRAINING DATA IN INTEGRATE HDF5 format
print('Writing data to file: ', f_prior_h5)
# write D1 and D2 to f_prior_data_h5
with h5py.File(f_prior_h5, 'w') as f:
    print('Writing D1 to file: ', f_prior_h5)
    f.create_dataset('D1', data=D1)
    
    print('Writing M1 to file: ', f_prior_h5)
    f.create_dataset('M1', data=M1)
    f['M1/'].attrs['name'] = 'Resistivity'
    f['M1/'].attrs['is_discrete'] = 0
    f['M1/'].attrs['clim'] = [1,250]
    f['M1/'].attrs['x'] = np.arange(M1.shape[1])
    
    print('Writing M2 to file: ', f_prior_h5)
    f.create_dataset('M2', data=M2)
    f['M2/'].attrs['name'] = 'Lithology'
    f['M2/'].attrs['is_discrete'] = 1
    f['M2/'].attrs['x'] = np.arange(M2.shape[1])
    f['M2/'].attrs['class_id'] = [0, 1, 2]
    f['M2/'].attrs['class_name'] = ['A', 'B', 'C']

    print('Writing M3 to file: ', f_prior_h5)
    f.create_dataset('M3', data=M3)
    f['M3/'].attrs['name'] = '+-'
    f['M3/'].attrs['is_discrete'] = 1
    f['M3/'].attrs['x'] = np.arange(M3.shape[1])
    f['M3/'].attrs['clim'] = [-.5, 1.5]
    f['M3/'].attrs['class_id'] = [0, 1]
    f['M3/'].attrs['class_name'] = ['+', '-']


    print('Writing M4 to file: ', f_prior_h5)
    f.create_dataset('M4', data=M5)
    f['M4/'].attrs['name'] = 'Thickness'    
    f['M4/'].attrs['is_discrete'] = 0
    f['M4/'].attrs['x'] = np.array(0)
    
    print('Writing M5 to file: ', f_prior_h5)
    f.create_dataset('M5', data=M6)
    f['M5/'].attrs['name'] = 'Elevation'    
    f['M5/'].attrs['is_discrete'] = 0
    f['M5/'].attrs['x'] = np.array(0)
    
print('DONE Writing data to file: ', f_prior_h5)
ig.integrate_update_prior_attributes(f_prior_h5)

#%% load observed data
print('Reading OBSERVED data from file: ', file_in_post_h5)
with h5py.File(file_in_post_h5, 'r') as f:
    d_obs = f['D_obs'][:]
    d_std = f['D_std'][:]
    d_std=d_std*2
    d_std[:,-1]=d_std[:,-1]*1000
    #d_std[:,2:4]=d_std[:,2:4]*1000
    EL_est = f['EL_est'][:]
    EL_OBS = f['EL_obs'][:]

nd=len(d_obs)

ELEVATION = d_obs[:,-1]
LINE = ELEVATION.copy()*0+1000
UTMX = np.arange(nd)
UTMY = UTMX.copy()*0

# write d_obs to f_data_h5['D1/d_obs']
print('Writing OBSERVED data from file: ', f_data_h5)
with h5py.File(f_data_h5, 'w') as f:
    f.create_dataset('D1/d_obs', data=d_obs)
    f.create_dataset('D1/d_std', data=d_std)
    # write attribute 'noise_model' as 'gaussian' to '/D1/' in f_data_h5
    f['/D1'].attrs['noise_model'] = 'gaussian'
    f.create_dataset('ELEVATION', data=ELEVATION[:,np.newaxis])
    f.create_dataset('LINE', data=LINE[:,np.newaxis])
    f.create_dataset('UTMX', data=UTMX[:,np.newaxis])
    f.create_dataset('UTMY', data=UTMY[:,np.newaxis])

# %%
f_post_h5 = ig.integrate_rejection(f_prior_h5, f_data_h5, N_use = 1000000, showInfo=1, parallel=True, updatePostStat=False, Ncpu = 8)
ig.integrate_posterior_stats(f_post_h5, showInfo=1)

# %%
ig.plot_profile(f_post_h5)
#ig.plot_data_prior_post(f_post_h5)
ig.plot_data_prior_post(f_post_h5, i_plot=10)
# %%

# %%
# Read '/T' from f_post_h5
with h5py.File(f_post_h5, 'r') as f:
    T = f['T'][:]
    EV = f['EV'][:]
    i_use = f['i_use'][:]
    M2_mode = f['/M2/Mode'][:]
# %%
