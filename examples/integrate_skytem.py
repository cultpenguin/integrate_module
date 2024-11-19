#!/usr/bin/env python
# %% [markdown]
# # Getting started with INTEGRATE
#
# This notebook contains an example using SkyTEM data

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
    # # # #%load_ext autoreload
    # # # #%autoreload 2
    pass
# %%
import integrate as ig
import h5py
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
# check if parallel computations can be performed
parallel = ig.use_parallel(showInfo=1)


#%% 
f_mat = 'tma_data.mat'
# load MAT file using scipy
data = sp.io.loadmat(f_mat)

H = data['H'][0]
DALL = data['data_print']
gate_times = data['gate_times'][0]
ng = gate_times.shape[0]

ns, nd = DALL.shape

UTMX = DALL[:,0]
UTMY = DALL[:,1]
LINE = DALL[:,2]
ELEV = DALL[:,3]
ALT = DALL[:,4]
D = DALL[:,5:]
# replace all values of 9999 with NaN
D[D == 9999] = np.nan


# find number of data that are not Nan in each row of D
NOBS = np.sum(~np.isnan(D), axis=1)

D_obs = D[:,0:ng]
D_std = D[:,ng:2*ng]

ALT_mean = np.mean(ALT)
ALT_std = np.std(ALT)
ALT_mode = sp.stats.mode(ALT)
print('Mean altitude: %f' % ALT_mean)
print('Mode altitude: %f' % ALT_mode.mode)
print('Standard deviation of altitude: %f' % ALT_std)
# Fit a chi2 distribution to the altitude data
ALT_chi2 = sp.stats.chi2.fit(ALT)
print('Chi2 fit to altitude data: ', ALT_chi2)
# generate ns samples from the fitted distribution
ALT_sample = sp.stats.chi2.rvs(*ALT_chi2, size=ns)


#%%
doPlot = False
if doPlot:
    plt.figure()
    plt.scatter(UTMX, UTMY, c=ELEV, cmap='jet', s=1)
    plt.colorbar()
    plt.title('Elevation')
    plt.axis('equal')
    plt.savefig('TMA_Elevation.png')
    plt.show()

    plt.figure()
    plt.scatter(UTMX, UTMY, c=ALT, cmap='jet', s=1)
    plt.colorbar()
    plt.title('Altitude')
    plt.axis('equal')
    plt.savefig('TMA_Altitude.png')
    plt.show()

#%%
if doPlot:
    plt.figure()
    plt.hist(ALT, bins=100, label='obs', color='b', alpha=1)
    plt.hist(ALT_sample, bins=100, label='obs', color='r', alpha=.7)
    plt.title('Altitude Histogram')
    plt.legend()
    plt.savefig('TMA_Altitude_Histogram.png')
    plt.show()

#%% plot data
if doPlot:
    plt.figure()
    plt.subplot(311)
    #plt.semilogy(gate_times, D_obs[0,0:1000], label='Observed')
    plt.semilogy(D_obs[0:ns,:], label='Observed',linewidth=0.5)
    plt.title('Observed Data')
    plt.subplot(312)
    plt.plot(D_std[0:ns,:], label='Observed',linewidth=0.5);
    plt.ylim([1, 2])
    plt.title('Standard Deviation')
    plt.subplot(313)
    plt.plot(ELEV,'k-',linewidth=0.1, label='Elevation')
    plt.plot(ELEV+ALT,'r-',linewidth=0.1, label='Elevation+Altitude')
    plt.plot(NOBS,'g-',linewidth=0.1, label='Nobs')
    plt.legend()
    plt.savefig('TMA_Data.png')

# find entroyes where NOBS<2
##nthres=22
#idx = np.where(NOBS < nthres)[0]
#print('Number of data with NOBS < %d: %d' % (nthres,idx.shape[0]))



#%% 
file_gex = '20170403_340m2_304_650Hz_at_SR_Update.gex'
G = ig.read_gex(file_gex)
stm_files, GEX = ig.gex_to_stm(file_gex)
print("Using GEX file: %s" % file_gex)

#%% 
N=100
# Layered model
f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='chi2', NLAY_deg=3, RHO_min=1, RHO_max=3000)
with h5py.File(f_prior_h5, 'r') as f:
    x = f['M1'].attrs['x']

# Plot some summary statistics of the prior model

#P = ig.load_prior(f_prior_h5)
M,idx = ig.load_prior_model(f_prior_h5)
M1 = M[0]
M2 = M[1]
M3 = ALT_sample[0:N]
# force M3 to be a 2d numpy array
M3 = M3.reshape(-1,1)

# update f_prior_h5 with 'M3' data, and M3/x attribute
with h5py.File(f_prior_h5, 'a') as f:
    f['M3'] = M3
    f['M3'].attrs['x'] = x
    f['M3'].attrs['is_discrete'] = 0

#if doPlot:
ig.plot_prior_stats(f_prior_h5)
if doPlot:
    plt.plot(M[0][0])

# %% 
thickness = np.diff(x)
#D_ref = ig.forward_gaaem_old(C=1./M1.T, thickness=thickness, file_gex=file_gex, tx_height=ALT_sample)
D_ref = ig.forward_gaaem(C=1./M1, thickness=thickness, file_gex=file_gex, tx_height=M3.flatten())

# %% Compute prior DATA
# Remember to set the id of the prior model parameters with ALTITUDE!!
f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, parallel=False, im=1, id=1, im_height=3)
#f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, parallel=False, im=1, id=1)
with h5py.File(f_prior_data_h5, 'r') as f:
    D1 = f['D1'][:]

# %%
plt.semilogy(D1,'k-')
plt.semilogy(D_ref,'r-')


# %% COnstruct the OBSERVED data 
f_data_h5 = 'DATA_TNO.h5'
ig.write_data_gaussian(D_obs, D_std = D_std, id=1, is_log = 0, f_data_h5=f_data_h5)

# %%
