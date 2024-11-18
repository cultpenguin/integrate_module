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
plt.figure()
plt.hist(ALT, bins=100, label='obs', color='b', alpha=1)
plt.hist(ALT_sample, bins=100, label='obs', color='r', alpha=.7)
plt.title('Altitude Histogram')
plt.legend()
plt.savefig('TMA_Altitude_Histogram.png')
plt.show()

#%% plot data
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
# %%
nthres=22
idx = np.where(NOBS < nthres)[0]
print('Number of data with NOBS < %d: %d' % (nthres,idx.shape[0]))



#%% 
file_gex = '20170403_340m2_304_650Hz_at_SR_Update.gex'
G = ig.read_gex(file_gex)
stm_files, GEX = ig.gex_to_stm(file_gex)
print("Using GEX file: %s" % file_gex)

#%% 
N=100
# Layered model
f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='chi2', NLAY_deg=8, RHO_min=1, RHO_max=3000)

# Plot some summary statistics of the prior model
ig.plot_prior_stats(f_prior_h5)

#%% 
#P = ig.load_prior(f_prior_h5)
M, x = ig.load_prior_model(f_prior_h5)
M1 = M[0]
M2 = M[1]

with h5py.File(f_prior_h5, 'r') as f_prior:
#    x = f_prior['M1'].attrs['x']
    print(x)
    for key in f_prior.keys():
        print(key)
        # Read teh 'x' attributes from '/M1'


# %% 
plt.plot(M[0][0])

# %% 
thickness = np.diff(x)
D_ref = ig.forward_gaaem(C=1./M1.T, thickness=thickness, file_gex=file_gex, tx_height=ALT_sample)



# %% Get tTEM data from DAUGAARD

# %% [markdown]
# ## 1. Setup the prior model ($\rho(\mathbf{m},\mathbf{d})$
#
# In this example a simple layered prior model will be considered

# %% [markdown]
# ### 1a. first, a sample of the prior model parameters, $\rho(\mathbf{m})$, will be generated

# %% A. CONSTRUCT PRIOR MODEL OR USE EXISTING
N=100000
# Layered model
f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='chi2', NLAY_deg=3, RHO_min=1, RHO_max=3000)

# Plot some summary statistics of the prior model
ig.plot_prior_stats(f_prior_h5)

# %% [markdown]
# ### 1b. Then, a corresponding sample of $\rho(\mathbf{d})$, will be generated

# %% Compute prior DATA
f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, parallel=parallel, showInfo=0, Ncpu=64)

# %% [markdown]
# ## Sample the posterior $\sigma(\mathbf{m})$
#
# The posterior distribution is sampling using the extended rejection sampler.

# %% READY FOR INVERSION
N_use = N
f_post_h5 = ig.integrate_rejection(f_prior_data_h5, 
                                   f_data_h5, 
                                   N_use = N_use, 
                                   showInfo=1, 
                                   parallel=parallel)

# %% Compute some generic statistic of the posterior distribution (Mean, Median, Std)
# This is typically done after the inversion
# ig.integrate_posterior_stats(f_post_h5)

# %% [markdown]
# ### Plot some statistic from $\sigma(\mathbf{m})$

# %% Plot prior, posterior, and observed  data
ig.plot_data_prior_post(f_post_h5, i_plot=100)
ig.plot_data_prior_post(f_post_h5, i_plot=0)

# %% Posterior analysis
# Plot the Temperature used for inversion
ig.plot_T_EV(f_post_h5, pl='T')
# Plot the evidnence (prior likelihood) estimated as part of inversion
ig.plot_T_EV(f_post_h5, pl='EV')

# %% Plot Profiles
ig.plot_profile(f_post_h5, i1=1000, i2=2000, im=1)
# %%

# Plot a 2D feature: Resistivity in layer 10
ig.plot_feature_2d(f_post_h5,im=1,iz=12, key='Median', uselog=1, cmap='jet', s=10)
#ig.plot_feature_2d(f_post_h5,im=1,iz=80,key='Median')

try:
    # Plot a 2D feature: The number of layers
    ig.plot_feature_2d(f_post_h5,im=2,iz=0,key='Median', title_text = 'Number of layers', uselog=0, clim=[1,6], cmap='jet', s=12)
except:
    pass

# %% Export to CSV
f_csv, f_point_csv = ig.post_to_csv(f_post_h5)

# %%
# Read the CSV file
#f_point_csv = 'POST_DAUGAARD_AVG_PRIOR_CHI2_NF_3_log-uniform_N100000_TX07_20231016_2x4_RC20-33_Nh280_Nf12_Nu100000_aT1_M1_point.csv'
import pandas as pd
df = pd.read_csv(f_point_csv)
df.head()

# %%
# Use Pyvista to plot X,Y,Z,Median
import pyvista as pv
import numpy as np
from pyvista import examples
#pv.set_jupyter_backend('trame')
#pv.set_plot_theme("document")
#p = pv.Plotter(notebook=True)
p = pv.Plotter()
filtered_df = df[(df['Median'] < 50) | (df['Median'] > 200)]
#filtered_df = df[(df['LINE'] > 1000) & (df['LINE'] < 1400) ]
points = filtered_df[['X', 'Y', 'Z']].values[:]
median = np.log10(filtered_df['Mean'].values[:])
opacity = np.where(filtered_df['Median'].values[:] < 100, 0.5, 1.0)
#p.add_points(points, render_points_as_spheres=True, point_size=3, scalars=median, cmap='jet', opacity=opacity)
p.add_points(points, render_points_as_spheres=True, point_size=6, scalars=median, cmap='hsv')
p.show_grid()
p.show()


# %%
