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
    # # # # # # # # #%load_ext autoreload
    # # # # # # # # #%autoreload 2
    pass

import integrate as ig
import numpy as np
import os
import matplotlib.pyplot as plt
import time
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
#case = 'FANGEL'
#case = 'HALD'
#case = 'GRUSGRAV' # NOT YET AVAILABLE

files = ig.get_case_data(case=case)
f_data_h5 = files[0]
file_gex= ig.get_gex_file_from_data(f_data_h5)
# check that file_gex exists
if not os.path.isfile(file_gex):
    print("file_gex=%s does not exist in the current folder." % file_gex)

print('CASE: %s' % case)
print('Using hdf5 data file %s with gex file %s' % (f_data_h5,file_gex))

# print all filename in files
for f in files:
    print(f)

    

# %% [markdown]
# ### Plot the geometry of the observed data

# %% plot the data
fig = ig.plot_data_xy(f_data_h5)

# %% [markdown]
# ### Plot the observed data

# %% Plot the observed data
ig.plot_data(f_data_h5)
ig.plot_data(f_data_h5, plType='plot', hardcopy=hardcopy)

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

t = []
t.append(time.time())

# %% SELECT THE PRIOR MODEL
# A1. CONSTRUCT PRIOR MODEL OR USE EXISTING
N=2000000
RHO_min = 10
RHO_max = 2500
RHO_max = 500
RHO_dist='log-uniform'
NLAY_min=1 
NLAY_max=12 
z_max = 90

f_prior_h5_geus =  'prior_detailed_general_N2000000_dmax90.h5'

useP=0
if useP==0:
    ## Layered model
    NLAY_max=30 
    f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='uniform', 
                                        z_max = z_max, 
                                        NLAY_min=NLAY_min, 
                                        NLAY_max=NLAY_max, 
                                        RHO_dist=RHO_dist, 
                                        RHO_min=RHO_min, 
                                        RHO_max=RHO_max)
elif useP==1:
    f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='uniform', 
                                        z_max = z_max, 
                                        NLAY_min=NLAY_min, 
                                        NLAY_max=NLAY_max, 
                                        RHO_dist=RHO_dist, 
                                        RHO_min=RHO_min, 
                                        RHO_max=RHO_max)

elif useP==2:
    ## N layer model with increasing thickness
    NLAY_max=30 
    f_prior_h5 = ig.prior_model_workbench(N=N, z_max = z_max, nlayers=NLAY_max, RHO_dist=RHO_dist, RHO_min = RHO_min, RHO_max = RHO_max)
    
else:
    f_prior_h5 = f_prior_h5_geus

if useP<3:
    import h5py
    with h5py.File(f_prior_h5,'a') as f_prior, h5py.File(f_prior_h5_geus,'r') as f_geus:
        f_prior['/M1'].attrs['clim'] = f_geus['/M1'].attrs['clim']
        f_prior['/M1'].attrs['cmap'] = f_geus['/M1'].attrs['cmap']
        
t.append(time.time())


# %% plot some 1D statistics of the prior
ig.plot_prior_stats(f_prior_h5, hardcopy = hardcopy)

# %% [markdown]
# ### Prior data, $\rho(d)$
# The prior data, i.e. the forwward response of of the realizations of the prior needs to be computed. Here we use only tTEM data, so onæy on type (tTEM) of data is computed.

# %% Compute prior data
f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, N=N)

t.append(time.time())

# %% [markdown]
# ## Sample the posterior $\sigma(\mathbf{m})$
#
# The posterior distribution is sampling using the extended rejection sampler.

# %% READY FOR INVERSION

N_use = N
#f_prior_data_h5 = 'gotaelv2_N1000000_fraastad_ttem_Nh280_Nf12.h5'
updatePostStat =True
#f_post_h5 = ig.integrate_rejection(f_prior_data_h5, f_data_h5, N_use = N_use, parallel=1, updatePostStat=updatePostStat, showInfo=1)
f_post_h5 = ig.integrate_rejection_multi(f_prior_data_h5, f_data_h5, N_use = N_use)

t.append(time.time())

#%% SHOW CPU TIME USAGE


print('1. Time elapsed sample prior models: %f' % (t[1]-t[0]))
print('2. Time elapsed prior data : %f' % (t[2]-t[1]))
print('3. Time elapsed sample posterior : %f' % (t[3]-t[2]))
print('Time elapsed total : %f' % (t[-1]-t[0]))
print('Time elapsed per sounding : %5.1f ms (N_use=%d)' % ( 1000*(t[-1]-t[0])/N , N_use) )

t_total = t[-1]-t[0]
# printe percentate of total time for each step
print('Relative time used %3.2f%%, %3.2f%%, %3.2f%%' % ( 100*(t[1]-t[0])/t_total, 100*(t[2]-t[1])/t_total, 100*(t[3]-t[2])/t_total) )

#wrote timimt til file
fname = '%s_time.txt' % (os.path.splitext(f_post_h5)[0])
with open(fname, 'w') as f:
    f.write('1. Time elapsed sample prior models: %f\n' % (t[1]-t[0]))
    f.write('2. Time elapsed prior data : %f\n' % (t[2]-t[1]))
    f.write('3. Time elapsed sample posterior : %f\n' % (t[3]-t[2]))
    f.write('Time elapsed total : %f\n' % (t[-1]-t[0]))
    f.write('Time elapsed per sounding : %5.1f ms (N_use=%d)\n' % ( 1000*(t[-1]-t[0])/N , N_use) )
    f.write('Realtive time used %3.2f%%, %3.2f%%, %3.2f%%\n' % ( 100*(t[1]-t[0])/t_total, 100*(t[2]-t[1])/t_total, 100*(t[3]-t[2])/t_total) )

# %% [markdown]
# ## Plot some statistics from $\sigma(\mathbf{m})$

# %% [markdown]
# ### The temperature refer to the annealing temperature used by the extended rejection sampler, in order to get 'enough' realizations.
# T=1, implies no anealing has occurred. Higher values of T implies increasingly difficulty of fitting the data within the noise, suggesting either that the lookup table size is too small and/or that the prior is not consistent with the data.

# %% Posterior analysis
# Plot the Temperature used for inversion
ig.plot_T_EV(f_post_h5, pl='T')
ig.plot_T_EV(f_post_h5, pl='EV')
ig.plot_T_EV(f_post_h5, pl='ND')
#

#%% 
X , Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)

# Find the number of unique numbers in LINE, and count how many times each number appears
unique, counts = np.unique(LINE, return_counts=True)
# Find the index of the most common number
most_common = np.argmax(counts)
# Find the most common number
most_common_number = unique[most_common]
# Find the index of the least common number
least_common = np.argmin(counts)
# Find the least common number
least_common_number = unique[least_common]
# Print the results
print('Most common number:', most_common_number)
print('Least common number:', least_common_number)

# find the index of the most common number in LINE
i_line = np.where(LINE == most_common_number)
print(i_line)

i1= 24;i2 = 54
i1= 876;i2 = 937
#i1= 11169;i2 = 11195

# plot prior profile 
ig.integrate_posterior_stats(f_post_h5, usePrior=True)
ig.plot_profile(f_post_h5, i1=i1, i2=i2, hardcopy=hardcopy, txt='_prior')
# plot posterior profile
ig.integrate_posterior_stats(f_post_h5)
ig.plot_profile(f_post_h5, i1=i1, i2=i2, hardcopy=hardcopy, txt='_post')


# %%
import h5py
with h5py.File(f_data_h5,'r') as f_prior:
    nd=len(f_prior['UTMX'][:].flatten())

i1 = np.linspace(0,nd-1,4).astype(int)
for i in i1:
    ig.plot_data_prior_post(f_post_h5, i_plot = i)
    #ig.plot_data_prior_post(f_post_h5, i_plot = 1199)

# %% Plot Profiles
ig.plot_profile(f_post_h5, i1=0, i2=np.min([2000,nd]), cmap='jet', hardcopy=hardcopy);

# %%
try:
    for iz in range(0,z_max,1):
        ig.plot_feature_2d(f_post_h5,im=1,iz=iz,key='Mean', title_text = '',  hardcopy=hardcopy, s=12)
except:
    pass

# ffmpeg -framerate 1 -pattern_type glob -i 'POST_DAUGAARD_AVG_PRIOR_UNIFORM_NL_1-8_log-uniform_N10000_TX07_20231016_2x4_RC20-33_Nh280_Nf12_Nu10000_aT1_1_11693_1_Mean*_feature.png' -c:v libx264 -pix_fmt yuv420p output.mp4

#%%
#import integrate as ig
#f_prior_h5='PRIOR_UNIFORM_NL_1-8_log-uniform_N10000_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5'
#f_post_h5 = 'POST_DAUGAARD_AVG_PRIOR_UNIFORM_NL_1-8_log-uniform_N10000_TX07_20231016_2x4_RC20-33_Nh280_Nf12_Nu10000_aT1.h5'
#clim, cmap = ig.get_clim_cmap(f_prior_h5)
#clim, cmap = ig.get_clim_cmap(f_prior_h5,'/M1')
#ig.plot_feature_2d(f_post_h5,im=1,iz=5,key='Median', title_text = 'XX', cmap='jet', s=12, vmin=10, vmax=100, hardcopy=hardcopy)
#ig.plot_feature_2d(f_post_h5,im=1,iz=5,key='Median', hardcopy=True, s=11)
#ig.plot_feature_2d(f_post_h5,im=1,iz=5,key='Median', title_text = 'XX', hardcopy=True, clim=[10 ,100], cmap='jet',s=11)

#%% post M2
try:
    ig.plot_feature_2d(f_post_h5,im=2,key='Median', title_text = 'Number of layers', uselog=0, cmap='jet_r', clim=[NLAY_min-.5, NLAY_max+.5], s=12)
    #ig.plot_feature_2d(f_post_h5,im=2,key='Median', title_text = 'Number of layers', cmap='hsv', uselog=0, clim=[.5,8.5], s=5)
except:
    pass


# %% Export to CSV
ig.post_to_csv(f_post_h5)
# %%

