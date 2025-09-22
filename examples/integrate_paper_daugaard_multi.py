#!/usr/bin/env python
# %% [markdown]
# # Daugaard Case Study with three lithology-resistivity prior models.
#
# This notebook contains an example of inverison of the DAUGAARD tTEM data using three different lithology-resistivity prior models

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
    # # # # # # # #%load_ext autoreload
    # # # # # # # #%autoreload 2
    pass

import integrate as ig
# Check if parallel computations can be performed
parallel = ig.use_parallel(showInfo=1)
import numpy as np
import os
import matplotlib.pyplot as plt
import h5py

from integrate.integrate_io import copy_prior
hardcopy=True

# %% [markdown]
# ## Download the data DAUGAARD data including non-trivial prior data realizations

# %%
useMergedPrior=True
useGenericPrior=True
N_use = 2000000

files = ig.get_case_data(case='DAUGAARD', loadType='prior_data') # Load data and prior+data realizations
f_data_h5 = files[0]
file_gex= ig.get_gex_file_from_data(f_data_h5)
# check that file_gex exists
if not os.path.isfile(file_gex):
    print("file_gex=%s does not exist in the current folder." % file_gex)

print('Using hdf5 data file %s with gex file %s' % (f_data_h5,file_gex))

ig.plot_geometry(f_data_h5, pl='NDATA', hardcopy= hardcopy, cmap='viridis')
plt.show()


# %% Load Dauagard data and increase std by a factor of 3
inflateNoise = 1
if inflateNoise != 1:
    gf=inflateNoise
    print("="*60)
    print("Increasing noise level (std) by a factor of %d" % gf)
    print("="*60)
    D = ig.load_data(f_data_h5)
    D_obs = D['d_obs'][0]
    D_std = D['d_std'][0]*gf
    f_data_old_h5 = f_data_h5
    f_data_h5 = 'DAUGAARD_AVG_gf%g.h5' % (gf) 
    ig.copy_hdf5_file(f_data_old_h5, f_data_h5)
    ig.write_data_gaussian(D_obs, D_std=D_std, f_data_h5=f_data_h5, file_gex=file_gex)

ig.plot_data(f_data_h5, hardcopy= hardcopy)
plt.show()

# %% Get geometry and data info
X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)

with h5py.File(f_data_h5,'r') as f_data:
    # find number of nan values on d_obs
    NON_NAN = np.sum(~np.isnan(f_data['/%s' % 'D1']['d_obs']), axis=1)



# Find a unique list of line number, then find the LINEnumber that occurs most frequently
unique_lines, counts = np.unique(LINE, return_counts=True)
most_frequent_line = unique_lines[np.argmax(counts)]
print("Most frequent line number:", most_frequent_line)

# find the indexes of the most frequent line
id_line = np.where(LINE == most_frequent_line)[0]

# Only use the first entries of id_line until the index change more than 2
id_line_diff = np.diff(id_line)
id_line_cut = np.where(id_line_diff > 2)[0]
if len(id_line_cut) > 0:
    id_line = id_line[:id_line_cut[0]+1]

# set id_line to 100,101...,1001
id_line = np.arange(2000, 3001)

# slect indexes to plot

i_plot_1 = 100
i_plot_2 = 1000

plt.figure(figsize=(10, 6))
plt.scatter(X, Y, c=NON_NAN, s=1,label='Survey Points')
plt.plot(X[id_line],Y[id_line], 'k.', markersize=4, label='All Survey Points')
plt.plot(X[i_plot_1],Y[i_plot_1], 'k*', markersize=10, label='P1')
plt.plot(X[i_plot_2],Y[i_plot_2], 'k*', markersize=10, label='P2')
plt.grid()
plt.colorbar(label='Elevation (m)')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Survey Points Colored by Number of Non-NaN Data Points')
plt.axis('equal')
plt.legend()
if hardcopy:
    plt.savefig('DAUGAARD_survey_points_elevation.png', dpi=300)
plt.show()

i1=np.min(id_line)
i2=np.max(id_line)+1


#%%  [markdown]
# ## Compute prior data from prior model if they do not already exist

f_prior_data_h5_list = []
f_prior_data_h5_list.append('daugaard_valley_new_N1000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5')
f_prior_data_h5_list.append('daugaard_standard_new_N1000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5')

if useMergedPrior:
    f_prior_data_merged_h5 = ig.merge_prior(f_prior_data_h5_list, f_prior_merged_h5='daugaard_merged.h5', showInfo=2)
    f_prior_data_h5_list.append(f_prior_data_merged_h5)

if useGenericPrior:
    N=N_use
    f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='chi2', NLAY_deg=4, RHO_min=1, RHO_max=3000, f_prior_h5='PRIOR.h5')
    f_prior_data_generic_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, NdoMakePriorCopy=True)
    f_prior_data_h5_list.append(f_prior_data_generic_h5)

# %%
# Select how many prior model realizations (N) should be generated

f_post_h5_list = []

for i_prior in range(len(f_prior_data_h5_list)):

    f_prior_data_h5= f_prior_data_h5_list[i_prior]
    ig.integrate_update_prior_attributes(f_prior_data_h5)

    # plot prior data and observed data
    ig.plot_data_prior(f_prior_data_h5, f_data_h5, i_plot=100, hardcopy=hardcopy)
    
    # Get filename without extension
    fileparts = os.path.splitext(f_prior_data_h5)
    f_post_h5 = 'post_%s_Nuse%d_inflateNoise%d.h5' % (fileparts[0], N_use,inflateNoise)

    f_post_h5 = ig.integrate_rejection(f_prior_data_h5, 
                                    f_data_h5, 
                                    f_post_h5, 
                                    N_use = N_use, 
                                    showInfo=1, 
                                    parallel=True, 
                                    updatePostStat=False)
   

    f_post_h5_list.append(f_post_h5)    


# %% 
for i_post in range(len(f_post_h5_list)):
    ig.integrate_posterior_stats(f_post_h5_list[i_post], showInfo=1)
    
#%%
for i_post in range(len(f_post_h5_list)):
    f_post_h5 = f_post_h5_list[i_post]
    
    ig.plot_data_prior_post(f_post_h5, i_plot=i_plot_1, hardcopy=hardcopy)
    ig.plot_data_prior_post(f_post_h5, i_plot=i_plot_2, hardcopy=hardcopy)

    ig.plot_data_prior_post(f_post_h5, i_plot=100, hardcopy=hardcopy)
    
    ig.plot_T_EV(f_post_h5, pl='LOGL_mean', hardcopy=hardcopy)

    ig.plot_profile(f_post_h5, i1=i1, i2=i2, hardcopy=hardcopy)

    ig.plot_feature_2d(f_post_h5,im=1,iz=15, key='Mean', uselog=1, s=10,hardcopy=hardcopy)
    plt.show()
    ig.plot_feature_2d(f_post_h5,im=2,iz=15, key='Mode', uselog=0, s=10,hardcopy=hardcopy)
    plt.show()


# %% [markdown]
# ## Effect of size of prior data set

# %% EFFECT OF SIZE

N_use_arr = [1000,10000,100000,1000000]
N_use_arr = [1000,10000,100000]

f_post_h5_N_list = []

for N_use in N_use_arr:
    for i_prior in range(len(f_prior_data_h5_list)):

        f_prior_data_h5= f_prior_data_h5_list[i_prior]
        # Get filename without extension
        fileparts = os.path.splitext(f_prior_data_h5)
        f_post_h5 = 'post_%s_Nuse%d_inflateNoise%d.h5' % (fileparts[0], N_use,inflateNoise)

        f_post_h5 = ig.integrate_rejection(f_prior_data_h5, 
                                        f_data_h5, 
                                        f_post_h5, 
                                        N_use = N_use, 
                                        showInfo=1, 
                                        parallel=True, 
                                        updatePostStat=True)
        
        f_post_h5_N_list.append(f_post_h5)
    
#%% 
for i_post in range(len(f_post_h5_N_list)):
    f_post_h5 = f_post_h5_N_list[i_post]

    
    ig.plot_data_prior_post(f_post_h5, i_plot=i_plot_1, hardcopy=hardcopy)
    ig.plot_data_prior_post(f_post_h5, i_plot=i_plot_2, hardcopy=hardcopy)
    
    ig.plot_data_prior_post(f_post_h5, i_plot=100, hardcopy=hardcopy)
    
    ig.plot_T_EV(f_post_h5, pl='LOGL_mean', hardcopy=hardcopy)

    ig.plot_profile(f_post_h5, i1=i1, i2=i2, hardcopy=hardcopy)

    ig.plot_feature_2d(f_post_h5,im=1,iz=15, key='Mean', uselog=1, s=10,hardcopy=hardcopy)
    plt.show()
    ig.plot_feature_2d(f_post_h5,im=2,iz=15, key='Mode', uselog=0, s=10,hardcopy=hardcopy)
    plt.show()




# %% [markdown]
# ## Effect of size of prior data set

# %% EFFECT OF SIZE

T_base_arr = [1,2,10,20,100]
N_use = 10000

f_post_h5_T_list = []

for T_base in T_base_arr:
    for i_prior in range(len(f_prior_data_h5_list)):

        f_prior_data_h5= f_prior_data_h5_list[i_prior]
        # Get filename without extension
        fileparts = os.path.splitext(f_prior_data_h5)
        f_post_h5 = 'post_%s_Nuse%d_T%d_inflateNoise%d.h5' % (fileparts[0], N_use,T_base,inflateNoise)

        f_post_h5 = ig.integrate_rejection(f_prior_data_h5, 
                                        f_data_h5, 
                                        f_post_h5, 
                                        T_base = T_base,
                                        N_use = N_use,
                                        autoT=False,
                                        showInfo=1, 
                                        parallel=True, 
                                        updatePostStat=True)
        
        f_post_h5_T_list.append(f_post_h5)




#%% 
# concatenate f_post_h5_list, f_post_h5_N_list, f_post_h5_T_list
f_post_h5_all_list = f_post_h5_list + f_post_h5_N_list + f_post_h5_T_list

#f_post_h5_all_list = f_post_h5_T_list


for i_post in range(len(f_post_h5_all_list)):
    f_post_h5 = f_post_h5_all_list[i_post]

    ig.plot_data_prior_post(f_post_h5, i_plot=i_plot_1, hardcopy=hardcopy)
    ig.plot_data_prior_post(f_post_h5, i_plot=i_plot_2, hardcopy=hardcopy)
    
    ig.plot_data_prior_post(f_post_h5, i_plot=100, hardcopy=hardcopy)
    
    ig.plot_T_EV(f_post_h5, pl='LOGL_mean', hardcopy=hardcopy)
    ig.plot_T_EV(f_post_h5, pl='T', hardcopy=hardcopy)
    ig.plot_T_EV(f_post_h5, pl='EV', hardcopy=hardcopy)
    ig.plot_T_EV(f_post_h5, pl='ND', hardcopy=hardcopy)

    ig.plot_profile(f_post_h5, i1=i1, i2=i2, hardcopy=hardcopy)

    #ig.plot_feature_2d(f_post_h5,im=1,iz=15, key='LogMean', uselog=1, s=10,hardcopy=hardcopy)
    ig.plot_feature_2d(f_post_h5,im=1,iz=15, key='Median', uselog=1, s=10,hardcopy=hardcopy)
    plt.show()
    ig.plot_feature_2d(f_post_h5,im=2,iz=15, key='Mode', uselog=0, s=10,hardcopy=hardcopy)
    plt.show()
