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
import numpy as np
import os
import matplotlib.pyplot as plt
import h5py
hardcopy=True

# %% [markdown]
# ## Download the data DAUGAARD data including non-trivial prior data realizations

# %%
files = ig.get_case_data(case='DAUGAARD') # Load only data
#files = ig.get_case_data(case='DAUGAARD', loadType='prior') # Load data and prior realizations
#files = ig.get_case_data(case='DAUGAARD', loadType='prior_data') # Load data and prior+data realizations
files = ig.get_case_data(case='DAUGAARD', loadType='post') # # Load data and posterior realizations
#files = ig.get_case_data(case='DAUGAARD', loadAll=True) # All of the above
f_data_h5 = files[0]
file_gex= ig.get_gex_file_from_data(f_data_h5)
# check that file_gex exists
if not os.path.isfile(file_gex):
    print("file_gex=%s does not exist in the current folder." % file_gex)

print('Using hdf5 data file %s with gex file %s' % (f_data_h5,file_gex))


# %% [markdown]
# ## Compute prior data from prior model if they do not already exist

# %%
# A1. CONSTRUCT PRIOR MODEL OR USE EXISTING
N_use = 2000000
        
f_prior_h5_list = []
f_prior_h5_list.append('prior_detailed_invalleys_N2000000_dmax90.h5')
f_prior_h5_list.append('prior_detailed_outvalleys_N2000000_dmax90.h5')

f_prior_data_h5_list = []
f_prior_data_h5_list.append('prior_detailed_invalleys_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5')
f_prior_data_h5_list.append('prior_detailed_outvalleys_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5')

# Go through f_prior_data_h5_list. If the file does not exist the compute, it by runinng ig.prior_data_gaaem
for i in range(len(f_prior_h5_list)):
    f_prior_data_h5= f_prior_data_h5_list[i]
    f_prior_h5= f_prior_h5_list[i]
    if not os.path.isfile(f_prior_data_h5):
        print('Prior data file %s does not exist. Computing it.' % f_prior_data_h5)
        # Compute prior data
        f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, N=N_use)
        f_prior_data_h5_list.append(f_prior_data_h5)
    else:
        print('Using existing prior data file %s' % f_prior_data_h5)



# %%
f_data_h5 = 'DAUGAARD_AVG.h5'

f_post_h5_list = []

N_use = 20000
N_use = 200000
#N_use = 2000000

for f_prior_data_h5 in f_prior_data_h5_list:
    print('Using prior model file %s' % f_prior_data_h5)

    #f_prior_data_h5 = 'gotaelv2_N1000000_fraastad_ttem_Nh280_Nf12.h5'
    updatePostStat =True
    f_post_h5 = ig.integrate_rejection(f_prior_data_h5, f_data_h5, 
                                       N_use = N_use, 
                                       parallel=1, 
                                       T_base = 10.0,
                                       autoT=True,
                                       updatePostStat=updatePostStat,                                     
                                       showInfo=1)
    f_post_h5_list.append(f_post_h5)


# %%
for f_post_h5 in f_post_h5_list:
    ig.plot_T_EV(f_post_h5, pl='T', hardcopy=hardcopy)
    plt.show()

# %%
for f_post_h5 in f_post_h5_list:
    ig.plot_T_EV(f_post_h5, pl='EV', hardcopy=hardcopy)
    plt.show()

# %%
for f_post_h5 in f_post_h5_list:
    #% Posterior analysis
    # Plot the Temperature used for inversion
    #ig.plot_T_EV(f_post_h5, pl='T')
    #ig.plot_T_EV(f_post_h5, pl='EV', hardcopy=hardcopy)
    #plt.show()

    #ig.plot_T_EV(f_post_h5, pl='ND')

    #% Plot Profiles
    ig.plot_profile(f_post_h5, i1=0, i2=2000, cmap='jet', hardcopy=hardcopy)
    plt.show()
    #% Export to CSV
    #ig.post_to_csv(f_post_h5)
    #plt.show()

# %%
X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)

nd=len(X)
nev=len(f_post_h5_list)

EV_mul = np.zeros((nev,nd))

iev = -1
for f_post_h5 in f_post_h5_list:
    iev += 1
    # Read '/EV' from f_post_h5
    with h5py.File(f_post_h5, 'r') as f_post:
        print(f_post_h5)
        #EV=(f_post['/EV'][:]) 
        EV_mul[iev]=(f_post['/EV'][:])

#% Normalize EV

EV_P = 0*EV_mul
E_max = np.max(EV_mul, axis=0)

for iev in range(nev):
    EV_P[iev] = np.exp(EV_mul[iev]-E_max)

# Use annealing to flaten prob
T_EV = 10
EV_P = EV_P**(1/T_EV)

EV_P_sum = np.sum(EV_P,axis=0)
for iev in range(nev):
    EV_P[iev] = EV_P[iev]/EV_P_sum


# %%
plt.figure(figsize=(10,6), dpi=600)
plt.subplot(1,1,1)
plt.plot(X, Y, '.', markersize=3, color='gray')
plt.scatter(X, Y, c=EV_P[0], cmap='RdBu_r', s=1, vmin=0, vmax=1, zorder=2)
plt.tight_layout()
plt.axis('equal')
plt.colorbar()
plt.title('P(In Valley)')
plt.xlabel('UTMX [m]')
plt.ylabel('UTMY [m]')
plt.grid()
plt.savefig('DAUGAARD_N%07d_EV_Pin.png' % (N_use), dpi=600)
plt.show()

#%%
plt.figure(figsize=(10,6), dpi=600)
plt.subplot(1,1,1)
plt.plot(X, Y, '.', markersize=3, color='gray')
plt.scatter(X, Y, c=EV_P[1], cmap='RdBu_r', s=1, vmin=0, vmax=1, zorder=2)
plt.tight_layout()
plt.axis('equal')
plt.colorbar()
plt.grid()
plt.title('P(Out of valleys)')
plt.xlabel('UTMX [m]')
plt.ylabel('UTMY [m]')
plt.savefig('DAUGAARD_N%07d_EV_Pout.png' % (N_use), dpi=600)
plt.show()

#%%
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# Create a discrete two-color colormap
colors = ['black', '#0173B2']  # Black and blue - colorblind friendly
cmap = mcolors.ListedColormap(colors)

plt.figure(figsize=(10,6), dpi=600)
# Get the index of the highest value in each column in EV_P_sum
EV_mode = np.argmax(EV_P, axis=0)
EV_P_max = np.max(EV_P, axis=0)
psize = (EV_P_max-0.5)*5+0.001
plt.subplot(1,1,1)
#plt.plot(X, Y, 'k.', markersize=4)
plt.scatter(X, Y, c=EV_mode, cmap=cmap, s=psize, zorder=2)
plt.axis('equal')
plt.grid()
plt.tight_layout()
#cbar = plt.colorbar(ticks=[0, 1])
cbar = plt.colorbar(ticks=[0.25, 0.75])
cbar.set_ticklabels(['In', 'Out'])
plt.xlabel('UTMX [m]')
plt.ylabel('UTMY [m]')

plt.savefig('DAUGAARD_N%07d_EV_mode.png' % (N_use), dpi=600)
# %%


