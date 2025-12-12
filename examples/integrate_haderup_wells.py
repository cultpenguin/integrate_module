#!/usr/bin/env python
# %% [markdown]
# # Haderup tTEM and boreholes
# This is an example demonstrating the use of borehole information using INTEGRATE.
# Specifically, the example demonstarte the use of information about lithology obtain from boreholes.
#

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
import time
import os
import integrate as ig
# check if parallel computations can be performed
parallel = ig.use_parallel(showInfo=1)
if __name__ == "__main__":
    parallel  = True

# Configure matplotlib backend for WSLg/GUI display
ig.setup_matplotlib_backend()

import h5py
import numpy as np
import matplotlib.pyplot as plt
import copy
from geoprior1d import geoprior1d
hardcopy=True

# remove all files with name 'da*IDEN*h5'
for file in os.listdir('.'):
    if file.startswith('daugaard') and 'IDEN' in file and file.endswith('.h5'):
        print("Removing existing file: %s" % file)
        os.remove(file)

t0 = time.time()

# %%
N=1000000
N_use = N
dmax=90
dz=1

P_single=0.99
inflateTEMNoise = 1
# Extrapolation options for distance weighting
r_data=10 # XY-distance based weight for extrapolating borehole information to the data grid
r_dis=100 # DATA-distance based weight for extrapolating borehole information to the data grid 
r_dis=300


# %%
# Get Daugaard data files
case = 'HADERUP'
files = ig.get_case_data(case=case, loadAll=True)
f_data_h5 = files[0]
f_prior_xls = files[3]
file_gex= ig.get_gex_file_from_data(f_data_h5)


print("Using data file: %s" % f_data_h5)
print("Using GEX file: %s" % file_gex)


# %% [markdown]
# ## Define the information from the WELLs

# %%
WELLS = []

W = {}
W['depth_top'] =     [0,  8, 12, 16, 34]
W['depth_bottom'] =  [8, 12, 16, 28, 36]
W['lithology_obs'] = [1,  2,  1,  5,  4]
W['lithology_prob'] = [.9, .9, .9, .9, .9]
W['X'] = 498832.5
W['Y'] = 6250843.1
W['name'] = '65. 795'                     
WELLS.append(W)

W = {}
W['depth_top'] =     [0,  9.8, 15.6, 17.8, 36.2, 39]
W['depth_bottom'] =  [9.8, 15.6, 17.8, 36.2, 39, 45.2]
W['lithology_obs'] = [1,  2,  1,  4,  5,  4]
W['lithology_prob'] = [.9, .9, .9, .9, .9, .9]
W['X'] = 498804.95
W['Y'] = 6249940.89
W['name'] = '65. 732'                     

WELLS.append(W)


# %% [markdown]
# ## Compute prior tTEM data
# Compute prior tTEM data, if they do not allready exist

# %%
#% prior models
f_prior_h5 = 'haderup_N%d_dmax%d_dz%d.h5' % (N, dmax, dz)

# check if f_prior_h5 exists, if not create it
if not os.path.isfile(f_prior_h5):
    print("Creating prior file: %s" % f_prior_h5)

    filename, flags = geoprior1d(
            input_data=f_prior_xls,
            Nreals=N, dmax=dmax, dz=dz,
            output_file = f_prior_h5  # Optional: specify custom output filename
        )

    # prior data 
    f_prior_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, doMakePriorCopy=False)
else:
    f_prior_h5_old = f_prior_h5
    f_prior_h5 = f_prior_h5.replace('.h5', '_copy.h5')
    ig.copy_hdf5_file(f_prior_h5_old, f_prior_h5, showInfo=2)

    print("Using existing f_prior_h5=%s" % (f_prior_h5))


# %% [markdown]
# ## Select a profile going through the wells

# %%
X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)
nd= len(X)

with h5py.File(f_data_h5,'r') as f_data:
    # find number of nan values on d_obs
    NON_NAN = np.sum(~np.isnan(f_data['/%s' % 'D1']['d_obs']), axis=1)

# select the profile line
X1 = WELLS[0]['X']+10; Y1=WELLS[0]['Y']+300;
X2= WELLS[1]['X']+0;   Y2=WELLS[1]['Y']+0;
X3= WELLS[1]['X']-140;   Y3=WELLS[1]['Y']-600;


# Find indeces of data points along points (Xl,Yl), within buffer distance
Xl = np.array([X1, X2, X3])
Yl = np.array([Y1, Y2, Y3])
buffer = 15.0
id_line, distances, segment_ids = ig.find_points_along_line_segments(
    X, Y, Xl, Yl, tolerance=buffer
)

plt.figure(figsize=(10, 6))
plt.scatter(X, Y, c=NON_NAN, s=1,label='Survey Points')
#plt.plot(X[id_line],Y[id_line], 'k-', markersize=8, label='Profile', zorder=2, linewidth=5)
plt.plot(X[id_line],Y[id_line], 'r.', markersize=8, label='Profile', zorder=2, linewidth=5)
plt.grid()

plt.plot(WELLS[0]['X'],WELLS[0]['Y'],'k*', markersize=15, label=WELLS[0]['name'], zorder=3)
plt.plot(WELLS[1]['X'],WELLS[1]['Y'],'k*', markersize=10, label=WELLS[1]['name'], zorder=3)
plt.colorbar(label='Elevation (m)')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Survey Points Colored by Number of Non-NaN Data Points')
plt.axis('equal')
plt.legend()
if hardcopy:
    plt.savefig('HADERUP_survey_points_nonnan.png', dpi=300)
plt.show()

# %%
# find indexes of data ppints closest to the well locations
i_well = []
for iw in np.arange(len(WELLS)):
    W = WELLS[iw]
    dists_to_well = np.sqrt((X - W['X'])**2 + (Y - W['Y'])**2)
    iw_closest = np.argmin(dists_to_well)
    i_well.append(iw_closest)
    print("Well %s is closest to data point index %d at distance %.2f m" % (W['name'], iw_closest, dists_to_well[iw_closest]))


# %%
with h5py.File(f_prior_h5, 'r') as f:
    # Read depth/position values (stored as 'x' attribute)
    z = f['M2'].attrs['x']
    # Read class identifiers
    class_id = f['M2'].attrs['class_id']
    class_name = f['M2'].attrs['class_name']
nm = len(z)
nclass = len(class_id)

# %% [markdown]
# ## Optionally inflate the noise in the tTEM data 

# %%
# inflateTEMNoise be be tested for values, 1,2,5,10

if inflateTEMNoise > 0:
    gf=inflateTEMNoise
    print("="*60)
    print("Increasing noise level (std) by a factor of %d" % gf)
    print("="*60)
    D = ig.load_data(f_data_h5)
    D_obs = D['d_obs'][0]
    D_std = D['d_std'][0]*gf
    f_data_old_h5 = f_data_h5
    # set new data file name, as f_data_h5, but append (before .h5) the gf value
    f_data_h5 = f_data_h5.replace('.h5', '_gf%g.h5' % gf)
    ig.copy_hdf5_file(f_data_old_h5, f_data_h5)
    ig.save_data_gaussian(D_obs, D_std=D_std, f_data_h5=f_data_h5, file_gex=file_gex)



# %%
# load prior im = 2


# %%
# Make prior data 'D2' that is simple and identity of the prior lithology type
#f_prior_h5 = ig.prior_data_identity(f_prior_h5, im=2, id=2, doMakePriorCopy=True)

# %%
# The fwell information can be interepreted and handles in defferent ways. 
# First lets conbsider the case where each entry in W defines the "probability that the most probable lithilogy in the defined inteval is a specific class"
#


# %% [markdown]
# ## Compute prior data
#

# %% [markdown]
# ### Compute prior tTEM data

# %% [markdown]
# ## Compute prior WELL 
# For each well W, a set of prior data mush be computed and store in f_prior_h5. The prior data must be computed in the exact way the 'observed data' are computed. 
#
# Say the priro data is written as f_prior['/D3'], then id_prior = 3, and say the observed is written to f_data['/D2/d_obs'], then the 'id_prior' mjst be set to 
#
#     f_prior['/D3']
#     f_data['/D2/d_obs']
#     f_data['/D2'].id_prior
#
#

# %%

# load the prior lithology data
M, idx = ig.load_prior_model(f_prior_h5)
M_lithology = M[1]
nm = M_lithology.shape[1]
nreal = M_lithology.shape[0]
z= np.arange(0,dmax,dz)

id_use_well=[]
P_obs_well=[]
for iw in np.arange(len(WELLS)):
    W = WELLS[iw]
    P_obs, lithology_mode = ig.compute_P_obs_sparse(M_lithology, 
                                                    z=z, 
                                                    class_id=class_id, 
                                                    W=W,
                                                    parallel=True,
                                                    showInfo=2)

    # Now we are ready to save and litholigy_mode to prior file P_obs to a data file, 
    # forst save the lithology_mode to a prior file as prior data, and get back the prior id of the prior data
    id_use = ig.save_prior_data(f_prior_h5, lithology_mode)

    id_use_well.append(id_use)
    P_obs_well.append(P_obs)

# %% [markdown]
# ### Now compute the observed data, both at the well location, and extrapolate thios observetion to the entire data grid

# %%
# This part can be rerun with different observed probabilities, without needing to recompute the prior data above.
for iw in np.arange(len(WELLS)):

    W = WELLS[iw]

    id_use = id_use_well[iw]
    P_obs = P_obs_well[iw]
    
    # apply P_obs to the whole data grid with distance based weighting
    d_obs, i_use = ig.Pobs_to_datagrid(P_obs, W['X'], W['Y'], f_data_h5, r_data=r_data, r_dis=r_dis, doPlot=False)


    # Next save the P_obs to a data file, with the correct prior id
    # Save P_obs to data file
    id_out, f_out = ig.save_data_multinomial(
        D_obs=d_obs,           # Shape: (nd, nclass, nm)
        i_use=i_use,           # Shape: (nd, 1) - binary mask
        id_prior=id_use,       # Which PRIOR data to use (D2 from prior file)
        f_data_h5=f_data_h5,   # Output file
        showInfo=1             # Verbosity
    )
    print("Wrote data id %d to file: %s" % (id_out, f_out))

    plt.figure()
    plt.imshow(P_obs)
    plt.xlabel('Number for "observed" lithology')
    plt.ylabel('Class ID')
    plt.title('P_obs for lithology observations')

# %% [markdown]
# ## Inversion
# The data is now ready for inversion with the rejection sampler.
#
# On total we have 3 data types (one tTEM and two WellLog). They can be all jointly inverted (the default) or one can select which data types to ínver using `id_use`
#
#     id_use = [1] # tTEM 
#     id_use = [2] # Well 1
#     id_use = [3] # Well 2
#     id_use = [2,3] # Wells 1,2
#     id_use = [1,2,3] # tTEM, Wells 1,2 (the default if id_use is not set)
#

# %%
# This prt of the can be rerun using different selection of data types without rerunning the abobe parts
nr=1000
id_use = [1] # tTEM 
id_use = [2] # Well 1
#id_use = [3] # Well 2
id_use = [2,3] # Well 1,2
id_use = [1,2,3] # tTEM, Well 1,2


# get string from id_use
fileparts = os.path.splitext(f_data_h5)
id_use_str = '_'.join(map(str, id_use))
f_post_h5 = 'post_%s_NoiseGain%d_id%s.h5' % (fileparts[0], inflateTEMNoise, id_use_str)
f_post_h5 = ig.integrate_rejection(f_prior_h5, 
                                f_data_h5, 
                                f_post_h5, 
                                showInfo=0, 
                                N_use = N_use,
                                id_use = id_use,
                                ip_range = id_line,
                                nr=nr,
                                parallel=parallel, 
                                autoT=True,
                                T_base=1,
                                updatePostStat=True)
ig.plot_profile(f_post_h5, im=1, ii=id_line, gap_threshold=50, xaxis='y', hardcopy=hardcopy, alpha = 1,std_min = 0.3, std_max = 0.6)

# %%

# %%
ig.plot_profile(f_post_h5, im=1, ii=id_line, gap_threshold=50, xaxis='y', hardcopy=hardcopy, alpha = 1,std_min = 0.3, std_max = 0.6)
ig.plot_profile(f_post_h5, im=2, ii=id_line, gap_threshold=50, xaxis='y', hardcopy=hardcopy, alpha=1, entropy_min =0.4, entropy_max=1.0)

# %%
for iw in np.arange(len(WELLS)):
    ig.plot_data_prior_post(f_post_h5, i_plot=i_well[iw], title=WELLS[iw]['name'], hardcopy=hardcopy)


# %%
t_end = time.time()
print("Total computation time: %.2f seconds\nTotal computation time: %.2f minutes\nTotal computation time: %.2f hours" % (t_end - t0, (t_end - t0)/60.0, (t_end - t0)/3600.0))


# %%
