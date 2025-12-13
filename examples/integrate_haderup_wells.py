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
    if file.startswith('haderup') and 'IDEN' in file and file.endswith('.h5'):
        print("Removing existing file: %s" % file)
        os.remove(file)

t0 = time.time()

# %%
P_single = 0.9
N=1000000
N_use = N
dmax=90
dz=1

# Inflated noise std by this factor
inflateTEMNoise = 2

# Extrapolation options for distance weighting
r_data=2 # XY-distance based weight for extrapolating borehole information to the data grid
r_dis=100 # DATA-distance based weight for extrapolating borehole information to the data grid 
r_dis=300


# %%
# Get Daugaard data files
case = 'HADERUP'
files = ig.get_case_data(case=case, 
                         loadAll=True, 
                         filelist=['HADERUP_MEAN_ALL_cleaned.h5','TX07_Haderup_mean.gex'])

f_data_h5 = 'HADERUP_MEAN_ALL_cleaned.h5'
#f_data_h5 = 'HADERUP_MEAN_ALL.h5'
f_prior_xls = 'prior_haderup_dec25.xlsx'
file_gex= ig.get_gex_file_from_data(f_data_h5)

print("Using data file: %s" % f_data_h5)
print("Using GEX file: %s" % file_gex)


# %% [markdown]
# ## Optionally inflate the noise in the tTEM data 
# Make a copy of the tTEM data, and oprionally increase noise level

# %%
# inflateTEMNoise be be tested for values, 1,2,5,10
# gain factor
gf=inflateTEMNoise
    
# Make a copy of the tTEM data
# set new data file name, as f_data_h5, but append (before .h5) the gf value
f_data_old_h5 = f_data_h5
f_data_h5 = f_data_h5.replace('.h5', '_gf%g.h5' % gf)
ig.copy_hdf5_file(f_data_old_h5, f_data_h5)

if inflateTEMNoise != 1:
    # Optinally inflate the noise
    print("="*60)
    print("Increasing noise level (std) by a factor of %d" % gf)
    print("="*60)
    D = ig.load_data(f_data_h5)
    D_obs = D['d_obs'][0]
    D_std = D['d_std'][0]*gf
    ig.copy_hdf5_file(f_data_old_h5, f_data_h5)
    ig.save_data_gaussian(D_obs, D_std=D_std, f_data_h5=f_data_h5, file_gex=file_gex)




# %% [markdown]
# ## Define the information from the WELLs
# 
# Information from well logs is provided through the W dictionar. Each well W contains:
# * X: X coordinate of the well (float)
# * Y: Y coordinate of the well (float)
# * name: Name of the well (string)  
# * depth_top: Top depth of each lithology interval (list)
# * depth_bottom: Bottom depth of each lithology interval (list)
#
# It can also contain information about discrete model parameters (such as lithology)

# * class_prob: Probability of the observed class in each interval (list or array)
# * class_obs: Observed class in each interval (list or array)
#
# It can also contain information about continiuous parameters (such as resistivity)
# * d_obs: observed continuous parameter in each interval (list or array)
# * d_std: standard deviation of the observed continuous parameter in each interval (list or array)
# * Cd: covariance matrix between the observed continuous parameters (2D array) (optional)
#
# In addtion to this information, one need to define how the well infomration interpretation is quantified.
#
# For example, 
# If each lithololy observation reflect the "probability that a specfific class id (lithology) 
# is the most probable class in the defined interval" one one 
# * method = 'mode_probablity'
#
# If all layhwers withing layer boundaries has to have the same lithology, then
# # * method = 'class_exact'
#
#   
# If each lithology observation reflect "the probability in each layer (as defined in the prior) of specific class id (lithology)" then
# * method = 'layer_probability'
#
# If each lithology observation reflect "the class probability that all layers within the top and bootom has a specific class" then
# * method = 'layer_probability_independent'
#
#
#

# %%
WELLS = []

W = {}
W['depth_top'] =     [0,  8, 12, 16, 34]
W['depth_bottom'] =  [8, 12, 16, 28, 36]
W['class_obs'] = [1,  2,  1,  5,  4]
W['class_prob'] = np.array([1.0, 1.0, 1.0, 1.0, 1.0])*P_single
W['X'] = 498832.5
W['Y'] = 6250843.1
W['name'] = '65. 795'
W['method'] = 'mode_probability'
WELLS.append(W)

W = {}
W['depth_top'] =     [0,  9.8, 15.6, 17.8, 36.2, 39]
W['depth_bottom'] =  [9.8, 15.6, 17.8, 36.2, 39, 45.2]
W['class_obs'] = [1,  2,  1,  4,  5,  4]
W['class_prob'] = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])*P_single
W['X'] = 498804.95
W['Y'] = 6249940.89
W['name'] = '65. 732'
W['method'] = 'mode_probability'  # FASTER MORE ROBUST
W['method'] = 'layer_probability'  # SLOWER AND TYOICALLY NOT REPRESENTATIVE OF ACTUAL INFORMATION

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
    print("Using existing f_prior_h5=%s" % (f_prior_h5))

f_prior_h5_old = f_prior_h5
f_prior_h5 = f_prior_h5.replace('.h5', '_copy.h5')
ig.copy_hdf5_file(f_prior_h5_old, f_prior_h5, showInfo=2)



# %% [markdown]
# ## Select a profile going through the wells

# %%
# The the X and Y locations for all data points
X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)


# Find the indexes of data points closest to the well locations
i_well = []
i_well_dist = []
for iw in np.arange(len(WELLS)):
    W = WELLS[iw]
    dists_to_well = np.sqrt((X - W['X'])**2 + (Y - W['Y'])**2)
    iw_closest = np.argmin(dists_to_well)
    i_well.append(iw_closest)
    i_well_dist.append(dists_to_well[iw_closest])
    print("Well %s is closest to data point index %d at distance %.2f m" % (W['name'], i_well[-1], i_well_dist[-1]))

# Find the index of data locations along a profile

# Select points along a profile line
X1 = WELLS[0]['X']+10; Y1=WELLS[0]['Y']+300;
X2= WELLS[1]['X']+0;   Y2=WELLS[1]['Y']+0;
X3= WELLS[1]['X']-140;   Y3=WELLS[1]['Y']-600;

# Find indexes, id_line, of data points along points (Xl,Yl), within buffer distance
Xl = np.array([X1, X2, X3])
Yl = np.array([Y1, Y2, Y3])
buffer = 15.0
id_line, distances, segment_ids = ig.find_points_along_line_segments(
    X, Y, Xl, Yl, tolerance=buffer
)

with h5py.File(f_data_h5,'r') as f_data:
    # find number of nan values on d_obs
    NON_NAN = np.sum(~np.isnan(f_data['/%s' % 'D1']['d_obs']), axis=1)

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
# Make prior data 'D2' that is simple and identity of the prior lithology type
#f_prior_h5 = ig.prior_data_identity(f_prior_h5, im=2, id=2, doMakePriorCopy=True)


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
'''
    1 Compute prior data lithology mode for well W, and save to f_prior_h5
    2 Update f_prior_h5 with prior data, return prior data id as id_prior
'''
def prior_data_welllog_class_mode(f_prior_h5, im_prior = None, W=None, parallel=False, **kwargs):
    '''
        1 Compute prior data lithology mode for well W, and save to f_prior_h5
        2 Update f_prior_h5 with prior data, return prior data id as id_prior
    '''
    showInfo = kwargs.get('showInfo', 1)

    with h5py.File(f_prior_h5, 'r') as f:
        # Read depth/position values (stored as 'x' attribute)
        z = f['M%d' % im_prior].attrs['x']
        # Read class identifiers
        class_id = f['M%d' % im_prior].attrs['class_id']
        class_name = f['M%d' % im_prior].attrs['class_name']
        M_lithology = f['M%d' % im_prior][:]

    P_obs, class_mode = ig.welllog_compute_P_obs_class_mode(M_lithology,
                                                             z=z,
                                                             class_id=class_id,
                                                             W=W,
                                                             parallel=parallel, showInfo=showInfo)
    
    id_prior = ig.save_prior_data(f_prior_h5, class_mode, showInfo=showInfo)
    
    return P_obs, id_prior



def prior_data_welllog_class_layer(f_prior_h5, im_prior = None, W=None, parallel=False, **kwargs):
    
    showInfo = kwargs.get('showInfo', 1)

    # IF W is none returnnwith a message
    if W is None:
        print("No well information provided, returning None")
        return None, None

    with h5py.File(f_prior_h5, 'r') as f:
        # Read depth/position values (stored as 'x' attribute)
        z = f['M%d' % im_prior].attrs['x']
        # Read class identifiers
        class_id = f['M%d' % im_prior].attrs['class_id']
        
    P_obs = ig.compute_P_obs_discrete(z=z, class_id=class_id, W=W)

    f_prior_h5, id_prior = ig.prior_data_identity(f_prior_h5, im=im_prior, doMakePriorCopy=False, showInfo=showInfo)
    
    return P_obs, id_prior



def prior_data_welllog(f_prior_h5, im_prior = None, W=None, parallel=False, **kwargs):
    
    showInfo = kwargs.get('showInfo', 1)

    # if W is none return
    if W is None:
        return None, None
    
    
    #if W['method'] does not exist , set it to 'mode_probability' 
    if 'method' not in W:
        W['method'] = 'mode_probability'

    if showInfo>0:
        print("Computing prior data for well: %s using method: %s" % (W['name'], W.get('method', 'mode_probability')))

    if W['method'] == 'mode_probability':
        P_obs, id_prior = prior_data_welllog_class_mode(f_prior_h5=f_prior_h5, im_prior=im_prior, W=W, parallel=parallel, showInfo=showInfo)
    elif W['method'] == 'layer_probability':        
        P_obs, id_prior = prior_data_welllog_class_layer(f_prior_h5, im_prior=im_prior, W=W, parallel=parallel, showInfo=showInfo)        
    elif W['method'] == 'layer_probability_independent':
        # To be implemented
        raise NotImplementedError("Method 'layer_probability_independent' not implemented yet")
    else:
        raise ValueError("Unknown method: %s" % W['method'])

    return P_obs, id_prior


# %% 

im_prior = 2  # lithology
id_prior_well=[]
P_obs_well=[]
for iw in np.arange(len(WELLS)):
    W = WELLS[iw]
    P_obs, id_prior = prior_data_welllog(f_prior_h5=f_prior_h5, im_prior=im_prior, W=W, parallel=parallel, showInfo=0)
    id_prior_well.append(id_prior)
    P_obs_well.append(P_obs)



# %% [markdown]
# ### Now compute the observed data, both at the well location, and extrapolate thios observetion to the entire data grid

# %%
# This part can be rerun with different observed probabilities, without needing to recompute the prior data above.
for iw in np.arange(len(WELLS)):

    W = WELLS[iw]

    id_prior = id_prior_well[iw]
    P_obs = P_obs_well[iw]
    
    # apply P_obs to the whole data grid with distance based weighting
    d_obs, i_use, T_use = ig.Pobs_to_datagrid(P_obs, W['X'], W['Y'], f_data_h5, r_data=r_data, r_dis=r_dis, doPlot=True)

    # Next save the P_obs to a data file, with the correct prior id
    # Save P_obs to data file
    id_out, f_out = ig.save_data_multinomial(
        D_obs=d_obs,           # Shape: (nd, nclass, nm)
        i_use=i_use,           # Shape: (nd, 1) - binary mask
        id_prior=id_prior,       # Which PRIOR data to use (D2 from prior file)
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
#id_use = [2] # Well 1
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
                                showInfo=1, 
                                N_use = N_use,
                                id_use = id_use,
                                ip_range = id_line,
                                nr=nr,
                                parallel=parallel, 
                                autoT=True,
                                T_base=1,
                                updatePostStat=True)


#%%
ig.plot_profile(f_post_h5, im=1, ii=id_line, gap_threshold=50, xaxis='y', hardcopy=hardcopy, alpha = 1,std_min = 0.3, std_max = 0.6)
ig.plot_profile(f_post_h5, im=2, ii=id_line, gap_threshold=50, xaxis='y', hardcopy=hardcopy, alpha=1, entropy_min =0.3, entropy_max=0.6)

# %%
for iw in np.arange(len(WELLS)):
    ig.plot_data_prior_post(f_post_h5, i_plot=i_well[iw], title=WELLS[iw]['name'], hardcopy=hardcopy)


# %%
t_end = time.time()
print("Total computation time: %.2f seconds\nTotal computation time: %.2f minutes\nTotal computation time: %.2f hours" % (t_end - t0, (t_end - t0)/60.0, (t_end - t0)/3600.0))


# %%
ig.plot_feature_2d(f_post_h5,im=1,iz=25, key='LogMean', uselog=1, cmap='jet', hardcopy=hardcopy, clim=[10, 200])
plt.show()
ig.plot_feature_2d(f_post_h5,im=2,iz=25, key='Mode', uselog=1, cmap='jet', hardcopy=hardcopy, clim=[.5, 6.5])
plt.show()

# %%
""" 
# get string from id_use
N_use = 50000
fileparts = os.path.splitext(f_data_h5)
id_use_str = '_full_'.join(map(str, id_use))
f_post_h5 = 'post_%s_NoiseGain%d_id%s.h5' % (fileparts[0], inflateTEMNoise, id_use_str)
f_post_h5 = ig.integrate_rejection(f_prior_h5, 
                                f_data_h5, 
                                f_post_h5, 
                                showInfo=0, 
                                N_use = N_use,
                                nr=nr,
                                id_use = [2,3],
                                parallel=parallel, 
                                autoT=True,
                                T_base=1,
                                updatePostStat=True)
ig.plot_feature_2d(f_post_h5,im=1,iz=25, key='LogMean', uselog=1, cmap='jet', hardcopy=hardcopy, clim=[10, 200])
plt.show()
ig.plot_feature_2d(f_post_h5,im=2,iz=25, key='Mode', uselog=1, cmap='jet', hardcopy=hardcopy, clim=[.5, 6.5])
plt.show()

 """

# %% Manual Test
'''
if __name__ == "__main__": # needed in Windows for parallel processing
    f_data_h5 = 'HADERUP_MEAN_ALL_cleaned_gf2.h5'
    f_prior_h5 = 'haderup_N1000000_dmax90_dz1_copy.h5'
    N_use = 10000
    id_use = [1] # tTEM 
    id_use = [2] # WELL 2
    id_use = [2,3] # WELLS 1 and 2
    # id_use = [1,2,3] # tTEM + WELLS 1 and 2 [Default if id_use not set]

    f_post_h5 = ig.integrate_rejection(f_prior_h5, 
                                    f_data_h5, 
                                    N_use = N_use,
                                    id_use = id_use,
                                    nr=nr,
                                    parallel=True, 
                                    autoT=True,
                                    T_base=1,
                                    updatePostStat=True)    


    ig.plot_feature_2d(f_post_h5,im=2,iz=25, key='Mode', uselog=1, cmap='jet', hardcopy=hardcopy, clim=[.5, 6.5])
    plt.show()
    ig.plot_feature_2d(f_post_h5,im=1,iz=25, key='Median', uselog=1, cmap='jet', hardcopy=hardcopy, clim=[1, 300])
    plt.show()




'''