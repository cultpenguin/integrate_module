#!/usr/bin/env python
# %% [markdown]
# # Daugaard example with multiple data sets (dual moment and two boreholes)
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
import os
import integrate as ig
# check if parallel computations can be performed
parallel = ig.use_parallel(showInfo=1)

import h5py
import numpy as np
import matplotlib.pyplot as plt
hardcopy=True

# remove all files with name 'da*IDEN*h5'
for file in os.listdir('.'):
    if file.startswith('daugaard') and 'IDEN' in file and file.endswith('.h5'):
        print("Removing existing file: %s" % file)
        os.remove(file)

def compute_P_obs_discrete(depth_top, depth_bottom, lithology_obs, z, class_id, P_single=0.8, P_prior=None):
    """
    Compute discrete observation probability matrix from depth intervals and lithology observations.
    
    This function creates a probability matrix where each depth point is assigned 
    probabilities based on observed lithology classes within specified depth intervals.
    
    Parameters
    ----------
    depth_top : array-like
        Array of top depths for each observation interval.
    depth_bottom : array-like
        Array of bottom depths for each observation interval.
    lithology_obs : array-like
        Array of observed lithology class IDs for each interval.
    z : array-like
        Array of depth/position values where probabilities are computed.
    class_id : array-like
        Array of unique class identifiers (e.g., [0, 1, 2] for 3 lithology types).
    P_single : float, optional
        Probability assigned to the observed class. Default is 0.8.
    P_prior : ndarray, optional
        Prior probability matrix of shape (nclass, nm). If None, uses uniform distribution
        for depths not covered by observations. Default is None.
    
    Returns
    -------
    P_obs : ndarray
        Probability matrix of shape (nclass, nm) where nclass is the number of classes
        and nm is the number of depth points. For each depth point covered by observations,
        the observed class gets probability P_single and other classes share (1-P_single).
        Depths not covered by any observation contain NaN or prior probabilities if provided.
    
    Examples
    --------
    >>> depth_top = [0, 10, 20]
    >>> depth_bottom = [10, 20, 30]
    >>> lithology_obs = [1, 2, 1]  # clay, sand, clay
    >>> z = np.arange(30)
    >>> class_id = [0, 1, 2]  # gravel, clay, sand
    >>> P_obs = compute_P_obs_discrete(depth_top, depth_bottom, lithology_obs, z, class_id)
    >>> print(P_obs.shape)  # (3, 30)
    """
    import numpy as np
    
    nm = len(z)
    nclass = len(class_id)
    
    # Compute probability for non-hit classes
    P_nohit = (1 - P_single) / (nclass - 1)
    
    # Initialize with NaN or prior
    if P_prior is not None:
        P_obs = P_prior.copy()
    else:
        P_obs = np.zeros((nclass, nm)) * np.nan
    
    # Loop through each depth point
    for im in range(nm):
        # Loop through each observation interval
        for i in range(len(depth_top)):
            # Check if current depth is within this interval
            if z[im] >= depth_top[i] and z[im] < depth_bottom[i]:
                # Assign probabilities for all classes
                for ic in range(nclass):
                    if class_id[ic] == lithology_obs[i]:
                        P_obs[ic, im] = P_single
                    else: 
                        P_obs[ic, im] = P_nohit
    
    return P_obs

def rescale_P_obs_temperature(P_obs, T=1.0):
    """
    Rescale discrete observation probabilities by temperature and renormalize.

    This function applies temperature annealing to probability distributions by raising
    each probability to the power (1/T), then renormalizing each column (depth point)
    so that probabilities sum to 1. Higher temperatures (T > 1) flatten the distribution,
    while lower temperatures (T < 1) sharpen it.

    Parameters
    ----------
    P_obs : ndarray
        Probability matrix of shape (nclass, nm) where nclass is the number of classes
        and nm is the number of model parameters (e.g., depth points).
        Each column should represent a probability distribution over classes.
    T : float, optional
        Temperature parameter for annealing. Default is 1.0 (no scaling).
        - T = 1.0: No change (original probabilities)
        - T > 1.0: Flattens distribution (less certain)
        - T < 1.0: Sharpens distribution (more certain)
        - T → ∞: Approaches uniform distribution
        - T → 0: Approaches one-hot distribution

    Returns
    -------
    P_obs_scaled : ndarray
        Temperature-scaled and renormalized probability matrix of shape (nclass, nm).
        Each column sums to 1.0. NaN values in input are preserved in output.

    Examples
    --------
    >>> P_obs = np.array([[0.8, 0.6, 0.5],
    ...                   [0.1, 0.2, 0.3],
    ...                   [0.1, 0.2, 0.2]])
    >>> P_scaled = rescale_P_obs_temperature(P_obs, T=2.0)
    >>> print(P_scaled)  # More uniform distribution
    >>> P_scaled = rescale_P_obs_temperature(P_obs, T=0.5)
    >>> print(P_scaled)  # Sharper distribution

    Notes
    -----
    The temperature scaling follows the Boltzmann distribution:
        P_new(c) ∝ P_old(c)^(1/T)

    After scaling, each column (depth point) is renormalized:
        P_new(c) = P_new(c) / sum_c(P_new(c))

    This is commonly used in simulated annealing and rejection sampling to control
    the strength of discrete observations during Bayesian inference.
    """
    import numpy as np

    # Copy to avoid modifying the original
    P_obs_scaled = P_obs.copy()

    # Get shape
    nclass, nm = P_obs.shape

    # Apply temperature scaling: p^(1/T)
    # Handle special case where T=1 (no scaling needed)
    if T != 1.0:
        P_obs_scaled = np.power(P_obs_scaled, 1.0 / T)

    # Renormalize each column (each depth point) to sum to 1
    for im in range(nm):
        col_sum = np.nansum(P_obs_scaled[:, im])

        # Only renormalize if the sum is non-zero and not NaN
        if col_sum > 0 and not np.isnan(col_sum):
            P_obs_scaled[:, im] = P_obs_scaled[:, im] / col_sum

    return P_obs_scaled

def Pobs_to_datagrid(P_obs, X, Y, f_data_h5, r_data=10, r_dis=100, doPlot=False):
    """
    Convert point-based discrete probability observations to gridded data with distance-based weighting.

    This function distributes discrete probability observations (e.g., from a borehole) across
    a spatial grid using distance-based weighting. Observations at location (X, Y) are applied
    to nearby grid points with decreasing influence based on distance. Temperature annealing
    is used to reduce the strength of observations far from the source point.

    Parameters
    ----------
    P_obs : ndarray
        Probability matrix of shape (nclass, nm) where nclass is the number of classes
        and nm is the number of model parameters (e.g., depth points).
        Each column represents a probability distribution over discrete classes.
    X : float
        X coordinate (e.g., UTM Easting) of the observation point.
    Y : float
        Y coordinate (e.g., UTM Northing) of the observation point.
    f_data_h5 : str
        Path to HDF5 data file containing survey geometry (X, Y coordinates).
    r_data : float, optional
        Inner radius in meters within which observations have full strength.
        Default is 10 meters.
    r_dis : float, optional
        Outer radius in meters for distance-based weighting. Beyond this distance,
        observations are fully attenuated (temperature → ∞). Default is 100 meters.
    doPlot : bool, optional
        If True, creates diagnostic plots showing weight distributions.
        Default is False.

    Returns
    -------
    d_obs : ndarray
        Gridded observation data of shape (nd, nclass, nm) where nd is the number
        of spatial locations in the survey. Each location gets temperature-scaled
        probabilities based on distance from (X, Y).
    i_use : ndarray
        Binary mask of shape (nd, 1) indicating which grid points should be used
        (1) or ignored (0) in the inversion. Points with temperature < 100 are used.

    Notes
    -----
    The function uses distance-based temperature annealing:
    1. Computes distance-based weights using `get_weight_from_position()`
    2. Converts distance weight to temperature: T = 1 / w_dis
    3. Caps maximum temperature at 100 (very weak influence)
    4. For each grid point:
       - If T < 100: include point (i_use=1) and apply temperature scaling
       - If T ≥ 100: exclude point (i_use=0) and set observations to NaN

    Temperature scaling reduces probability certainty with distance:
    - T = 1 (close to observation): Original probabilities preserved
    - T > 1 (far from observation): Probabilities become more uniform
    - T ≥ 100 (very far): Observations effectively ignored

    Examples
    --------
    >>> # Borehole observation at specific location
    >>> P_obs = compute_P_obs_discrete(depth_top, depth_bottom, lithology, z, class_id)
    >>> X_well, Y_well = 543000.0, 6175800.0
    >>> d_obs, i_use = Pobs_to_datagrid(P_obs, X_well, Y_well, 'survey_data.h5',
    ...                                  r_data=10, r_dis=100)
    >>> # Write to data file
    >>> ig.write_data_multinomial(d_obs, i_use=i_use, id=2, f_data_h5='survey_data.h5')

    See Also
    --------
    rescale_P_obs_temperature : Temperature scaling function
    compute_P_obs_discrete : Create P_obs from depth intervals
    get_weight_from_position : Distance-based weighting function
    """
    import numpy as np
    import integrate as ig

    # Get grid dimensions from data file
    X_grid, Y_grid, _, _ = ig.get_geometry(f_data_h5)
    nd = len(X_grid)
    nclass, nm = P_obs.shape

    # Initialize output arrays
    i_use = np.zeros((nd, 1))
    d_obs = np.zeros((nd, nclass, nm)) * np.nan

    # Compute distance-based weights for all grid points
    w_combined, w_dis, w_data, i_use_from_func = ig.get_weight_from_position(
        f_data_h5, X, Y, r_data=r_data, r_dis=r_dis, doPlot=doPlot
    )

    # Convert distance weight to temperature
    # w_dis is 1 at observation point, decreases with distance
    # T = 1/w_dis means T increases with distance (weaker influence)
    T_all = 1 / w_dis

    # Cap maximum temperature at 100 (beyond this, observation has negligible effect)
    T_all[T_all > 100] = 100

    # Apply temperature scaling to each grid point
    for ip in np.arange(nd):
        T = T_all[ip]

        # Only use points where temperature is reasonable (< 100)
        if T < 100:
            i_use[ip] = 1
            # Scale probabilities based on distance (higher T = more uniform distribution)
            P_obs_local = rescale_P_obs_temperature(P_obs, T=T)
            d_obs[ip, :, :] = P_obs_local
        # else: i_use[ip] = 0 and d_obs[ip] stays NaN

    return d_obs, i_use

# %%
P_single=0.99
inflateTEMNoise = 4
N_use = 100000

case = 'DAUGAARD'
files = ig.get_case_data(case=case)
f_data_h5 = files[0]
f_data_h5 = 'DAUGAARD_AVG.h5'
file_gex= ig.get_gex_file_from_data(f_data_h5)

print("Using data file: %s" % f_data_h5)
print("Using GEX file: %s" % file_gex)
X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)
nd= len(X)
nclass = 8 # optionally get from data
nm = 90 # optionally get from data


with h5py.File(f_data_h5,'r') as f_data:
    # find number of nan values on d_obs
    NON_NAN = np.sum(~np.isnan(f_data['/%s' % 'D1']['d_obs']), axis=1)

# select the profile line
W1_X = 542983.01;W1_Y = 6175822.76;W1_name = 'DAU'  
W2_X = 543584.098;W2_Y = 6175788.478;W2_name = '116.1602'
X1 = W1_X-100; 
Y1=W1_Y;
X2= W2_X+1800; 
Y2=W2_Y-100;

# Find points within buffer distance
Xl = np.array([X1, X2])
Yl = np.array([Y1, Y2])
buffer = 10.0
indices, distances, segment_ids = ig.find_points_along_line_segments(
    X, Y, Xl, Yl, tolerance=buffer
)
id_line = indices

plt.figure(figsize=(10, 6))
plt.scatter(X, Y, c=NON_NAN, s=1,label='Survey Points')
#plt.plot(X[id_line],Y[id_line], 'k-', markersize=8, label='Profile', zorder=2, linewidth=5)
plt.plot(X[id_line],Y[id_line], 'r.', markersize=8, label='Profile', zorder=2, linewidth=5)
plt.grid()
plt.plot(W1_X,W1_Y,'k*', markersize=15, label=W1_name, zorder=3)
plt.plot(W2_X,W2_Y,'k*', markersize=10, label=W2_name, zorder=3)
plt.colorbar(label='Elevation (m)')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Survey Points Colored by Number of Non-NaN Data Points')
plt.axis('equal')
plt.legend()
if hardcopy:
    plt.savefig('DAUGAARD_survey_points_nonnan.png', dpi=300)
plt.show()




#%% Optionally merge prior 

f_prior_data_merged_full_h5 = 'daugaard_merged.h5'

# Check if merged prior already exists
if os.path.exists(f_prior_data_merged_full_h5):
    print("Merged prior file %s already exists. Using it." % f_prior_data_merged_full_h5)
else:
    print("Merged prior file %s does not exist. Merging priors." % f_prior_data_merged_full_h5)

    f_prior_data_h5_list = []
    f_prior_data_h5_list.append('daugaard_valley_new_N1000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5')
    f_prior_data_h5_list.append('daugaard_standard_new_N1000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5')

    f_prior_data_merged_full_h5 = ig.merge_prior(f_prior_data_h5_list, f_prior_merged_h5='f_prior_data_merged_full.h5', showInfo=2)

if N_use >0:

    f_prior_data_h5 = 'daugaard_merged_N%d.h5' % N_use
    if not os.path.exists(f_prior_data_h5):
        print("Creating prior file with %d samples: %s" % (N_use, f_prior_data_h5))
        ig.copy_prior(f_prior_data_merged_full_h5, f_prior_data_h5, N_use=N_use, showInfo=2)
else:   
    f_prior_data_h5 = f_prior_data_merged_full_h5
    print("Using full prior file: %s" % f_prior_data_h5)

with h5py.File(f_prior_data_h5, 'r') as f:
    # Read depth/position values (stored as 'x' attribute)
    z = f['M2'].attrs['x']
    # Read class identifiers
    class_id = f['M2'].attrs['class_id']
    class_name = f['M2'].attrs['class_name']
nm = len(z)
nclass = len(class_id)


# %% Load Dauagard data and increase std by a factor of 3
# inflateTEMNoise be be tested for values, 1,2,5,10

#if inflateTEMNoise != 1:
if inflateTEMNoise > 0:
    gf=inflateTEMNoise
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


#%% DESCRIBE THE WELL DATA
# Define wells information in different ways. 
# Frist deifne what direct information from well log
WELLS=[]
WELLS_compressed=[]

'''
TOP 	BOTTOM	CLASS	UTMX	UTMY	8-class prior
0	0,3	Muld	542983,01	6175822,76	2
0,3	0,5	Sand	542983,01	6175822,76	2
0,5	1	Silt	542983,01	6175822,76	2
1	1,5	Sand	542983,01	6175822,76	2
1,5	2	Silt	542983,01	6175822,76	2
2	10	Sand	542983,01	6175822,76	2
10	10,5	Grus	542983,01	6175822,76	5
10,5	13,2	Sand	542983,01	6175822,76	2
13,2	16,6	Grus	542983,01	6175822,76	5
16,6	20	Moræneler	542983,01	6175822,76	3
'''
W = {}
W['depth_top'] =    [0  , 0.3, 0.5, 1, 1.5, 2, 10, 10.5, 13.2, 16.6]
W['depth_bottom'] = [0.3, 0.5, 1, 1.5, 2, 10, 10.5, 13.2, 16.6, 20]
W['lithology_obs'] = [2, 2, 2, 2, 2, 2, 5, 2, 5, 3]
W['X'] = 542983.01
W['Y'] = 6175822.76
W['name'] = 'DAU02'
WELLS.append(W)  

import copy
W_compressed = copy.deepcopy(W)
W_compressed['depth_top'] =    [0   , 13.2, 16.6]
W_compressed['depth_bottom'] = [13.2, 16.6, 20]
W_compressed['lithology_obs'] = [2, 5, 3]
WELLS.append(W_compressed)

'''
TOP 	BOTTOM	CLASS	UTMX	UTMY	8-class prior 
0	8	Moræneler	543584,098	6175788,478	3
8	15	Grus	543584,098	6175788,478	5
15	17	Moræneler	543584,098	6175788,478	3
17	20	Grus	543584,098	6175788,478	5
20	23,5	Grus	543584,098	6175788,478	5
23,5	45	Miocæn sand	543584,098	6175788,478	6
45	46	Miocæn ler	543584,098	6175788,478	7
'''
W = {}
W['depth_top'] =     [0,  8, 15, 17, 20, 23.5, 45]
W['depth_bottom'] =  [8, 15, 17, 20, 23.5, 45, 46]
W['lithology_obs'] = [3,  5,  3,  5,  5,  6,  7]
W['X'] = 543584.098
W['Y'] = 6175788.478
W['name'] = '116.1602'                      
WELLS.append(W)

W_compressed = copy.deepcopy(W)
W_compressed['depth_top'] =     [0,  8, 15,   17, 23.5, 45]
W_compressed['depth_bottom'] =  [8, 15, 17, 23.5,   45, 46]
W_compressed['lithology_obs'] = [3,  5,  3,    5,    6, 7]
WELLS.append(W_compressed)

# SINGLE LAYER 
W = {}
W['depth_top'] =     [20]
W['depth_bottom'] = [24]
W['lithology_obs'] = [5] 
W['X'] = 542983.01
W['Y'] = 6175822.76
W['name'] = 'DAU02'
WELLS.append(W.copy())

W['X'] = 543584.098
W['Y'] = 6175788.478
WELLS.append(W.copy())

# %% Write different types of data
# load prior im = 2


#%%  First we add a data set that is simple the lithology at the well position
# Make prior data 'D2' that is simple and identity of the prior lithology type
f_prior_data_h5 = ig.prior_data_identity(f_prior_data_h5, im=2, id=2, doMakePriorCopy=True)

#%%

for iw in np.arange(len(WELLS)):
    print("considering well %d: %s" % (iw+1, WELLS[iw]['name']))

    W = WELLS[iw]
    depth_top = W['depth_top']
    depth_bottom = W['depth_bottom']
    lithology_obs = W['lithology_obs']
    X_well = W['X']
    Y_well = W['Y']
    P_obs = compute_P_obs_discrete(depth_top, depth_bottom, lithology_obs, z, class_id, P_single=P_single, P_prior=None)
    plt.figure()
    plt.imshow(P_obs)

    # apply P_obs to the whole data grid with distance based weighting
    d_obs, i_use = Pobs_to_datagrid(P_obs, X_well, Y_well, f_data_h5, r_data=10, r_dis=100, doPlot=False)

    #% Write to DATA file
    id_out, f_out = ig.write_data_multinomial(
        D_obs=d_obs,           # Shape: (nd, nclass, nm)
        i_use=i_use,           # Shape: (nd, 1) - binary mask
        id=2+iw,                  # Data ID (will create /D2/ in DATA.h5)
        id_use=2,              # Which PRIOR data to use (D2 from prior file)
        f_data_h5=f_data_h5,   # Output file
        showInfo=1             # Verbosity
    )

    plt.show()

# %% TEST
'''
Dall, idx = ig.load_prior_data(f_prior_data_h5)
Mall, idx = ig.load_prior_model(f_prior_data_h5)
DOBS = ig.load_data(f_data_h5)

id_test = 4
#P_obs = DOBS['d_obs'][1][id_line[80]]
P_obs = DOBS['d_obs'][id_test][id_line[40]]
D = Dall[1]
logL =  ig.likelihood_multinomial(D, P_obs, class_id=class_id)
plt.figure()
plt.subplot(2,1,1)
plt.plot(np.exp(logL[::1]-np.max(logL)),'.')
plt.ylim(-.1,1.1)
plt.grid()
plt.ylabel('Likelihood - normalized')
plt.subplot(2,1,2)
plt.plot(np.exp(logL[::5]),'.')
plt.ylabel('log-Likelihood')
plt.suptitle('id_test=%d' % id_test)
'''

#%% Now use a sparse representation of the well log data. 
# Instead of reprsentiug all nz model paraeters, we cnvert the log obserbavtion to 
# represent the actual number of layer obverservation.
# This we need to implement a function that goes through a reaælizations 
# of the prior and returns the lithology at each depth intervals, and we nee to deice how we do that.
#

# load the prior lithology data
M, idx = ig.load_prior_model(f_prior_data_h5)
M_lithology = M[1]
nm = M_lithology.shape[1]
nreal = M_lithology.shape[0]

for iw in np.arange(4):
    #iw = 0

    lithology_obs = WELLS[iw]['lithology_obs']
    depth_top = WELLS[iw]['depth_top']
    depth_bottom = WELLS[iw]['depth_bottom']
    X_well = WELLS[iw]['X']
    Y_well = WELLS[iw]['Y']
    nl=len(lithology_obs)
    lithology_mode = np.zeros((nreal, nl), dtype=int)

    from tqdm import tqdm
    for im in tqdm(np.arange(len(M_lithology)), desc='prior_discrete_data'):
        M_test = M_lithology[im]
        for i in range(len(depth_top)):
            z_top = depth_top[i]
            z_bottom = depth_bottom[i]
            id_top = np.argmin(np.abs(z - z_top))
            id_bottom = np.argmin(np.abs(z - z_bottom))
            #print("Layer %d: depth %.2f-%.2f, ids %d-%d, lithology obs %d" % (i+1, z_top, z_bottom, id_top, id_bottom, lithology_obs[i]))
            if id_top==id_bottom:
                lithology_layer = M_test[id_top]
                lithology_mode_layer = lithology_layer
            else:
                lithology_layer = M_test[id_top:id_bottom]
                # Find the most frequent lithology in this layer
                values, counts = np.unique(lithology_layer, return_counts=True)
                lithology_mode_layer = values[np.argmax(counts)]
            #print("  Most frequent lithology in layer: %d" % lithology_mode)
            lithology_mode[im, i] = lithology_mode_layer

    # Convert observed lithologies to d_obs probabilities 
    n_obs = len(lithology_obs)
    P_obs = np.zeros((nclass, n_obs))*np.nan
    for i in range(len(depth_top)):
        for j in range(nclass):
            if class_id[j] == lithology_obs[i]:
                P_obs[j, i] = P_single
            else:
                P_obs[j, i] = (1-P_single)/(nclass-1)


    # apply P_obs to the whole data grid with distance based weighting
    d_obs, i_use = Pobs_to_datagrid(P_obs, X_well, Y_well, f_data_h5, r_data=10, r_dis=100, doPlot=False)

    plt.figure()
    plt.imshow(P_obs)
    plt.xlabel('Number for "observed" lithology')
    plt.ylabel('Class ID')
    plt.title('P_obs for lithology observations')

    # Now we are ready to save and litholigy_mode to prior file P_obs to a data file, 
    # forst save the lithology_mode to a prior file as prior data, and get back the prior id of the prior data
    id_use = ig.save_prior_data(f_prior_data_h5, lithology_mode)

    # Next save the P_obs to a data file, with the correct prior id
    # Save P_obs to data file
    id_out, f_out = ig.write_data_multinomial(
        D_obs=d_obs,           # Shape: (nd, nclass, nm)
        i_use=i_use,           # Shape: (nd, 1) - binary mask
        id_use=id_use,              # Which PRIOR data to use (D2 from prior file)
        f_data_h5=f_data_h5,   # Output file
        showInfo=1             # Verbosity
    )
    print("Wrote data id %d to file: %s" % (id_out, f_out))

# %% Rejection inversion
# Get filename without extension
nr=1000
id_use = [1] # tTEM 
#id_use = [2] # Well1 - independent
#id_use = [3] # Well1 - compressed
#id_use = [4] # Well2 - independent
#id_use = [5] # Well2 - compressed
#id_use = [6] # Well1 - simple test -> cat 5
#id_use = [7] # Well2 - simple test-> cat 5
#id_use = [8] # well 1, dependent
#id_use = [9] # well 1, dependent compressed
#id_use = [10] # well 2, dependent
#id_use = [11] # well 2, dependent compressed

#id_use = [1] # tTEM 

id_use = [2,4] # Both wells independent
#id_use = [3,5] # Both wells compressed
id_use = [8,10] # Both wells, dependent
#id_use = [9,11] # Both wells, dependent compressed

#id_use = [1,2,4] # TEM + both wells independent
#id_use = [1,3,5] # TEM + both wells compressed
#id_use = [1,8,10] # TEM + both wells, dependent

# get string from id_use
fileparts = os.path.splitext(f_prior_data_h5)
id_use_str = '_'.join(map(str, id_use))
f_post_h5 = 'post_%s_NoiseGain%d_id%s_Ps%d.h5' % (fileparts[0], inflateTEMNoise, id_use_str,100*P_single)
f_post_h5 = ig.integrate_rejection(f_prior_data_h5, 
                                f_data_h5, 
                                f_post_h5, 
                                showInfo=0, 
                                id_use = id_use,
                                ip_range = id_line,
                                nr=nr,
                                parallel=False, 
                                autoT=False,
                                T_base=1,
                                updatePostStat=True)

# Plot profile
im_plot = 1 # Resistivity
im_plot = 2 # Lithology
#im_plot = 3 # Hypothesis
ig.plot_profile(f_post_h5, im=im_plot, ii=id_line, gap_threshold=50, xaxis='x', hardcopy=hardcopy)

#%
#ig.plot_profile(f_post_h5, im=1, ii=id_line, gap_threshold=50, xaxis='x', hardcopy=hardcopy)
