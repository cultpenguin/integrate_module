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
import copy
hardcopy=True

# remove all files with name 'da*IDEN*h5'
for file in os.listdir('.'):
    if file.startswith('daugaard') and 'IDEN' in file and file.endswith('.h5'):
        print("Removing existing file: %s" % file)
        os.remove(file)


# %%
N_use = 1000000
P_single=0.99
inflateTEMNoise = 4
# Extrapolation options for distance weighting
r_data=10 
r_dis=100


# Get Daugaard data files
case = 'DAUGAARD'
files = ig.get_case_data(case=case)
f_data_h5 = files[0]
f_data_h5 = 'DAUGAARD_AVG.h5'
file_gex= ig.get_gex_file_from_data(f_data_h5)

print("Using data file: %s" % f_data_h5)
print("Using GEX file: %s" % file_gex)
X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)
nd= len(X)


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

# Find indeces of data points along points (Xl,Yl), within buffer distance
Xl = np.array([X1, X2])
Yl = np.array([Y1, Y2])
buffer = 10.0
id_line, distances, segment_ids = ig.find_points_along_line_segments(
    X, Y, Xl, Yl, tolerance=buffer
)

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
        ig.copy_prior(f_prior_data_merged_full_h5, f_prior_data_h5, N_use=N_use, showInfo=1, loadtomem=True)
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
    ig.save_data_gaussian(D_obs, D_std=D_std, f_data_h5=f_data_h5, file_gex=file_gex)


#%% DESCRIBE THE WELL DATA
# Define wells information in different ways. 
# Frist deifne what direct information from well log
WELLS=[]
WELL_NAMES=[]


# WELL 1: DAU02
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
W['lithology_prob'] = [P_single, P_single, P_single, P_single, P_single, P_single, P_single, P_single, P_single, P_single]
W['X'] = 542983.01
W['Y'] = 6175822.76
W['name'] = 'DAU02 - Full'
WELLS.append(W)  
WELL_NAMES.append('%s Full' % W['name'])

W_compressed = copy.deepcopy(W)
W_compressed['depth_top'] =    [0   , 13.2, 16.6]
W_compressed['depth_bottom'] = [13.2, 16.6, 20]
W_compressed['lithology_obs'] = [2, 5, 3]
W_compressed['lithology_prob'] = [P_single, P_single, P_single]
W_compressed['name'] = 'DAU02 - Compressed'
WELLS.append(W_compressed)
WELL_NAMES.append('%s Compressed' % W['name'])

# SINGLE LAYER: lithoilogy 5 from 20-24 m
W = {}
W['depth_top'] =     [20]
W['depth_bottom'] = [24]
W['lithology_obs'] = [5] 
W['lithology_prob'] = [P_single]
W['X'] = 542983.01
W['Y'] = 6175822.76
W['name'] = 'DAU02'
WELLS.append(W.copy())
WELL_NAMES.append('%s Single Layer' % W['name'])


# WELL 2: 116.1602
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
W['lithology_prob'] = [P_single, P_single, P_single, P_single, P_single, P_single, P_single]
W['X'] = 543584.098
W['Y'] = 6175788.478
W['name'] = '116.1602 - Full'                      
WELLS.append(W)
WELL_NAMES.append('%s Full' % W['name'])

W_compressed = copy.deepcopy(W)
W_compressed['depth_top'] =     [0,  8, 15,   17, 23.5, 45]
W_compressed['depth_bottom'] =  [8, 15, 17, 23.5,   45, 46]
W_compressed['lithology_obs'] = [3,  5,  3,    5,    6, 7]
W_compressed['lithology_prob'] = [P_single, P_single, P_single, P_single, P_single, P_single]
W['name'] = '116.1602 - Compressed'                      
WELLS.append(W_compressed)  
WELL_NAMES.append('%s Compressed' % W['name'])


# SINGLE LAYER: lithoilogy 5 from 20-24 m
W = {}
W['depth_top'] =     [20]
W['depth_bottom'] = [24]
W['lithology_obs'] = [5] 
W['lithology_prob'] = [P_single]
W['X'] = 543584.098
W['Y'] = 6175788.478
W['name'] = '116.1602 - Single Layer'
WELLS.append(W.copy())
WELL_NAMES.append('%s Single Layer' % W['name'])

# %% Write different types of data
# load prior im = 2


#%%  First we add a data set that is simple the lithology at the well position
# Make prior data 'D2' that is simple and identity of the prior lithology type
f_prior_data_h5 = ig.prior_data_identity(f_prior_data_h5, im=2, id=2, doMakePriorCopy=True)

#%%

for iw in np.arange(len(WELLS)):
    print("considering well %d: %s" % (iw+1, WELLS[iw]['name']))

    W = WELLS[iw]
    #depth_top = W['depth_top']
    #depth_bottom = W['depth_bottom']
    #lithology_obs = W['lithology_obs']
    #lithology_prob = W['lithology_prob']
    #X_well = W['X']
    #Y_well = W['Y']
    #P_obs = ig.compute_P_obs_discrete( z=z, class_id=class_id, depth_top=W['depth_top'], depth_bottom=W['depth_bottom'], lithology_obs=W['lithology_obs'],lithology_prob=W['lithology_prob'], P_prior=None)
    P_obs = ig.compute_P_obs_discrete(z=z, class_id=class_id, W=W)
    
    plt.figure()
    plt.imshow(P_obs)

    # apply P_obs to the whole data grid with distance based weighting
    if (iw==0)|(iw==3):
        doPlot=True
    else:
        doPlot=False
    d_obs, i_use = ig.Pobs_to_datagrid(P_obs, W['X'], W['Y'], f_data_h5, r_data=r_data, r_dis=r_dis, doPlot=doPlot)

    #% Write to DATA file
    id_out, f_out = ig.save_data_multinomial(
        D_obs=d_obs,           # Shape: (nd, nclass, nm)
        i_use=i_use,           # Shape: (nd, 1) - binary mask
        id=2+iw,                  # Data ID (will create /D2/ in DATA.h5)
        id_prior=2,            # Which PRIOR data to use (D2 from prior file)
        f_data_h5=f_data_h5,   # Output file
        showInfo=1             # Verbosity
    )

    plt.show()

#%% Now use a sparse representation of the well log data. 
# Instead of reprsentiug all nz model paraeters, we cnvert the log obserbavtion to 
# represent the actual number of layer obverservation.
# This we need to implement a function that goes through a realizations
# of the prior and returns the lithology at each depth intervals, and we need to decide how we do that.
#

# load the prior lithology data
M, idx = ig.load_prior_model(f_prior_data_h5)
M_lithology = M[1]
nm = M_lithology.shape[1]
nreal = M_lithology.shape[0]

for iw in [0,1,3,4]:
    #iw = 0
    W = WELLS[iw]
    P_obs, lithology_mode = ig.compute_P_obs_sparse(M_lithology, z=z, class_id=class_id, W=W)
 
    # apply P_obs to the whole data grid with distance based weighting
    d_obs, i_use = ig.Pobs_to_datagrid(P_obs, W['X'], W['Y'], f_data_h5, r_data=r_data, r_dis=r_dis, doPlot=False)

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
    id_out, f_out = ig.save_data_multinomial(
        D_obs=d_obs,           # Shape: (nd, nclass, nm)
        i_use=i_use,           # Shape: (nd, 1) - binary mask
        id_prior=id_use,       # Which PRIOR data to use (D2 from prior file)
        f_data_h5=f_data_h5,   # Output file
        showInfo=1             # Verbosity
    )
    print("Wrote data id %d to file: %s" % (id_out, f_out))

# %% Rejection inversion
# Get filename without extension
nr=1000
id_use = [1] # tTEM 
#id_use = [2] # Well1 - independent
#id_use = [3] # Well1 - ind compressed
#id_use = [4] # Well1 - simple test -> cat 5
#id_use = [5] # Well2 - independent
#id_use = [6] # Well2 - ind compressed
#id_use = [7] # Well1 - simple test -> cat 5
#id_use = [8] # well 1, dependent
#id_use = [9] # well 1, dependent compressed
#id_use = [10] # well 2, dependent
#id_use = [11] # well 2, dependent compressed

#id_use = [1] # tTEM 

#id_use = [2,5] # Both wells independent
#id_use = [3,6] # Both wells compressed
#id_use = [8,10] # Both wells, dependent
#id_use = [9,11] # Both wells, dependent compressed

id_use = [9,11] # both wells, dependent # One should alsways chweck the wells alone, to check for concistency with the prior.
#id_use = [1] # TEM + both wells, dependent
id_use = [1, 9,11] # Both wells independent + tTEM

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
                                parallel=True, 
                                autoT=True,
                                T_base=1,
                                updatePostStat=True)
#ig.plot_profile(f_post_h5, im=1, ii=id_line, gap_threshold=50, xaxis='x', hardcopy=hardcopy, alpha = 1,std_min = 0.3, std_max = 0.6,)
ig.plot_profile(f_post_h5, im=2, ii=id_line, gap_threshold=50, xaxis='x', hardcopy=hardcopy, alpha=1, entropy_min =0.4, entropy_max=1.0)


# %% Plot profile
#ig.plot_profile(f_post_h5, im=1, ii=id_line)

# %%
