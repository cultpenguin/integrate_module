#!/usr/bin/env python
# %% [markdown]
# # Ex for Daniel



# %%
try:
    # Check if the code is running in an IPython kernel (which includes Jupyter notebooks)
    get_ipython()
    # If the above line doesn't raise an error, it means we
    #  are in a Jupyter environment
    # Execute the magic commands using IPython's run_line_magic function
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except:
    # If get_ipython() raises an error, we are not in a Jupyter environment
    # # # # # # # #%load_ext autoreload
    # # # # # # # #%autoreload 2
    pass

# %%
import integrate as ig
from integrate.integrate_io import get_geometry
import numpy as np
# Check if parallel computations can be performed
parallel = ig.use_parallel(showInfo=1)
hardcopy = True 
import matplotlib.pyplot as plt

# %% [markdown]
# ## 0. Get TTEM data
# erda/DGMAP/DATA/Soeballe_NN/

# D_OBS
f_data_h5 = 'data_diamond_soeballe_250724.h5'

X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)

# iline should be the index of all points along a specific line. If nothing is specified, just use all data
useLINE = 100801
if useLINE>0:
    idx_line = LINE == useLINE
    iline = np.where(idx_line)[0]
else:
    iline = np.arange(X.shape[0])


# PRIOR MODEL AND DATA
f_prior_h5 = 'prior_general.h5'
f_prior_h5 = 'prior_soeballe.h5'
f_prior_h5 = 'prior_chi2_dmax120_v1.h5'

#f_prior_h5 = 'prior_soeballe.h5'
f_prior_data_h5 = f_prior_h5

# Make Tx Rx height prior data!!
f_prior_data_h5 = ig.prior_data_identity(f_prior_data_h5, im=2, id=2, doMakePriorCopy=True)
f_prior_data_h5 = ig.prior_data_identity(f_prior_data_h5, im=3, id=3, doMakePriorCopy=False)
#f_prior_data_h5 = ig.prior_data_identity(f_prior_data_h5, im=2, doMakePriorCopy=True)  # Tx
#f_prior_data_h5 = ig.prior_data_identity(f_prior_data_h5, im=3, doMakePriorCopy=False)  # Rx


# %%
doLoadData = True
if doLoadData:
    # load prior 
    M, idx = ig.load_prior_model(f_prior_h5)
    D, idx = ig.load_prior_data(f_prior_h5)
    #print("Loaded prior model from %s with %d realizations" % (f_prior_h5, M.shape[0]))
    #print("Loaded prior data from %s with %d realizations" % (f_prior_h5, D.shape[0]))

    # load observd data
    D_obs = ig.load_data(f_data_h5)
    d_obs = D_obs['d_obs']
    d_std = D_obs['d_std']

# %% 
doFixData = True
if doFixData:
    f_data_h5='d_test.h5'
    ig.save_data_gaussian(d_obs[0][:,1:], D_std = 3*d_obs[0][:,1:]*d_std[0][:,1:], f_data_h5=f_data_h5, 
                            id=1,
                            showInfo=0, 
                            UTMX=X, 
                            UTMY=Y,
                            ELEVATION=ELEVATION,
                            LINE=LINE,
                            name='Diamond Data',
                            id_prior = 1,                        
                            delete_if_exist=True,                            
            )
    ig.save_data_gaussian(d_obs[1], D_std = d_std[1], f_data_h5=f_data_h5, 
                            id=2,
                            showInfo=0, 
                            name='Rx',
                            id_prior = 2,
                            delete_if_exist=False,                        
            )
    ig.save_data_gaussian(d_obs[2], D_std = d_std[2], f_data_h5=f_data_h5, 
                            id=3,
                            showInfo=0, 
                            name='Rx',
                            id_prior = 3,
                            delete_if_exist=False,                        
            )
    
    
    
    
    #D_obs = ig.load_data(f_data_h5)
    #d_obs = D_obs['d_obs']
    #d_std = D_obs['d_std']



#%% 
ig.plot_prior_stats(f_prior_h5, hardcopy=hardcopy)
# %% 
ig.plot_data_prior(f_prior_h5,f_data_h5,nr=1000,hardcopy=hardcopy)

# %%
ig.plot_geometry(f_data_h5, pl='ELEVATION')
# %%
# The electromagnetic data (d_obs and d_std) can be plotted using ig.plot_data:
ig.plot_data(f_data_h5, hardcopy=hardcopy)
# Plot data channel 15 in an XY grid
ig.plot_data_xy(f_data_h5, data_channel=15, cmap='jet');

# %% 
try:
    plt.figure(figsize=(10,8))
    plt.subplot(2,2,1)
    plt.scatter(X,Y,c=d_obs[1], s=1);
    plt.colorbar(label='d_obs[1]');
    plt.subplot(2,2,2)
    plt.scatter(X,Y,c=d_obs[2], s=1);
    plt.colorbar(label='d_obs[2]');
    plt.subplot(2,2,3)
    plt.scatter(X,Y,c=d_obs[1]- d_obs[2], s=1);
    plt.colorbar(label='d_obs[1]- d_obs[2]');
    plt.subplot(2,2,4)
    plt.plot(d_obs[1]- d_obs[2],'.');
    plt.xlabel('#');
    plt.ylabel('d_obs[1]- d_obs[2]'); 

except:
    pass



# %%

# %%
# Rejection sampling of the posterior can be done with default settings using:
#f_post_h5 = ig.integrate_rejection(f_prior_h5, f_data_h5)

N=1000000
# However, you can control several important options.
# You can choose to use only a subset of the prior data. Decreasing the sample 
# size makes the inversion faster but increasingly approximate.
N_use = 1000000   # Number of prior samples to use (use all available)
T_base = 1  # Base annealing temperature for rejection sampling
autoT = 1   # Automatically estimate optimal annealing temperature
f_post_h5 = ig.integrate_rejection(f_prior_data_h5, 
                                   f_data_h5, 
                                   ip_range=iline,
                                   #id_use = [1], # ONLY dB/dT
                                   #id_use = [2], # ONLY Tx
                                   #id_use = [1,2,3] # ALL
                                   #f_post_h5 = 'POST.h5', 
                                   nr=1000,
                                   N_use = N_use, 
                                   autoT = autoT,
                                   T_base = T_base,                            
                                   showInfo=1, 
                                   parallel=parallel)

# %%
ig.plot_profile(f_post_h5, ii=iline, hardcopy=hardcopy, xaxis='y', im=1, alpha=1, std_min = 0.4, std_max=.7)


# %% [markdown]
# ## 3. Plot statistics from the posterior $\sigma(\mathbf{m})$

# %%
ig.plot_data_prior_post(f_post_h5, i_plot=1619,hardcopy=hardcopy)
ig.plot_data_prior_post(f_post_h5, i_plot=1830,hardcopy=hardcopy)


# %% [markdown]
# ### Resistivity profiles
#
# Plot a profile showing posterior statistics of model parameter M1 (resistivity)
# along a section of the survey line.

# %%
ig.plot_profile(f_post_h5, ii=iline, hardcopy=hardcopy, xaxis='y')
#ig.plot_profile(f_post_h5, ii=iline, im=1, hardcopy=hardcopy)
#ig.plot_profile(f_post_h5, ii=iline, im=2, hardcopy=hardcopy)
#ig.plot_profile(f_post_h5, ii=iline, im=3, hardcopy=hardcopy)
#ig.plot_profile(f_post_h5, i1=1, i2=2800, im=1, hardcopy=hardcopy)
#ig.plot_profile(f_post_h5, i1=1, i2=2800, im=2, hardcopy=hardcopy)
#ig.plot_profile(f_post_h5, i1=1, i2=2800, im=3, hardcopy=hardcopy)


# %% [markdown]
# ### Evidence and annealing temperature
# The evidence quantifies how well the data fits the model,
# while temperature controls the acceptance rate in rejection sampling.

# %%
# Plot the annealing temperature used for inversion
ig.plot_T_EV(f_post_h5, pl='T',hardcopy=hardcopy)
# Plot the evidence (log-likelihood) estimated during inversion
#ig.plot_T_EV(f_post_h5, pl='EV',hardcopy=hardcopy)
# Plot the normalized mean-loglikelihood
# Values less than one suggest overfitting
# Values above one suggest underfitting
#ig.plot_T_EV(f_post_h5, pl='LOGL_mean',hardcopy=hardcopy)



# %% [markdown]
# ### Plot 2D spatial features
#
# Plot the median resistivity at specific depths (layers 5, 30, and 50)
# to show lateral variations in subsurface structure.

# %%

# Plot 2D features: Resistivity at different depths
# try:
#     ig.plot_feature_2d(f_post_h5,im=1,iz=5, key='Median', uselog=1, cmap='jet', s=10,hardcopy=hardcopy)
#     plt.show()
# except:
#     pass

# try:
#     ig.plot_feature_2d(f_post_h5,im=1,iz=30, key='Median', uselog=1, cmap='jet', s=10,hardcopy=hardcopy)
#     plt.show()
# except:
#     pass

# try:
#     ig.plot_feature_2d(f_post_h5,im=1,iz=50, key='Median', uselog=1, cmap='jet', s=10,hardcopy=hardcopy)
#     plt.show()
# except:
#     pass

# %%
# try:
#     # Plot a 2D feature: The estimated number of layers
#     ig.plot_feature_2d(f_post_h5,im=3,iz=0,key='Median', uselog=0, clim=[1,6], cmap='jet', s=12,hardcopy=hardcopy)
#     plt.show()
# except:
#     pass

# %% [markdown]
# ## Export results to CSV format
# Export the posterior results to CSV files for use in GIS software or further analysis.

# %%
doExport = False
if doExport:
    f_csv, f_point_csv = ig.post_to_csv(f_post_h5)

    # Read the exported CSV file for inspection
    # Example filename (actual filename will be generated automatically):
    #f_point_csv = 'POST_DAUGAARD_AVG_PRIOR_CHI2_NF_3_log-uniform_N100000_TX07_20231016_2x4_RC20-33_Nh280_Nf12_Nu100000_aT1_M1_point.csv'
    import pandas as pd
    df = pd.read_csv(f_point_csv)
    df.head()

    # Optional: Use PyVista for 3D visualization of X,Y,Z coordinates with median resistivity
    plPyVista = False
    if plPyVista:
        import pyvista as pv
        import numpy as np
        from pyvista import examples
        #pv.set_jupyter_backend('client')
        pv.set_plot_theme("document")
        p = pv.Plotter(notebook=True)
        p = pv.Plotter()
        filtered_df = df[(df['Median'] < 50) | (df['Median'] > 200)]
        #filtered_df = df[(df['LINE'] > 1000) & (df['LINE'] < 1400) ]
        points = filtered_df[['X', 'Y', 'Z']].values[:]
        median = np.log10(filtered_df['Mean'].values[:])
        opacity = np.where(filtered_df['Median'].values[:] < 100, 0.5, 1.0)
        #p.add_points(points, render_points_as_spheres=True, point_size=3, scalars=median, cmap='jet', opacity=opacity)
        p.add_points(points, render_points_as_spheres=True, point_size=6, scalars=median, cmap='hot')
        p.show_grid()
        p.show()

