#!/usr/bin/env python
# %% [markdown]
# # Getting started with INTEGRATE
#
# This notebook contains a simple example of geeting started with INTEGRATE

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
    # # # # #%load_ext autoreload
    # # # # #%autoreload 2
    pass
# %%
import integrate as ig
import numpy as np
import matplotlib.pyplot as plt
# check if parallel computations can be performed
parallel = ig.use_parallel(showInfo=1)
hardcopy = True
# %% Get tTEM data from SGU
case = 'FRAASTAD'

f_data_h5='Fra20200930_202001001_1_AVG_export.h5'
file_gex='fraastad_ttem.gex'
f_prior_h5 = 'prior_gotaelv_N1000000_dmax90.h5'

print("Using data file: %s" % f_data_h5)
print("Using GEX file: %s" % file_gex)

ig.check_data(f_data_h5)

# %% [markdown]
# ## 1. Setup the prior model ($\rho(\mathbf{m},\mathbf{d})$
#
# In this example a simple layered prior model will be considered

# %% [markdown]
# ### 1a. first, a sample of the prior model parameters, $\rho(\mathbf{m})$, will be generated

# %% A. CONSTRUCT PRIOR MODEL OR USE EXISTING
N=1000000
doLoadSimplePrior = 0
if doLoadSimplePrior:
    # Layered model
    f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='chi2', NLAY_deg=3, RHO_min=1, RHO_max=3000)
else:   
    ig.integrate_update_prior_attributes(f_prior_h5)

# Plot some summary statistics of the prior model
ig.plot_prior_stats(f_prior_h5)

# %% [markdown]
# ### 1b. Then, a corresponding sample of $\rho(\mathbf{d})$, will be generated

# %% Compute prior DATA
f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, parallel=parallel, showInfo=0, Ncpu=64, N=N)

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


# %% 
X, Y, LINE, ELEVATION = ig.get_geometry(f_post_h5)
#% get index of position where  LINE >2000
ii = np.where((LINE > 2100) & (LINE < 2150))[0]
i1=ii[0]
i2=ii[-1]

# %% [markdown]
# ### Plot some statistic from $\sigma(\mathbf{m})$
ig.plot_geometry(f_data_h5, hardcopy=hardcopy)

#plt.show()
ig.plot_geometry(f_data_h5, ii=ii, pl='LINE', hardcopy=hardcopy)

# %% Plot prior, posterior, and observed  data
ig.plot_data_prior_post(f_post_h5, i_plot=i1, hardcopy=hardcopy)
ig.plot_data_prior_post(f_post_h5, i_plot=i2, hardcopy=hardcopy)

# %% Posterior analysis
# Plot the Temperature used for inversion
ig.plot_T_EV(f_post_h5, pl='T', hardcopy=hardcopy)
# Plot the evidence (prior likelihood) estimated as part of inversion
ig.plot_T_EV(f_post_h5, pl='EV', hardcopy=hardcopy)


# %% Plot Profiles
ig.plot_profile(f_post_h5, i1=i1, i2=i2, im=1, hardcopy=hardcopy)
ig.plot_profile(f_post_h5, i1=i1, i2=i2, im=2, hardcopy=hardcopy)
# %%

# Plot a 2D feature: Resistivity in layer 10
ig.plot_feature_2d(f_post_h5,im=1,iz=12, key='Median', uselog=1, cmap='jet', s=10)
#ig.plot_feature_2d(f_post_h5,im=1,iz=80,key='Median')
ig.plot_feature_2d(f_post_h5,im=2,iz=12, key='Mode', uselog=0, cmap='jet', s=10)

try:
    # Plot a 2D feature: The number of layers
    ig.plot_feature_2d(f_post_h5,im=2,iz=0,key='Median', title_text = 'Number of layers', uselog=0, clim=[1,6], cmap='jet', s=12)
except:
    pass


# %% Get posterior statistics
import h5py
with h5py.File(f_post_h5,'r') as f_post:
    print(f_post['/'].keys())
    i_use = f_post['/i_use'][:]
with h5py.File(f_prior_data_h5,'r') as f_prior:
    class_id = f_prior['/M2'].attrs['class_id']
    class_name = f_prior['/M2'].attrs['class_name']


M = ig.load_prior_model(f_prior_data_h5)
CLASS=M[0][1]    
X, Y, LINE, ELEVATION = ig.get_geometry(f_post_h5)

nd=i_use.shape[0]
nr=i_use.shape[1]


for class_use in [1,2,3,4,5]:
    thick_class_mean = np.zeros(nd)
    thick_class_prob = np.zeros(nd)
    thick_min = 10
    for id in np.arange(nd):
        i_use[id]
        post_reals = CLASS[i_use[id],:]
        # find the nummber of layers that have post_reals==class_id
        thick_class = np.sum(post_reals==class_use,axis=1)
        thick_class_mean[id] = np.mean(thick_class)
        thick_class_prob[id] = np.sum(thick_class>thick_min)/nr


    name = class_name[np.where(class_id==class_use)[0][0]]

    plt.figure()
    plt.scatter(X, Y, c=thick_class_mean, cmap='jet', s=1)
    plt.axis('equal')
    plt.grid()
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Mean thickness of %s [%d]' % (name,class_use)  )
    fout = 'thick_mean_%s_%d_N%d.png' % (name,class_use,N)
    fout = fout.replace(' ','_')
    fout = fout.replace(',','_')
    plt.savefig(fout, dpi=300)
        
    plt.figure()
    plt.scatter(X, Y, c=thick_class_prob, cmap='jet', s=1, vmin=0, vmax=1)
    plt.axis('equal')
    plt.grid()
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Probability of having thickness > %3.1fm for class %s [%d]' % (thick_min, name,class_use))
    fout = 'thick_prob_%s_%d_N%d.png' % (name,class_use,N)
    fout = fout.replace(' ','_')
    fout = fout.replace(',','_')
    plt.savefig(fout, dpi=300)  
    
    
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
'''
pv.set_plot_theme("document")
p = pv.Plotter(notebook=True)
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
'''

# %%
