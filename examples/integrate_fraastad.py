#!/usr/bin/env python
# %% [markdown]
# # INTEGRATE Fraastad example

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
    #%load_ext autoreload
    #%autoreload 2
    pass

import integrate as ig
import numpy as np
import matplotlib.pyplot as plt

# %% Choose the GEX file used for forward modeling. THis should be stored in the data file.
#file_gex= ig.get_gex_file_from_data(f_data_h5, id=id)
f_data_h5 = 'Fra20200930_202001001_1_AVG_export.h5'
file_gex ='fraastad_ttem.gex'
print("Using GEX file: %s" % file_gex)


# %% [markdown]
# ## 1. Setup the prior model, $\rho(\mathbf{m},\mathbf{d})$.

# A1. CONSTRUCT PRIOR MODEL OR USE EXISTING
N=25000
RHO_min = 1
RHO_max = 1500
z_max = 50 

useP=3
if useP==1:
    ## Layered model
    #f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='chi2', NLAY_deg=5, z_max = z_max, rho_dist='log-uniform', RHO_min=RHO_min, RHO_max=RHO_max)
    #f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='uniform', z_max = z_max, NLAY_min=1, NLAY_max=3, rho_dist='log-uniform', RHO_min=RHO_min, RHO_max=RHO_max)
    f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='uniform', z_max = z_max, NLAY_min=1, NLAY_max=8, rho_dist='log-uniform', RHO_min=RHO_min, RHO_max=RHO_max)
elif useP==2:
    ## N layer model with increasing thickness
    #f_prior_h5 = ig.prior_model_workbench(N=N, z2 = 30, nlayers=20, rho_min = RHO_min, rho_max = RHO_max)
    #f_prior_h5 = ig.prior_model_workbench(N=N, z2 = 30, nlayers=5, rho_dist='log-uniform', rho_min = RHO_min, rho_max = RHO_max)
    f_prior_h5 = ig.prior_model_workbench(N=N, rho_mean=45, rho_std=55, rho_dist='log-normal', z2 = 30, nlayers=12, rho_min = RHO_min, rho_max = RHO_max)
else:
    f_prior_h5 = 'gotaelv_Daugaard_N1000000.h5'
    f_prior_h5 = 'gotaelv2_N50000.h5'
    f_prior_h5 = 'gotaelv2_N1000000.h5'


ig.plot_prior_stats(f_prior_h5)

# %% [markdown]
# ## 2. Compute prior data, $\rho(\mathbf{d})$.

# %% A2. Compute prior DATA
f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, Nproc=0, N=N)
#f_prior_data_h5 = 'gotaelv_Daugaard_N1000000_fraastad_ttem_Nh280_Nf12.h5'

# %% [markdown]
# ## Sample the posterior $\sigma(\mathbf{m})$
#
# The posterior distribution is sampling using the extended rejection sampler.

# %% READY FOR INVERSION
N_use = 100000
#f_prior_data_h5 = 'gotaelv2_N1000000_fraastad_ttem_Nh280_Nf12.h5'
updatePostStat =True
f_post_h5 = ig.integrate_rejection(f_prior_data_h5, f_data_h5, N_use = N_use, parallel=1, updatePostStat=updatePostStat, showInfo=1)

# % Compute some generic statistic of the posterior distribution (Mean, Median, Std)
#if not updatePostStat:
#    ig.integrate_posterior_stats(f_post_h5)



# %% [markdown]
# ### Plot some statistics from $\sigma(\mathbf{m})$

# %% Posterior analysis
# Plot the Temperature used for inversion
ig.plot_T_EV(f_post_h5, pl='T')
ig.plot_T_EV(f_post_h5, pl='EV')
ig.plot_T_EV(f_post_h5, pl='ND')
#

#%%
ig.plot_data_prior_post(f_post_h5, i_plot = 0)
ig.plot_data_prior_post(f_post_h5, i_plot = 1199)

# %% Plot Profiles
ig.plot_profile(f_post_h5, i1=7000, i2=7300, im=1)
try:
    ig.plot_profile(f_post_h5, i1=7000, i2=7300, im=2)
except:
    pass
 # %%

## Plot a 2D feature: Resistivity in layer 10
#ig.plot_feature_2d(f_post_h5,im=1,iz=12, key='Median', uselog=1, cmap='jet', s=10, clim=np.log10([RHO_min,RHO_max]))
##ig.plot_feature_2d(f_post_h5,im=1,iz=80,key='Median')

#%% 
#for iz in range(40):
#    ig.plot_feature_2d(f_post_h5,im=1,iz=iz, key='Median', uselog=1, cmap='jet', s=10, clim=np.log10([RHO_min,RHO_max]))

#%%

try:
    # Plot a 2D feature: The number of layers
    #ig.plot_feature_2d(f_post_h5,im=2,iz=0,key='Median', title_text = 'Number of layers', cmap='jet', s=12)
    ig.plot_feature_2d(f_post_h5,im=2,iz=22,key='Mode', title_text = 'Lithology Mode', cmap='jet', s=12)
except:
    pass


# %% Compute cumulative category

im = 2 
icat=np.array([2])
thick_mean, thick_median, thick_std, class_names, X, Y = ig.posterior_cumulative_thickness(f_post_h5,im=2, icat=icat)
plt.scatter(X,Y, c=thick_median, cmap='jet', s=10)
plt.colorbar()
plt.title('Cumulative Thickness - median - %s ' % class_names)
plt.show()

#%% 
im = 2 
icat=np.array([4])
thick_mean, thick_median, thick_std, class_names, X, Y = ig.posterior_cumulative_thickness(f_post_h5,im=2, icat=icat)
plt.scatter(X,Y, c=thick_std, cmap='jet', s=10)
plt.colorbar()
plt.title('Cumulative Thickness - std - %s ' % class_names)
plt.show()


#%% PRIOIOR POST 
im = 2 
icat=np.array([2])
clim=[0,10]
thick_mean, thick_median, thick_std, class_names, X, Y = ig.posterior_cumulative_thickness(f_post_h5,im=2, icat=icat, usePrior=False)
thick_mean_prior, thick_median_prior, thick_std_prior, class_names, X, Y = ig.posterior_cumulative_thickness(f_post_h5,im=2, icat=icat, usePrior=True)
plt.subplot(1,2,1)
plt.scatter(X,Y, c=thick_median, cmap='jet', s=1)
plt.colorbar()
plt.clim(clim)
plt.title('POST')
plt.axis('equal')

plt.subplot(1,2,2)
plt.scatter(X,Y, c=thick_median_prior, cmap='jet', s=1)
plt.colorbar()
plt.clim(clim)
plt.axis('equal')
plt.title('PRIOR')

plt.suptitle('Cumulative Thickness - median - %s ' % class_names)
plt.show()

plt.savefig('POST_CUMULATIVE_THICKNESS_%d_COMPARE.png' % ic)


#%%
for ic in [0,1,2,3,4]:
    icat=np.array([ic])
    
    thick_mean, thick_median, thick_std, class_names, X, Y = ig.posterior_cumulative_thickness(f_post_h5,im=2, icat=icat)
    #fig, ax = plt.subplots(1,2, figsize=(15,5))
    #%ax[0].scatter(X,Y, c=thick_median, cmap='jet', s=10)
    #%plt.colorbar()
    #%plt.title('Cumulative Thickness - std - %s ' % class_names)
    #%plt.show()

    fig, ax = plt.subplots(1,2, figsize=(15,5))
    im0 = ax[0].scatter(X,Y, c=thick_median, cmap='jet', s=10)
    ax[0].set_title('Cumulative Thickness - Median - %s ' % class_names)
    ax[0].axis('equal')
    
    im1 = ax[1].scatter(X,Y, c=thick_std, cmap='jet', s=10)
    ax[1].set_title('Cumulative Thickness - std - %s ' % class_names)
    ax[1].axis('equal')
    
    cbar0 = fig.colorbar(im0, ax=ax[0])
    cbar1 = fig.colorbar(im1, ax=ax[1])

    plt.savefig('POST_CUMULATIVE_THICKNESS_%d.png' % ic)

    plt.show()







# %%
import h5py
import integrate as ig

with h5py.File(f_post_h5,'r') as f_post:
    f_prior_h5 = f_post['/'].attrs['f5_prior']
    f_data_h5 = f_post['/'].attrs['f5_data']

X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)

Mstr = '/M%d' % im
with h5py.File(f_prior_h5,'r') as f_prior:
    if not Mstr in f_prior.keys():
        print('No %s found in %s' % (Mstr, f_prior_h5))
        #return 1
    if not f_prior[Mstr].attrs['is_discrete']:
        print('M%d is not discrete' % im)
        #return 1

#%%

with h5py.File(f_prior_h5,'r') as f_prior:
    try:
        z = f_prior[Mstr].attrs['z'][:].flatten()
    except:
        z = f_prior[Mstr].attrs['x'][:].flatten()
    is_discrete = f_prior[Mstr].attrs['is_discrete']
    if 'clim' in f_prior[Mstr].attrs.keys():
        clim = f_prior[Mstr].attrs['clim'][:].flatten()
    else:
        # if clim set in kwargs, use it, otherwise use default
        if 'clim' in kwargs:
            clim = kwargs['clim']
        else:
            clim = [.1, 2600]
            clim = [10, 500]
    if 'class_id' in f_prior[Mstr].attrs.keys():
        class_id = f_prior[Mstr].attrs['class_id'][:].flatten()
    else:   
        print('No class_id found')
    if 'class_name' in f_prior[Mstr].attrs.keys():
        class_name = f_prior[Mstr].attrs['class_name'][:].flatten()
    else:
        class_name = []
    n_class = len(class_name)
    if 'cmap' in f_prior[Mstr].attrs.keys():
        cmap = f_prior[Mstr].attrs['cmap'][:]
    else:
        cmap = plt.cm.hot(np.linspace(0, 1, n_class)).T
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(cmap.T)            
    #print(cmap)
    #print(cmap.shape)
    #print('class_name = %s' % class_name)
    #print('clim %f-%f' % (clim[0],clim[1]))

#%%

with h5py.File(f_post_h5,'r') as f_post:
    P=f_post[Mstr+'/P'][:]
    i_use = f_post['/i_use'][:]

ns,nr=i_use.shape

f_prior = h5py.File(f_prior_h5,'r')
M_prior = f_prior[Mstr][:]
f_prior.close()
nz = M_prior.shape[1]

thick_mean = np.zeros((ns))
thick_median = np.zeros((ns))
thick_std = np.zeros((ns))
#%%

thick = np.diff(z)

for i in range(ns):
    jj = i_use[i,:].astype(int)
    m_sample = M_prior[jj,:]
        
    cum_thick = np.zeros((nr))
    for ic in range(len(icat)):
    
        # the number of values of i_cat in the sample

        i_match = (m_sample == class_id[icat[ic]]).astype(int)
        i_match = i_match[:,0:nz-1]
        
        n_cat = np.sum(m_sample==icat[ic], axis=0)
    
        cum_thick = cum_thick + np.sum(i_match*thick, axis=1)

    thick_mean[i] = np.mean(cum_thick)
    thick_median[i] = np.median(cum_thick)
    thick_std[i] = np.std(cum_thick)

class_out = class_name[icat]

