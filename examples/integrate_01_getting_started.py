#!/usr/bin/env python
# %% [markdown]
# # Getting started with INTEGRATE
#
# This notebook contains a simple example of geeting started with INTEGRATE

#%% 
try:
    # Check if the code is running in an IPython kernel (which includes Jupyter notebooks)
    get_ipython()
    # If the above line doesn't raise an error, it means we are in a Jupyter environment
    # Execute the magic commands using IPython's run_line_magic function
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except:
    # If get_ipython() raises an error, we are not in a Jupyter environment
    %load_ext autoreload
    %autoreload 2

# %%
import integrate as ig



# %% Choose the GEX file used for forward modeling. THis should be stored in the data file.
#f_data_h5 = 'tTEM_20230727_20230814_RAW_export.h5'
f_data_h5 = 'tTEM_20230727_20230814_AVG_export_J1000.h5'
#f_data_h5 = 'tTEM_20230727_20230814_AVG_export_J200.h5'
#id = 1  
#file_gex= ig.get_gex_file_from_data(f_data_h5, id=id)


file_gex ='ttem_example.gex'
print("Using GEX file: %s" % file_gex)


# %% [markdown]
# ## 1. Setup the prior model ($\rho(\mathbf{m},\mathbf{d})$
#
# In this example a simple layered prior model will be considered

# %% [markdown]
# ### 1a. first, a sample of the prior model parameters, $\rho(\mathbf{m})$, will be generated

# %% A. CONSTRUCT PRIOR MODEL OR USE EXISTING
#N=5000000
#f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='chi2', NLAY_deg=5)
N=100000
f_prior_h5 = 'PRIOR_Daugaard_N100000.h5'

# %% [markdown]
# ### 1b. Then, a corresponding sample of $\rho(\mathbf{d})$, will be generated

# %% Compute prior DATA
f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex)

# %% [markdown]
# ## Sample the posteriorm $\sigma(\mathbf{m})$
#
# The posterior distribution is sampling using the extended rejection sampler.

# %% READY FOR INVERSION [markdown]
f_post_h5 = ig.integrate_rejection(f_prior_data_h5, f_data_h5, N_use = 5000000, parallel=1, updatePostStat=False, showInfo=1)

# %% Compute some generic statistic of the posterior distribtiuon (Mean, Median, Std)
ig.integrate_posterior_stats(f_post_h5)

#%% 
import h5py
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

f_prior = h5py.File(f_prior_data_h5, 'r')
f_post = h5py.File(f_post_h5, 'a')

name = '/M2'
dataset = f_prior[name]
i_use = f_post['i_use'][:]
nsounding, nr = i_use.shape
nm = dataset.shape[1]
# Get number of classes for name    


class_id = f_prior[name].attrs['class_id']
n_classes = len(class_id)


#m_post = np.zeros((nm, nr))
M_all = dataset[:]
M_mode = np.zeros((nsounding,nm))
M_entropy = np.zeros((nsounding,nm))
M_P= np.zeros((nsounding,n_classes,nm))

t0 = time.time()

for iid in range(nsounding):

    # Get the indices of the rows to use
    ir = np.int64(i_use[iid,:]-1)
    
    # Load ALL DATA AND EXTRACT
    #m_post = dataset[:][ir,:]
    m_post = M_all[ir,:]
    
    # Load only the needed data
    #m_post = np.zeros((nr,nm))
    #for j in range(nr):
    #    m_post[j,:] = dataset[ir[j],:]
    
    #ir = np.sort(ir)
    #m_post = dataset[ir,:] # dows not work when ir contains duplicates
    

    """
    # Compute the mode
    m_mode_single, count_mode = sp.stats.mode(m_post)
    M_mode[iid,:] = m_mode_single
    """
   
    # Compute the class probability
    n_count = np.zeros((n_classes,nm))
    for ic in range(n_classes):
        n_count[ic,:]=np.sum(class_id[ic]==m_post, axis=0)/nr    
    M_P[iid,:,:] = n_count

    # Compute the entropy
    M_entropy[iid,:]=sp.stats.entropy(n_count, base=n_classes)
    """
    for im in range(nm):
        M_entropy[iid,im] = sp.stats.entropy(n_count[:,im], base=n_classes)
    """


t1 = time.time()

print('Elapsed time: %f' % (t1-t0))

#%% 
m_real = m_post[:30]    
# copmute pdf using the the dicrete values in class_id
# Compute histogram
hist, bin_edges = np.histogram(m_real, bins=class_id)

#%% 

plt.figure()
plt.subplot(2,1,1)
plt.imshow(M_mode[0:-1:1,:].T, aspect='auto')
plt.subplot(2,1,2)
plt.imshow(M_entropy[0:-1:1,:].T, aspect='auto')
plt.colorbar()

#m_mean = np.exp(np.mean(np.log(m_post), axis=0))
#m_median = np.median(m_post, axis=0)
#m_std = np.std(np.log10(m_post), axis=0)

#M_mean[iid,:] = m_mean
#M_median[iid,:] = m_median
#M_std[iid,:] = m_std




# %% [markdown]
# ### Plot some statistic from $\sigma(\mathbf{m})$

# %% Posterior analysis
# Plot the Temperature used for inversion
ig.plot_T(f_post_h5)

# %% Plot Profiles
ig.plot_profile_continuous(f_post_h5, i1=1000, i2=2000, im=1)
# %%

# Plot a 2D feature: The number of layers
ig.plot_feature_2d(f_post_h5,im=2,iz=0,key='Median', title_text = 'Number of layers', cmap='jet', s=12)

# Plot a 2D feature: Resistivity in layer 10
ig.plot_feature_2d(f_post_h5,im=1,key='Median', uselog=1, cmap='jet', s=10)
#ig.plot_feature_2d(f_post_h5,im=1,iz=80,key='Median')


# %%
