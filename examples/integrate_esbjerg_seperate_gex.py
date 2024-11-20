#!/usr/bin/env python
# %% [markdown]
# # INTEGRATE on ESBJERG data
# 8 individual tTEM survey wer conducted, using 4 distinct GEX files .
# Below these data are inverted in 4 separate inversions on for each specific GEX file.
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
    # # # # # #%load_ext autoreload
    # # # # # #%autoreload 2
    pass
# %%
import integrate as ig
import numpy as np
import matplotlib.pyplot as plt
# check if parallel computations can be performed
parallel = ig.use_parallel(showInfo=1)
hardcopy = True

# %% Get tTEM data from ESBJERG
f_data_h5_files = ig.get_case_data(case='ESBJERG',loadType='gex' )
nf=4
f_data_h5_files = f_data_h5_files[:nf]



N=500000
# All surveys are inverted with the same prior model
f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='chi2', NLAY_deg=3, RHO_min=1, RHO_max=500)
#f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='uniform', NLAY_min=1, NLAY_max=8, RHO_min=1, RHO_max=500)

plFigs = True

f_post_h5_files = []
f_prior_data_h5_files = []

for i in range(nf):
    f_data_h5 = f_data_h5_files[i]
    #X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)

    file_gex= ig.get_gex_file_from_data(f_data_h5)
    # Extract filename without extension from f_data_h5
    filename = f_data_h5.split('.')[0]
    
    if plFigs:
        ig.plot_geometry(f_data_h5, pl='LINE', hardcopy=hardcopy)
    
    # Compute prior DATA - 
    # Even though the prior model parameters are the same, the prior data diffeer, due to using a different GEX file.
    f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, parallel=parallel, showInfo=0)
    if plFigs:
        ig.plot_data_prior(f_prior_data_h5,f_data_h5,nr=1000,alpha=1, ylim=[1e-13,1e-5], hardcopy=hardcopy)

    # Perform inversion
    N_use = N
    f_post_h5 = ig.integrate_rejection(f_prior_data_h5, 
                                   f_data_h5, 
                                   N_use = N_use, 
                                   showInfo=1, 
                                   Ncpu = 10,
                                   parallel=parallel)

    f_post_h5_files.append(f_post_h5)
    f_prior_data_h5_files.append(f_prior_data_h5)

    # Plot posterior stats
    if plFigs:
        # compare prior an posterior data
        ig.plot_data_prior_post(f_post_h5, i_plot=100, hardcopy=hardcopy);
        ig.plot_data_prior_post(f_post_h5, i_plot=0, hardcopy=hardcopy);        

        # Plot the Temperature used for inversion
        ig.plot_T_EV(f_post_h5, pl='T', hardcopy=hardcopy)
        # Plot the evidence (prior likelihood) estimated as part of inversion
        ig.plot_T_EV(f_post_h5, pl='EV', hardcopy=hardcopy)

        # Plot Profiles
        ig.plot_profile(f_post_h5, im=1, hardcopy=hardcopy)

    # %% Export to CSV
    # f_csv, f_point_csv = ig.post_to_csv(f_post_h5)

#%% 
def merge_posterior(f_post_h5_files, f_data_h5_files, f_post_merged_h5 = ''):
    import h5py

    nf = len(f_post_h5_files)
    # Check that legth of f_data_h5_files is the same as f_post_h5_files
    if len(f_data_h5_files) != nf:
        raise ValueError('Length of f_data_h5_files must be the same as f_post_h5_files')

    if len(f_post_merged_h5) == 0:
        f_post_merged_h5 = 'POST_merged_N%d.h5' % nf

    f_data_merged_h5 = 'DATA_merged_N%d.h5' % nf

    f_data_merged_h5 = ig.merge_data(f_data_h5_files, f_data_merged_h5=f_data_merged_h5)


    for i in range(len(f_post_h5_files)):
        #  get 'i_sample' from the merged file
        f_post_h5 = f_post_h5_files[i]
        with h5py.File(f_post_h5, 'r') as f:
            i_use_s = f['i_use'][:]
            T_s = f['T'][:]
            EV_s = f['EV'][:]
            f_prior_h5 = f['/'].attrs['f5_prior']
            f_data_h5 = f['/'].attrs['f5_data']
            if i == 0:
                i_use = i_use_s
                T = T_s
                EV = EV_s 
            else:
                i_use = np.concatenate((i_use,i_use_s))
                T = np.concatenate((T,T_s))
                EV = np.concatenate((EV,EV_s))

    # Write the merged data to             
    with h5py.File(f_post_merged_h5, 'w') as f:
        f.create_dataset('i_use', data=i_use)
        f.create_dataset('T', data=T)
        f.create_dataset('EV', data=EV)
        f.attrs['f5_prior'] = f_prior_h5
        f.attrs['f5_data'] = f_data_merged_h5
    return f_post_merged_h5


#%% Combine posterior stats into one file plot data again
#ig.merge_posterior(f_post_h5_files, f_post_h5 = 'POST_ESBJERG_ALL_merged.h5')
import h5py

f_data_merged_h5 = ig.merge_data(f_data_h5_files, f_data_merged_h5='ESBJERG_DATA_merged.h5')

for i in range(len(f_post_h5_files)):
    #  get 'i_sample' from the merged file
    f_post_h5 = f_post_h5_files[i]
    with h5py.File(f_post_h5, 'r') as f:
        i_use_s = f['i_use'][:]
        T_s = f['T'][:]
        EV_s = f['EV'][:]
        f_prior_h5 = f['/'].attrs['f5_prior']
        f_data_h5 = f['/'].attrs['f5_data']
        if i == 0:
            i_use = i_use_s
            T = T_s
            EV = EV_s 
        else:
            i_use = np.concatenate((i_use,i_use_s))
            T = np.concatenate((T,T_s))
            EV = np.concatenate((EV,EV_s))

f_post_merged_h5  = 'POST_ESBJERG_ALL_%d_merged.h5' % N
# Write the merged data to             
with h5py.File(f_post_merged_h5, 'w') as f:
    f.create_dataset('i_use', data=i_use)
    f.create_dataset('T', data=T)
    f.create_dataset('EV', data=EV)
    f.attrs['f5_prior'] = f_prior_h5
    f.attrs['f5_data'] = f_data_merged_h5



ig.integrate_posterior_stats(f_post_merged_h5)

#%% 
ig.plot_geometry('ESBJERG_DATA_merged.h5')
ig.plot_profile(f_post_merged_h5, im=1, i1=0, i2=1000, hardcopy=hardcopy)
ig.plot_T_EV(f_post_merged_h5, pl='T', hardcopy=hardcopy)
ig.plot_feature_2d(f_post_merged_h5,im=1,iz=20, key='Median', uselog=1, cmap='jet', s=2, hardcopy=hardcopy)

