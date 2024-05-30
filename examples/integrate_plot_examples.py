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
    #%load_ext autoreload
    #%autoreload 2
    pass

# %%
import integrate as ig
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import h5py
# %%

f_post_h5 = 'POST_Fra20200930_202001001_1_AVG_export_PRIOR_CHI2_NF_5_log-uniform_N5000000_fraastad_ttem_Nh280_Nf12_Nu1000000_aT1.h5'
#f_post_h5 = 'POST_Fra20200930_202001001_1_AVG_export_PRIOR_UNIFORM_NL_1-8_log-uniform_N5000000_fraastad_ttem_Nh280_Nf12_Nu1000000_aT1.h5'
#f_post_h5 = 'POST_Fra20200930_202001001_1_AVG_export_PRIOR_UNIFORM_NL_8-8_log-uniform_N5000000_fraastad_ttem_Nh280_Nf12_Nu1000000_aT1.h5'
f_post_h5 = 'POST_Fra20200930_202001001_1_AVG_export_PRIOR_WB5_N500000_log-uniform_R1_800.h5_fraastad_ttem_Nh280_Nf12_Nu500000_aT1.h5'
f_post_h5 = 'POST_Fra20200930_202001001_1_AVG_export_PRIOR_WB12_N50000_log-uniform_R1_800.h5_fraastad_ttem_Nh280_Nf12_Nu50000_aT1.h5'
f_post_h5 = 'POST_DAUGAARD_AVG_PRIOR_Daugaard_N2000000_TX07_20230731_2x4_RC20-33_Nh280_Nf12_Nu1000_aT1.h5'
f_post = h5py.File(f_post_h5,'r')
f_prior_h5 = f_post['/'].attrs['f5_prior']
f_data_h5 = f_post['/'].attrs['f5_data']


ig.plot_prior_stats(f_prior_h5)

#m2 = ax[1,0].imshow(M[0:nr,:].T, aspect='auto')
#fig.colorbar(m2, ax=ax[1,1])

# %% DATA
#ig.plot_data(f_data_h5, i_plot=[], Dkey='D1', hardcopy=False)
##ig.plot_data(f_data_h5, i_plot=[], Dkey='D1', hardcopy=False)
#ig.plot_data(f_data_h5, i_plot=3000+np.arange(4000), Dkey='D1', hardcopy=False)
ig.plot_data(f_data_h5)
#ig.plot_data(f_data_h5, i_plot=15000+np.arange(500))

#%%
ig.plot_T_EV(f_post_h5, pl='T')
ig.plot_T_EV(f_post_h5, pl='EV')
ig.plot_T_EV(f_post_h5, pl='ND', s=5)
#ig.plot_T_EV(f_post_h5, pl='all', s=5)


#f_post.close()
#f_data.close()

# %%
## plot prior post data

ig.plot_data_prior_post(f_post_h5, i_plot = 0)
ig.plot_data_prior_post(f_post_h5, i_plot = 1199)


#%% plot profile
ig.plot_profile_continuous(f_post_h5,i1=0, i2=2000, clim=[1,1000])

#%% 
