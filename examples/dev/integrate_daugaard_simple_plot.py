#!/usr/bin/env python
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
    # # # # # # # # #%load_ext autoreload
    # # # # # # # # #%autoreload 2
    pass

import integrate as ig
import numpy as np
import os
import matplotlib.pyplot as plt
import h5py
hardcopy=True

#%% 
f_post_h5_arr=[]
f_post_h5_arr.append('POST_DAUGAARD_AVG_prior_detailed_general_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12_Nu2000000_aT1.h5')
f_post_h5_arr.append('POST_DAUGAARD_AVG_prior_detailed_general_N2000000_dmax90_TX07_20231016_2x4_RC20-33_N500000_Nh280_Nf12_Nu500000_aT1.h5')
f_post_h5_arr.append('POST_DAUGAARD_AVG_prior_detailed_general_N2000000_dmax90_TX07_20231016_2x4_RC20-33_N10000_Nh280_Nf12_Nu10000_aT1.h5')
f_post_h5_arr.append('POST_DAUGAARD_AVG_PRIOR_WB30_N500000_log-uniform_R10_500_TX07_20231016_2x4_RC20-33_Nh280_Nf12_Nu500000_aT1.h5')
f_post_h5_arr.append('POST_DAUGAARD_AVG_PRIOR_UNIFORM_NL_1-12_log-uniform_N500000_TX07_20231016_2x4_RC20-33_Nh280_Nf12_Nu500000_aT1.h5')
f_post_h5_arr.append('POST_DAUGAARD_AVG_PRIOR_UNIFORM_NL_1-12_log-uniform_N2000000_TX07_20231016_2x4_RC20-33_Nh280_Nf12_Nu2000000_aT1.h5');

i1= 24;i2 = 54
i1= 876;i2 = 937
#i1= 11169;i2 = 11195

#%% 

for f_post_h5 in f_post_h5_arr:

    with h5py.File(f_post_h5,'r') as f_post:
        f_prior_h5 = f_post['/'].attrs['f5_prior']
        f_data_h5 = f_post['/'].attrs['f5_data']

    # plot prior stats
    ig.plot_prior_stats(f_prior_h5, hardcopy=hardcopy)
    plt.show()

    # plot EV and T
    ig.plot_T_EV(f_post_h5, pl='all', s=10, hardcopy=hardcopy)
    #ig.plot_T_EV(f_post_h5, pl='EV', s=5, hardcopy=hardcopy)
    plt.show()


#%% Plot prior and posterior data

for f_post_h5 in f_post_h5_arr:
    with h5py.File(f_post_h5,'r') as f_post:
        f_prior_h5 = f_post['/'].attrs['f5_prior']
        f_data_h5 = f_post['/'].attrs['f5_data']

    with h5py.File(f_data_h5,'r') as f_prior:
        nd=len(f_prior['UTMX'][:].flatten())

    i_p = np.linspace(0,nd-1,4).astype(int)


    for i in i_p:
        ig.plot_data_prior_post(f_post_h5, i_plot = i, nr=400, hardcopy=hardcopy)
        a=1
        
    # Plot prior and data on the same plot
    ig.plot_data_prior_post(f_post_h5,nr=400, hardcopy=hardcopy)
    plt.show()



#%% PLOT XY
fig = ig.plot_data_xy(f_data_h5, pl_type='elevation', hardcopy=hardcopy)
X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)
i1= 876;i2 = 937
ii=np.arange(i1,i2,1)
plt.plot(X[ii]/1000, Y[ii]/1000, 'r.', markersize=11)


fig = ig.plot_data_xy(f_data_h5, pl_type='elevation', hardcopy=hardcopy)
plt.show()

fig = ig.plot_data_xy(f_data_h5, pl_type='line', hardcopy=hardcopy)
plt.show()

fig = ig.plot_data_xy(f_data_h5, hardcopy=hardcopy)
plt.show()


#%%
for f_post_h5 in f_post_h5_arr:

    # plot prior profile 
    ig.integrate_posterior_stats(f_post_h5, usePrior=True)
    ig.plot_profile(f_post_h5, i1=i1, i2=i2, hardcopy=hardcopy, txt='_prior')
    # plot posterior profile
    ig.integrate_posterior_stats(f_post_h5)
    ig.plot_profile(f_post_h5, i1=i1, i2=i2, hardcopy=hardcopy, txt='_post')

    ig.plot_profile(f_post_h5, hardcopy=hardcopy, txt='_post_all')

    ig.plot_profile(f_post_h5, i1=0, i2=2000, hardcopy=hardcopy, txt='_post_all')


#%%
#%% post M2
for f_post_h5 in f_post_h5_arr:

    try:
        ig.plot_feature_2d(f_post_h5,im=2,key='Median', title_text = 'Number of layers', uselog=0, clim=[.5, 12.5],cmap='jet_r',  s=12, hardcopy=hardcopy)        
        ig.plot_feature_2d(f_post_h5,im=2,key='Mean', title_text = 'Number of layers', uselog=0, clim=[3.5, 9.5],cmap='jet_r',  s=12, hardcopy=hardcopy)        
        plt.show()
    except:
        pass

#%% Finally compute the mean thickness of XX above ZZ meters depth
for f_post_h5 in f_post_h5_arr:
    with h5py.File(f_post_h5,'r') as f_post:
        try:
            f_prior_h5 = f_post['/'].attrs['f5_prior']
            f_data_h5 = f_post['/'].attrs['f5_data']

            
            X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)
            with h5py.File(f_prior_h5,'r') as f_prior:
                M2 = f_prior['/M2'][:]
                class_name = f_prior['/M2'].attrs['class_name']
                class_id = f_prior['/M2'].attrs['class_id']
            
            i_use = f_post['i_use'][:]

            nd = i_use.shape[0]
            mean_post = np.zeros(nd)
            std_post = np.zeros(nd)
            p_post = np.zeros(nd)
            i_cat = np.array([2,3,6]) # Gravel AND SAND
            i_cat = np.array([3]) # gravel
            i_cat = np.array([2,6]) # SAND
            
            txt=''
            for i in np.arange(len(i_cat)): 
                print(i)
                # find index of class_name equal to i_cat[i]    
                j = np.where(class_id == i_cat[i])
                txt += '%s[%d], ' % (class_name[j],i_cat[i] )
                print(txt)
            print(txt)

            for i in np.arange(0,nd,1):
                post  = M2[i_use[i]]
                # count many times one of the categories in i_cat is present in post
                thick = np.sum(np.isin(post,i_cat), axis=1);
                # find ratio of entires in mean larger than 20
                p_post[i] = np.sum(thick>20)/thick.size
                mean_post[i] = np.mean(thick)
                std_post[i] = np.std(thick)

            fig = plt.figure(figsize=(10,8))            
            plt.scatter(X,Y,c=mean_post, cmap='jet', s=5)
            plt.grid()
            plt.colorbar()
            plt.axis('equal')
            plt.title('Mean cumulative thickness  [m] of %s'%txt)    
            plt.savefig('%s_%s_cumThick.png' % (os.path.splitext(f_post_h5)[0],'M2'))

    
            fig = plt.figure(figsize=(10,8))            
            plt.scatter(X,Y,c=std_post, cmap='jet', s=5)
            plt.grid()
            plt.colorbar()
            plt.axis('equal')
            plt.title('Std cumulative thickness [s] of %s'%txt)    
            plt.savefig('%s_%s_cumThickStd.png' % (os.path.splitext(f_post_h5)[0],'M2'))

            fig = plt.figure(figsize=(10,8))
            plt.scatter(X,Y,c=p_post, cmap='viridis', s=5, vmin=0, vmax=1)
            plt.grid()
            plt.colorbar()
            plt.axis('equal')
            plt.title('Probability of cumulative thickness > 20 [m] of %s'%txt)
            plt.savefig('%s_%s_cumThickProb.png' % (os.path.splitext(f_post_h5)[0],'M2'))
    

        except:
            pass




# %%
