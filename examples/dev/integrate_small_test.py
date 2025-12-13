#!/usr/bin/env python
# %% [markdown]
# # INTEGRATE Daugaard Case Study with three eology-resistivity prior models.
#
# This notebook contains an example of inverison of the DAUGAARD tTEM data using three different geology-resistivity prior models

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
    # # # # # # # #%load_ext autoreload
    # # # # # # # #%autoreload 2
    pass

import integrate as ig
import numpy as np
import os
import matplotlib.pyplot as plt
import h5py
hardcopy=True
import time

#%%

# CONSTRUCT NEW PRIOR
MakeNewPrior = True
if MakeNewPrior:
    file1 = 'prior_detailed_invalleys_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5'
    file2 = 'prior_detailed_outvalleys_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5'
    file12 = 'prior_detailed_inout_N4000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5'

    ig.copy_hdf5_file(file1, file12, N=100)

    # Read D1 from f1['D1'], and D2 from f1['D2']. 
    # COmbine D1 and D2 to D12 havig doubvle size of D1 
    # Then replace f12['D1'] with D12
    # Then copy the rest of the data from f1 to f12
    f1 = h5py.File(file1, 'r')
    f2 = h5py.File(file2, 'r')
    f12 =  h5py.File(file12, 'r+')
    N_in = f1['D1'].shape[0]

    print('updating D1')
    D1 = f1['D1'][:]
    D2 = f2['D1'][:]
    D12 = np.concatenate((D1, D2), axis=0)
    del f12['D1']
    f12['D1']=D12

    print('updating M1')
    M1 = f1['M1'][:]
    M2 = f2['M1'][:]
    M12 = np.concatenate((M1, M2), axis=0)
    del f12['M1']
    dataset = f12.create_dataset('M1', data=M12)
    dataset.attrs.update(f1['M1'].attrs)

    print('updating M2')
    M1 = f1['M2'][:]
    M2 = f2['M2'][:]
    M12 = np.concatenate((M1, M2), axis=0)
    del f12['M2']
    #f12['M2']=M12
    dataset = f12.create_dataset('M2', data=M12)
    dataset.attrs.update(f1['M2'].attrs)

    makeM3 = True
    if makeM3:    
        print('creating M3')
        M3a = np.zeros(N_in, dtype=int) + 1
        M3b = np.zeros(N_in, dtype=int) + 2
        M3 = np.concatenate((M3a, M3b), axis=0)
        # Force M3 to be of shape [N,1]
        M3 = M3.reshape(-1,1)
        
        dataset = f12.create_dataset('M3', data=M3, dtype='i4')  # 'i4' represents 32-bit integers
        dataset.attrs['description'] = 'This is an integer dataset'
        dataset.attrs['x'] = np.array([0])
        dataset.attrs['is_discrete'] = 1
        dataset.attrs['class_id'] = [1,2]
        dataset.attrs['class_name'] = ['inside','outside']
        dataset.attrs['cmap'] = [.5, 2.5]
        
    f1.close()
    f2.close()
    f12.close()
    ig.hdf5_scan(file12)


#%%
if MakeNewPrior:
    # use M3 as data
    file12_out, _ = ig.prior_data_identity(file12, im=3, id=0)
    ig.hdf5_scan(file12_out)

#%%
MakeNewData = MakeNewPrior 
if MakeNewData:

    f_data_h5=  'DAUGAARD_AVG.h5'
    X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)

    Nobs = len(X)
    n_classes = 2
    d_obs = np.zeros((Nobs,n_classes))
    x_min = np.min(X)
    x_max = np.max(X)
    dx = x_max - x_min
    for i in range(Nobs):
        d_obs[i,0] = (X[i]-x_min)/dx
        d_obs[i,1] = 1-d_obs[i,0]

    plt.scatter(X,Y, c=d_obs[:,0], s=1, vmin=0, vmax=1)
    plt.colorbar()

    f_data_h5 = ig.copy_hdf5_file('DAUGAARD_AVG.h5', 'DAUGAARD_AVG_inout.h5')
    with h5py.File(f_data_h5, 'a') as f:
        f.create_group('D2')
        # Set attribute noise_model for D2
        f['D2'].attrs['noise_model'] = 'multinomial'
        dataset = f.create_dataset('D2/d_obs', data=d_obs, dtype=np.float64)
        #" write attributes"

    #ig.hdf5_scan(f_data_h5)

# %% SELECT THE CASE TO CONSIDER AND DOWNLOAD THE DATA
loadData = False
if loadData:
    files = ig.get_case_data(case='DAUGAARD', loadType='prior_data') # Load data and prior+data realizations
    f_data_h5 = files[0]
    file_gex= ig.get_gex_file_from_data(f_data_h5)

    f_prior_h5 = 'prior_detailed_invalleys_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5'


f_data_h5 = 'DAUGAARD_RAW.h5'
f_data_h5 = 'DAUGAARD_AVG.h5'
f_data_h5 = 'DAUGAARD_AVG_inout.h5'
f_prior_h5 = 'prior_detailed_invalleys_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5'
f_prior_h5 = 'prior_detailed_outvalleys_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5'
f_prior_h5 = 'prior_detailed_inout_N4000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5'
f_prior_h5 = file12_out
f_prior_h5 = ig.copy_hdf5_file(file12_out, 'prior.h5')

#%%

#f_prior_data_h5 = 'gotaelv2_N1000000_fraastad_ttem_Nh280_Nf12.h5'
updatePostStat =False
N_use = 400000
f_prior_h5='prior.h5'
f_data_h5='DAUGAARD_AVG_inout.h5'
f_post_h5 = ig.integrate_rejection(f_prior_h5, f_data_h5, 
                                N_use = N_use, 
                                parallel=1, 
                                updatePostStat=updatePostStat, 
                                showInfo=1,
                                Nproc = 8)

#%% 
# get geometry
ig.hdf5_scan(f_post_h5)
X, Y, LINE, ELEVATION = ig.get_geometry(f_post_h5)
with h5py.File(f_post_h5, 'r') as f:
    M3_mode = f['/M2/Mode'][:]
    M3_entropy = f['/M3/Entropy'][:]
    M3_P = f['/M2/P'][:]
P=M3_P[:,2,10]
plt.scatter(X,Y, c=P, s=1, vmin=0, vmax=1)
plt.colorbar()

#%% 
ig.plot_profile(f_post_h5, i1=0, i2=2000, cmap='jet', hardcopy=hardcopy)


# %% Likelihood computation
# We need to construct a way to compute the likelihood for a given
# noise model, likelihood_gaussian(d, d_obs, d_std, Cd_inv)
#
# Then we need a general sampling rejection sampler that works
# in either single data points, or a set of data points
# For each data point it should compute the likelihood for all data types. 
# Then apply annealing
# Then combined multiple likelihoos into one.
# Then sample from that likleihood using rejection sampling
# to obtained the indexes..
#

# set which data types to use

def likelihood_gaussian_diagonal(D, d_obs, d_std):
    """
    Compute the likelihood for a single data point
    """
    # Compute the likelihood
    # Sequential
    #L = np.zeros(D.shape[0])
    #for i in range(D.shape[0]):
    #    L[i] = -0.5 * np.nansum(dd[i]**2 / d_std**2)
    # Vectorized
    dd = D - d_obs
    d_var = d_std**2
    L = -0.5 * np.nansum(dd**2 / d_var, axis=1)

    return L

def likelihood_gaussian_full(D, d_obs, Cd):
    a = 1
    return 1

'''
def integrate_rejection_range(f_prior_h5, 
                              f_data_h5, 
                              N_use=1000, 
                              id_use=[1,2], 
                              ip_range=[], 
                              nr=400,
                              autoT=1,
                              T_base = 1,
                              **kwargs):

    from tqdm import tqdm
    import numpy as np
    import h5py
    import time

    # get optional arguments
    showInfo = kwargs.get('showInfo', 0)

    # Get number of data points from, f_data_h5
    with h5py.File(f_data_h5, 'r') as f_data:
        Ndp = f_data['/D1/d_obs'].shape[0]
    # if ip_range is empty then use all data points
    if len(ip_range)==0:
        ip_range = np.arange(Ndp)

    nump=len(ip_range)
    if showInfo>1:
        print('Number of data points to invert: %d' % nump)
    i_use_all = np.zeros((nump, nr), dtype=np.int32)
    T_all = np.zeros(nump)
    EV_all = np.zeros(nump)

    
    with h5py.File(f_prior_h5, 'r') as f_prior:
        N = f_prior['/D1'].shape[0]

    
    if N_use>N:
        N_use = N

    if N_use<N:  
        idx = np.sort(np.random.choice(N, N_use, replace=False))

    i=0
    # GET A LIST OF THE NOISE MODEL TYPE
    noise_model=[]
    with h5py.File(f_data_h5, 'r') as f_data:
        for id in id_use:
            DS = '/D%d' % id
            # if f_data[DS] has noise_model attribute then use it
            if 'noise_model' in f_data[DS].attrs:
                noise_model.append(f_data[DS].attrs['noise_model'])
                if showInfo>0:
                    print('Noise model for %s is %s' % (DS, noise_model[-1]))
            else:
                print('No noise_model attribute in %s' % DS)
                noise_model.append('none')
                        
    
    # load the 'mulitiple' fdata 
    # consider making it available as shared data
    D = []
    doRandom=False    
    with h5py.File(f_prior_h5, 'r') as f_prior:
        for id in id_use:
            DS = '/D%d' % id
            N = f_prior[DS].shape[0]
            #print('Reading %s' % DS)
            if N_use<N:
                if doRandom:
                    print('Start Reading %s ' % DS)
                    Dsub = f_prior[DS][idx]
                    print('End Reading %s ' % DS)
                else:
                    Dsub = f_prior[DS][0:N_use]
                D.append(Dsub)
            else:        
                D.append(f_prior[DS][:])

            #print(D[-1].shape)

    # THIS IS THE ACTUAL INVERSION!!!!
    for j in tqdm(range(len(ip_range))):
        ip = ip_range[j]
        t=[]
        N = D[0].shape[0]
        NDsets = len(id_use)
        L = np.zeros((NDsets, N))

        for i in range(len(D)):
            t0=time.time()
            id = id_use[i]
            DS = '/D%d' % id
            if noise_model[i]=='gaussian':
                with h5py.File(f_data_h5, 'r') as f_data:
                    d_obs = f_data['%s/d_obs' % DS][ip]
                    d_std = f_data['%s/d_std' % DS][ip] * (1+i*0.1)

                L_single = likelihood_gaussian_diagonal(D[i], d_obs, d_std)
                #L.append(L_single)
                L[i] = L_single
                t.append(time.time()-t0)
            elif noise_model[i]=='multinomial':
                with h5py.File(f_data_h5, 'r') as f_data:
                    d_obs = f_data['%s/d_obs' % DS][ip]
                    class_id = [1,2]

                    useVetorized = True
                    if useVetorized:
                        D_ind = np.zeros(D[i].shape[0], dtype=int)
                        D_ind[:] = np.searchsorted(class_id, D[i].squeeze())
                        L_single = np.log(d_obs[D_ind])
                    else:
                        D_ind = np.zeros(D[id].shape[0], dtype=int)
                        for i in range(D_ind.shape[0]):
                            for j in range(len(class_id)):
                                if D[id][i]==class_id[j]:
                                    D_ind[i]=j
                                    break
                        L_single = np.zeros(D[id].shape[0])

                        for i in range(D_ind.shape[0]):
                            L_single[i] = np.log(d_obs[D_ind[i]])

                L[i] = L_single           
                t.append(time.time()-t0)

            else: 
                # noise model not regcognized
                # L_single = -1
                pass

        
        t0=time.time()


        # NOw we have all the likelihoods for all data types. Copmbine them into ooe
        L_single = L
        L = np.sum(L_single, axis=0)
        #plt.plot(L.T)


        # AUTO ANNEALE
        t0=time.time()
        #autoT=1
        # Compute the annealing temperature
        if autoT == 1:
            T = ig.logl_T_est(L)
        else:
            T = T_base        
        # maxlogL = np.nanmax(logL)
        t.append(time.time()-t0)

        # Find ns realizations of the posterior, using the log-likelihood values logL, and the annealing tempetrature T 
        
        P_acc = np.exp((1/T) * (L - np.nanmax(L)))
        P_acc[np.isnan(P_acc)] = 0

        # Select the index of P_acc propportion to the probabilituy given by P_acc
        t0=time.time()
        try:
            i_use = np.random.choice(N, nr, p=P_acc/np.sum(P_acc))
        except:
            print('Error in np.random.choice for ip=%d' % ip)   
            i_use = np.random.choice(N, nr)
        t.append(time.time()-t0)
        
        # find the number of unique indexes
        n_unique = len(np.unique(i_use))


        # Compute the evidence
        maxlogL = np.nanmax(L)
        exp_logL = np.exp(L - maxlogL)
        EV = maxlogL + np.log(np.nansum(exp_logL)/len(L))

        t.append(time.time()-t0)

        i_use_all[j] = i_use
        T_all[j] = T
        EV_all[j] = EV


        if showInfo>1:
            for i in range(len(t)):
                if i<len(D):
                    print(' Time id%d: %f - %s' % (i,t[i],noise_model[i]))
                else:
                    print(' Time id%d, sampling: %f' % (i,t[i]))
            print('Time total: %f' % np.sum(t))
        
    return i_use_all, T_all, EV_all, ip_range
'''
'''
def integrate_posterior_chunk(args):
    i_chunk, ip_chunks, f_prior_h5, f_data_h5, N_use, id_use, autoT, T_base, nr = args
    ip_range = ip_chunks[i_chunk]
    #print(f'Chunk {i_chunk+1}/{len(ip_chunks)}, ndp={len(ip_range)}')

    i_use, T, EV, ip_range = integrate_rejection_range(
        f_prior_h5=f_prior_h5,
        f_data_h5=f_data_h5,
        N_use=N_use,
        id_use=id_use,
        ip_range=ip_range,
        autoT=autoT,
        T_base=T_base,
        nr=nr,
    )

    return i_use, T, EV, ip_range

def integrate_posterior_main(ip_chunks, f_prior_h5, f_data_h5, N_use, id_use, autoT, T_base, nr, Ncpu):
    from multiprocessing import Pool

    with Pool(Ncpu) as p:
        results = p.map(integrate_posterior_chunk, [(i, ip_chunks, f_prior_h5, f_data_h5, N_use, id_use, autoT, T_base, nr) for i in range(len(ip_chunks))])

    # Get sample size N from f_prior_h5
    with h5py.File(f_prior_h5, 'r') as f_prior:
        N = f_prior['/D1'].shape[0]

    # Get number of data points from, f_data_h5
    with h5py.File(f_data_h5, 'r') as f_data:
        Ndp = f_data['/D1/d_obs'].shape[0]

    i_use_all = np.random.randint(0, N, (Ndp, nr))
    T_all = np.zeros(Ndp)*np.nan
    EV_all = np.zeros(Ndp)*np.nan
    
    for i, (i_use, T, EV, ip_range) in enumerate(results):
        for i in range(len(ip_range)):
                ip = ip_range[i]
                #print('ip=%d, i=%d' % (ip,i))
                i_use_all[ip] = i_use[i]
                T_all[ip] = T[i]
                EV_all[ip] = EV[i]

    return i_use_all, T_all, EV_all
'''
'''
def integrate_rejection_multi(f_post_h5='post.h5',
                              f_prior_h5='prior.h5', 
                              f_data_h5='DAUGAARD_AVG_inout.h5', 
                              N_use=1000, 
                              id_use=[1,2], 
                              ip_range=[], 
                              nr=400,
                              autoT=1,
                              T_base = 1,
                              Nchunks=0,
                              Ncpu=1,
                              useParallel=True,
                              **kwargs):
    from datetime import datetime   
    from multiprocessing import Pool

    # get optional arguments
    showInfo = kwargs.get('showInfo', 0)

    # Get sample size N from f_prior_h5
    with h5py.File(f_prior_h5, 'r') as f_prior:
        N = f_prior['/D1'].shape[0]

    # Get number of data points from, f_data_h5
    with h5py.File(f_data_h5, 'r') as f_data:
        Ndp = f_data['/D1/d_obs'].shape[0]
        print('Number of data points: %d' % Ndp)    
    # if ip_range is empty then use all data points
    if len(ip_range)==0:
        ip_range = np.arange(Ndp)
    Ndp_invert = len(ip_range)
    print('Number of data points to invert: %d' % Ndp_invert)
    

    # set i_use_all to be a 2d Matrie of size (nump,nr) of random integers in range(N)
    i_use_all = np.random.randint(0, N, (Ndp, nr))
    T_all = np.zeros(Ndp)
    EV_all = np.zeros(Ndp)

    date_start = str(datetime.now())
    t_start = datetime.now()
    

    if Nchunks==0:
        if useParallel:
            Nchunks = Ncpu
        else:   
            Nchunks = 1
    ip_chunks = np.array_split(ip_range, Nchunks) 

    # PERFORM INVERSION PERHAPS IN PARALLEL

    if useParallel:
        i_use_all, T_all, EV_all = integrate_posterior_main(
            ip_chunks=ip_chunks,
            f_prior_h5=f_prior_h5,
            f_data_h5=f_data_h5,
            N_use=N_use,
            id_use=id_use,
            autoT=autoT,
            T_base=T_base,
            nr=nr,
            Ncpu=Ncpu,
        )


    else:

        for i_chunk in range(len(ip_chunks)):        
            ip_range = ip_chunks[i_chunk]
            print('Chunk %d/%d, ndp=%d' % (i_chunk+1, len(ip_chunks), len(ip_range)))

            i_use, T, EV, ip_range = ig.integrate_rejection_range(f_prior_h5=f_prior_h5, 
                                        f_data_h5=f_data_h5,
                                        N_use=N_use, 
                                        id_use=id_use,
                                        ip_range=ip_range,
                                        autoT=autoT,
                                        T_base = T_base,
                                        nr=nr,
                                        )
        
            for i in range(len(ip_range)):
                ip = ip_range[i]
                #print('ip=%d, i=%d' % (ip,i))
                i_use_all[ip] = i_use[i]
                T_all[ip] = T[i]
                EV_all[ip] = EV[i]

    # WHere T_all is Inf set it to Nan
    T_all[T_all==np.inf] = np.nan
    EV_all[EV_all==np.inf] = np.nan

    date_end = str(datetime.now())
    t_end = datetime.now()
    t_elapsed = (t_end - t_start).total_seconds()
    t_per_sounding = t_elapsed / Ndp_invert
    if (showInfo>-1):
        print('T_av=%3.1f, Time=%5.1fs/%d soundings ,%4.1fms/sounding, %3.1fit/s' % (np.nanmean(T_all),t_elapsed,Ndp_invert,t_per_sounding*1000,Ndp_invert/t_elapsed))

    # SAVE THE RESULTS to f_post_h5
    with h5py.File(f_post_h5, 'w') as f_post:
        f_post.create_dataset('i_use', data=i_use_all)
        f_post.create_dataset('T', data=T_all)
        f_post.create_dataset('EV', data=EV_all)
        #f_post.create_dataset('ip_range', data=ip_range)
        f_post.attrs['date_start'] = date_start
        f_post.attrs['date_end'] = date_end
        f_post.attrs['inv_time'] = t_elapsed
        f_post.attrs['f5_prior'] = f_prior_h5
        f_post.attrs['f5_data'] = f_data_h5
        f_post.attrs['N_use'] = N_use

    updatePostStat=False
    if updatePostStat:
        ig.integrate_posterior_stats(f_post_h5, **kwargs)

    return T_all, EV_all, i_use_all, 
'''

#%% The new version of integrate_rejection using multidata
updatePostStat =False
N_use = 100000
f_prior_h5='prior.h5'
f_data_h5='DAUGAARD_AVG_inout.h5'
Ncpu = 16

ip_range = []
ip_range=np.arange(0,1000,1)   
f_post_h5 = 'post.h5'
T, E, i_use = ig.integrate_rejection_multi(f_post_h5=f_post_h5,
                            f_prior_h5=f_prior_h5, 
                            f_data_h5=f_data_h5, 
                            N_use=N_use, 
                            id_use=[1,2],
                            T_base = 1,
                            autoT=1,
                            ip_range=ip_range,
                            Nchunks=0,
                            Ncpu=Ncpu,
                            useParallel=True,
                            updatePostStat=updatePostStat,                            
                            )

ig.plot_T_EV(f_post_h5, pl='T')

#%%  TEST OLD
f_post_h5 = ig.integrate_rejection(f_prior_h5, f_data_h5, 
                                N_use = N_use, 
                                parallel=1, 
                                updatePostStat=updatePostStat, 
                                showInfo=1,
                                Nproc = Ncpu)



#%%
ig.integrate_posterior_stats(f_post_h5)
ig.plot_profile(f_post_h5, i1=0, i2=2000, cmap='jet', hardcopy=hardcopy)


#%% test chunks
f_post_h5 = 'post.h5'
f_prior_h5 = 'prior.h5'
f_data_h5 = 'DAUGAARD_AVG_inout.h5'
N_use = 4000
id_use = [1, 2]
T_base = 1
autoT = 1
ip_range = np.arange(0, 2000)
nr = 400
Nchunks = 2
Ncpu = 2

ip_chunks = np.array_split(ip_range, Nchunks)
#i_use_all, T_all, EV_all = integrate_posterior_main(
i_use_all, T_all, EV_all = integrate_posterior_main(
    ip_chunks=ip_chunks,
    f_prior_h5=f_prior_h5,
    f_data_h5=f_data_h5,
    N_use=N_use,
    id_use=id_use,
    autoT=autoT,
    T_base=T_base,
    nr=nr,
)

# %%
X, Y, LINE, ELEVATION = ig.get_geometry(f_post_h5)
# read Mode from M3 in f_post_h5
with h5py.File(f_post_h5, 'r') as f:
    M3_mode = f['/M3/Mode'][:]
    M3_entropy = f['/M3/Entropy'][:]
    M3_P = f['/M3/P'][:]
    M2_entropy = f['/M2/Entropy'][:]
    
plt.scatter(X,Y, c=np.mean(M2_entropy, axis=1), s=1)
plt.scatter(X,Y, c=np.mean(M2_entropy, axis=1), s=1)
plt.colorbar()    



# %%
ig.plot_T_EV(f_post_h5, pl='T')
# %%
with h5py.File(f_post_h5, 'r') as f_post:
    EV = f_post['/EV'][:]

# %%

# %% direct computation of the likelihood
ip_range = []
ip_range=np.arange(100,1000)   
i_use, T, EV, ip_range = integrate_rejection_range(f_prior_h5='prior.h5', 
                                     f_data_h5='DAUGAARD_AVG_inout.h5', 
                                     N_use=4000, 
                                     id_use=[1,2],
                                     ip_range=ip_range,
                                     autoT=0,
                                     T_base = 1000,
                                     )

plt.figure()
plt.plot(EV,label='EV')
plt.plot(T,label='T')
plt.legend()
plt.show()

# %%
