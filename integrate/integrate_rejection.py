"""
Rejection Sampling Module for INTEGRATE

This module contains functions for Bayesian inversion using rejection sampling methodology.
It includes the main rejection sampling algorithm, likelihood calculations, and parallel 
processing support for efficient posterior sampling.

Key Functions:
- integrate_rejection(): Main rejection sampling function
- integrate_rejection_range(): Core rejection sampling for data point ranges  
- likelihood_*(): Various likelihood calculation functions
- Shared memory functions for parallel processing
"""

import numpy as np
import h5py
import os
import time
import multiprocessing
from multiprocessing import shared_memory
from datetime import datetime
from tqdm import tqdm
import logging

# Set up logging
logger = logging.getLogger(__name__)

def integrate_rejection(f_prior_h5='prior.h5', 
                              f_data_h5='DAUGAAD_AVG_inout.h5',
                              f_post_h5='',                              
                              N_use=100000000000, 
                              id_use=[], 
                              ip_range=[], 
                              nr=400,
                              autoT=1,
                              T_base = 1,
                              Nchunks=0,
                              Ncpu=0,
                              parallel=True,
                              use_N_best=0,  
                              **kwargs):
    """
    Perform probabilistic inversion using rejection sampling.
    
    This is the main function for Bayesian inversion using rejection sampling methodology.
    It samples the posterior distribution by rejecting prior samples that are inconsistent 
    with observed data within a temperature-controlled tolerance. Supports parallel processing
    and automatic temperature estimation for efficient sampling.
    
    Parameters
    ----------
    f_prior_h5 : str, optional
        Path to HDF5 file containing prior model and data samples.
        Default is 'prior.h5'.
    f_data_h5 : str, optional
        Path to HDF5 file containing observed data for inversion.
        Default is 'DAUGAAD_AVG_inout.h5'.
    f_post_h5 : str, optional
        Output path for posterior samples. If empty, auto-generated from prior filename.
        Default is empty string.
    N_use : int, optional
        Maximum number of prior samples to use for inversion.
        Default is 100000000000.
    id_use : list, optional
        List of data identifiers to use for inversion. If empty, uses all available data.
        Default is empty list.
    ip_range : list, optional
        List of data point indices to invert. If empty, inverts all data points.
        Default is empty list.
    nr : int, optional
        Number of posterior samples to retain per data point.
        Default is 400.
    autoT : int, optional
        Automatic temperature estimation method (1=enabled, 0=disabled).
        Default is 1.
    T_base : float, optional
        Base temperature for rejection sampling when autoT=0.
        Default is 1.
    Nchunks : int, optional
        Number of chunks for parallel processing. If 0, auto-determined.
        Default is 0.
    Ncpu : int, optional
        Number of CPU cores to use. If 0, auto-determined from system.
        Default is 0.
    parallel : bool, optional
        Enable parallel processing if environment supports it.
        Default is True.
    use_N_best : int, optional
        Use only the N best-fitting samples (0=disabled).
        Default is 0.
    **kwargs : dict
        Additional keyword arguments including showInfo, updatePostStat, post_dir.
    
    Returns
    -------
    str
        Path to the output HDF5 file containing posterior samples and statistics.
    
    Notes
    -----
    The function automatically determines optimal processing parameters based on data size
    and system capabilities. Temperature annealing is used to improve sampling efficiency.
    
    Large datasets may require significant memory and processing time. Monitor system
    resources during execution.
    
    Examples
    --------
    >>> import integrate as ig
    >>> f_post = ig.integrate_rejection('prior.h5', 'data.h5', N_use=10000)
    >>> print(f"Results saved to: {f_post}")
    """
    import integrate as ig
    
    # get optional arguments
    showInfo = kwargs.get('showInfo', 0)
    updatePostStat = kwargs.get('updatePostStat', True)
    # If set, Nproc will be used as the number of processors
    Ncpu = kwargs.get('Nproc', Ncpu)
    Ncpu = kwargs.get('N_cpu', Ncpu) # Allow using N_cpu instead of Ncpu
    Nchunks = kwargs.get('N_chunks', Nchunks) # Allow using N_chunks instead of Nchunks
    posterior_output_path = kwargs.get('post_dir', os.getcwd())

    if Ncpu < 1 :
        Ncpu =  int(multiprocessing.cpu_count())
        # Set Ncpu to be min of Ncpu and 8
        # as no gain is expected from using more than 8 processors
        Ncpu = min(Ncpu, 8) 


    # Set default f_post_h5 filename if not set    
    if len(f_post_h5)==0:
        # Extract the base name of f_prior_h5 without its path or extension
        f_prior_basename = os.path.splitext(os.path.basename(f_prior_h5))[0]
        
        # Construct the new filename
        f_post_h5 = os.path.join(posterior_output_path, "POST_%s_Nu%d_aT%d.h5" % (f_prior_basename, N_use, autoT))

    # Check that f_post_h5 allready exists, and warn the user   
    if os.path.isfile(f_post_h5):
        if (showInfo>0):    
            print('File %s allready exists' % f_post_h5)
            print('Overwriting...')    

    
    # Load observed data from f_data_h5
    DATA = ig.load_data(f_data_h5, showInfo=showInfo)
    Ndt = len(DATA['d_obs']) # Number of data types
    if len(id_use)==0:
        id_use = np.arange(1,Ndt+1) 
    Ndt = len(id_use) # Number of data types used        
    # Perhaps load only the data types that are used
    DATA = ig.load_data(f_data_h5, id_arr=id_use, showInfo=showInfo)

    if (showInfo>0):    
        for i in range(Ndt):
            print('Data type %d: %s, Using %d/%d data' % (id_use[i], DATA['noise_model'][i], np.sum(DATA['i_use'][i]), len(DATA['i_use'][i])))

    # Load the prior data from the h5 files
    #D = load_prior_data(f_prior_h5, id_use = id_use, N_use = N_use, Randomize=True)[0]
    id_data_use = 1+np.arange(np.max(DATA['id_use']))
    D, idx = ig.load_prior_data(f_prior_h5, id_use = id_data_use, N_use = N_use, Randomize=True)
    # Convert D to 32 bit float
    #D = [d.astype(np.float32) for d in D]
    #D = [d.astype(np.float128) for d in D]
    # Pritn the memory size of D
    if showInfo>0:
        print('Memory size of D: %s' % str(np.array(D).nbytes))   

    #D, idx = load_prior_data(f_prior_h5, id_use = id_use, N_use = N_use, Randomize=True)
    # M, idx = load_prior_model(f_prior_h5, idx=idx, N_use=N_use, Randomize=True)

    # Get sample size N from f_prior_h5
    N = D[0].shape[0]
    if N_use>N:
        N_use = N



    # Get number of data points from, f_data_h5
    Ndp = DATA['d_obs'][0].shape[0]
    
    # if ip_range is empty then use all data points
    if len(ip_range)==0:
        ip_range = np.arange(Ndp)
    Ndp_invert = len(ip_range)
            
        
    if Ncpu ==1:
        parallel = False

    
    if showInfo>0:
        print('<--INTEGRATE_REJECTION-->')
        print('f_prior_h5=%s\nf_data_h5=%s\nf_post_h5=%s' % (f_prior_h5, f_data_h5, f_post_h5))
    
    if showInfo>1:
        print('Number of data points: %d (available), %d (used). Nchunks=%s, Ncpu=%d,use_N_best=%d' % (Ndp,Ndp_invert,Nchunks,Ncpu,use_N_best))    
        print('N_use = %d' % (N_use))
        print('use_N_best=%d' % use_N_best)
        print('Number of data types: %d' % Ndt)
        print('Using these data types: %s' % str(id_use))
        print('Loading these prior data model types:', str(id_data_use))
    
    
    # set i_use_all to be a 2d Matrie of size (nump,nr) of random integers in range(N)
    i_use_all = np.random.randint(0, N, (Ndp, nr))
    N_UNIQUE_all = np.zeros(Ndp)*np.nan
    T_all = np.zeros(Ndp)*np.nan
    EV_all = np.zeros(Ndp)*np.nan
    # 'posterior' evience - mean posterior likelihood TODO
    EV_post_all  = np.zeros(Ndp)*np.nan

    date_start = str(datetime.now())
    t_start = datetime.now()
    
    # PERFORM INVERSION PERHAPS IN PARALLEL

    if parallel:
        # Split the ip_range into Nchunks
        if Nchunks==0:
            if parallel:
                Nchunks = Ncpu
            else:   
                Nchunks = 1
        ip_chunks = np.array_split(ip_range, Nchunks) 

        if showInfo>1:
            print('Ncpu = %d\nNchunks=%d' % (Ncpu, Nchunks))

        i_use_all, T_all, EV_all, EV_post_all, N_UNIQUE_all = integrate_posterior_main(
            ip_chunks=ip_chunks,
            D=D, 
            DATA = DATA,
            idx = idx,  
            N_use=N_use,
            id_use=id_use,
            autoT=autoT,
            T_base=T_base,
            nr=nr,
            Ncpu=Ncpu,
            use_N_best=use_N_best            
        )


    else:

        #for i_chunk in range(len(ip_chunks)):        
        #    ip_range = ip_chunks[i_chunk]
        #    if showInfo>0:
        #        print('Chunk %d/%d, ndp=%d' % (i_chunk+1, len(ip_chunks), len(ip_range)))

            i_use, T, EV, EV_post, N_UNIQUE, ip_range = integrate_rejection_range(D=D, 
                                        DATA = DATA,
                                        idx = idx,                                   
                                        N_use=N_use, 
                                        id_use=id_use,
                                        ip_range=ip_range,
                                        autoT=autoT,
                                        T_base = T_base,
                                        nr=nr,
                                        use_N_best=use_N_best,
                                        **kwargs
                                        )
        
            for i in range(len(ip_range)):
                ip = ip_range[i]
                #print('ip=%d, i=%d' % (ip,i))
                i_use_all[ip] = i_use[i]
                T_all[ip] = T[i]
                EV_all[ip] = EV[i]
                EV_post_all[ip] = EV_post[i]
                N_UNIQUE_all[ip] = N_UNIQUE[i]

    # WHere T_all is Inf set it to Nan
    T_all[T_all==np.inf] = np.nan
    EV_all[EV_all==np.inf] = np.nan
    
    date_end = str(datetime.now())
    t_end = datetime.now()
    t_elapsed = (t_end - t_start).total_seconds()
    t_per_sounding = t_elapsed / Ndp_invert
    if (showInfo>-1):
        print('integrate_rejection: Time=%5.1fs/%d soundings, %4.1fms/sounding, %3.1fit/s' % (t_elapsed,Ndp_invert,t_per_sounding*1000,Ndp_invert/t_elapsed), end='')
        print('integrate_rejection: T_av=%3.1f, EV_av=%3.1f' % (np.nanmean(T_all), np.nanmean(EV_all)))

    # SAVE THE RESULTS to f_post_h5
    with h5py.File(f_post_h5, 'w') as f_post:
        f_post.create_dataset('i_use', data=i_use_all)
        f_post.create_dataset('T', data=T_all)
        f_post.create_dataset('EV', data=EV_all)
        f_post.create_dataset('EV_post', data=EV_post_all)
        f_post.create_dataset('N_UNIQUE', data=N_UNIQUE_all)
        #f_post.create_dataset('ip_range', data=ip_range)
        f_post.attrs['date_start'] = date_start
        f_post.attrs['date_end'] = date_end
        f_post.attrs['inv_time'] = t_elapsed
        f_post.attrs['f5_prior'] = f_prior_h5
        f_post.attrs['f5_data'] = f_data_h5
        f_post.attrs['N_use'] = N_use

    if updatePostStat:
        ig.integrate_posterior_stats(f_post_h5, **kwargs)

    #return f_post_h5 T_all, EV_all, i_use_all
    return f_post_h5



def integrate_rejection_range(D, 
                              DATA, 
                              idx = [],
                              N_use=1000, 
                              id_use=[1,2], 
                              ip_range=[], 
                              nr=400,
                              autoT=1,
                              T_base = 1,
                              **kwargs):
    """
    Perform rejection sampling for a specific range of data points.
    
    This function implements the core rejection sampling algorithm for a subset of data points.
    It evaluates likelihood for each data point in the range and accepts/rejects prior samples
    based on temperature-controlled criteria. Used internally by integrate_rejection for
    both serial and parallel processing.
    
    Parameters
    ----------
    D : list
        List of forward modeled data arrays for each data type.
    DATA : dict
        Dictionary containing observed data including 'd_obs', 'd_std', and other data arrays.
    idx : list, optional
        Indices of prior samples to use. If empty, uses sequential indexing.
        Default is empty list.
    N_use : int, optional
        Maximum number of prior samples to evaluate.
        Default is 1000.
    id_use : list, optional
        List of data identifiers to use for likelihood calculation.
        Default is [1, 2].
    ip_range : list, optional
        Range of data point indices to process. If empty, processes all data points.
        Default is empty list.
    nr : int, optional
        Number of posterior samples to retain per data point.
        Default is 400.
    autoT : int, optional
        Automatic temperature estimation method (1=enabled, 0=disabled).
        Default is 1.
    T_base : float, optional
        Base temperature for rejection sampling when autoT=0.
        Default is 1.
    **kwargs : dict
        Additional arguments including useRandomData, showInfo, use_N_best.
    
    Returns
    -------
    tuple
        Tuple containing (i_use_all, T_all, EV_all, EV_post_all, N_UNIQUE_all, ip_range)
        where:
        - i_use_all : ndarray, shape (nump, nr)
            Indices of accepted posterior samples for each data point
        - T_all : ndarray, shape (nump,)
            Temperature values used for each data point
        - EV_all : ndarray, shape (nump,)
            Evidence values for each data point
        - EV_post_all : ndarray, shape (nump,)
            Posterior evidence values for each data point
        - N_UNIQUE_all : ndarray, shape (nump,)
            Number of unique samples for each data point
        - ip_range : ndarray
            Range of data point indices that were processed
    
    Notes
    -----
    This function is the computational core of the rejection sampling algorithm.
    It handles temperature annealing and likelihood evaluation for efficient sampling.
    
    The algorithm evaluates the likelihood of observed data given forward modeled data
    for each prior sample, then uses temperature-controlled acceptance criteria to
    select posterior samples that are consistent with observations.
    """
    
    import integrate as ig

    # get optional arguments
    use_N_best = kwargs.get('use_N_best', 0)
    #print("use_N_best=%d" % use_N_best)
    showInfo = kwargs.get('showInfo', 0)
    if (showInfo<0):
        disableTqdm=True
    else:
        disableTqdm=False
    
    useRandomData = kwargs.get('useRandomData', True)
    #useRandomData = kwargs.get('useRandomData', False)
    

    # Get number of data points
    Ndp = DATA['d_obs'][0].shape[0]
    # if ip_range is empty then use all data points
    if len(ip_range)==0:
        ip_range = np.arange(Ndp)

    nump=len(ip_range)
    if showInfo>1:
        print('Number of data points to invert: %d' % nump)
    i_use_all = np.zeros((nump, nr), dtype=np.int32)
    T_all = np.zeros(nump)*np.nan
    EV_all = np.zeros(nump)*np.nan
    EV_post_all = np.zeros(nump)*np.nan
    N_UNIQUE_all = np.zeros(nump)*np.nan
    
    
    # Get the lookup sample size
    N = D[0].shape[0]

    Ndt = len(id_use) # Number of data types used   
    if showInfo>2:
        print('Numbe of data type used, Ndt=%d' % Ndt)

    if N_use>N:
        N_use = N

    if len(idx)==0:
        idx = np.arange(N_use)
    
    #i=0
    
    noise_model = DATA['noise_model']
    id_prior_use = DATA['id_use']
    i_use_data = DATA['i_use']
    #print(noise_model)
    #print(id_prior_use)
    #print(i_use_data)

    # Convert class id to index
    # The class_id_list, could /should be loaded prior_h5:/M1/class_id !!

    # Select whether to convert CLASS to IDX before doing inversion?
    # class_is_idx = True is MUCH faster!!
    class_is_idx = True
    #class_is_idx = False
    

    class_id_list = []
    updated_data_ids = []
    for i in range(Ndt):
        i_prior = id_prior_use[i]-1 
        if (noise_model[i]=='multinomial'):
            Di, class_id, class_id_out = ig.class_id_to_idx(D[i_prior])
            #print(class_id_out)
            if (class_is_idx)&(id not in updated_data_ids):
                updated_data_ids.append(id)
                D[i_prior]=Di
                if showInfo>1:
                    print('Updated prior id %d' % i_prior)
            
            if (class_is_idx):                
                class_id_list.append(class_id_out)
            else:
                class_id_list.append(class_id)

        else:    
            class_id_list.append([])

    if showInfo>1:
        print('class_id_list',class_id_list)
        print('len(class_id_list)',len(class_id_list))

    
    # THIS IS THE ACTUAL INVERSION!!!!
    # Start looping over the data points
    for j in tqdm(range(len(ip_range)), miniters=10, disable=disableTqdm, desc='rejection', leave=False):
        ip = ip_range[j] # This is the index of the data point to invert
        t=[]
        N = D[0].shape[0]
        NDsets = len(id_use)
        L = np.zeros((NDsets, N))

        
        # Loop over the number of data types Ndt
        for i in range(Ndt):
            use_data_point = i_use_data[i][ip]
            #use_data_point = 1 # FORCE USE OF DATA POINT
            #use_data_point = 0 # FORCE NOT TO USE DATA POINT
            if showInfo>3:    
                print("-i=%d, Using data type %d" % (i,Ndt))
                print("len(D)",len(D))

            if (use_data_point==1):
                #if i_use_data[i]==1:
                #    print('Using data %d' % i)
                #
                if showInfo>3:    
                    print('j=%4d Using data %d --> %d' % (j,i,use_data_point))

                # ONLY PERFORM CALUCATION IF I_USE_DATA = 1.. UPDATE LOAD_DATA, TO ALWAY PROVIDE I_USE 
                # Select the proper data types. It is give, the integer in 'D1/', 'D2/' etc, so we need to subtract 1
                # as D1 is the first data types D[0]
                i_prior = id_prior_use[i]-1 
                
                if showInfo>3:    
                    print("--Using id_prior_use",str(i_prior))
                
                t0=time.time()
                #id = id_use[i_prior]
                if noise_model[i]=='gaussian':
                    d_obs = DATA['d_obs'][i][ip]
                    
                    if DATA['Cd'][0] is not None:                    
                        # if Cd is 3 dimensional, take the first slice
                        if len(DATA['Cd'][0].shape) == 3:
                            Cd = DATA['Cd'][0][ip]
                        else:
                            Cd = DATA['Cd'][0][:]

                        L_single = likelihood_gaussian_full(D[i_prior], d_obs, Cd, N_app = use_N_best)
                        
                    elif DATA['d_std'][0] is not None:
                        d_std = DATA['d_std'][i][ip]
                        L_single = likelihood_gaussian_diagonal(D[i_prior], d_obs, d_std, use_N_best)

                    else:
                        print('No d_std or Cd in %s' % DS)

                    L[i] = L_single
                    t.append(time.time()-t0)
                elif noise_model[i]=='multinomial':
                    d_obs = DATA['d_obs'][i][ip]

                    if showInfo>3:
                        print(D[i])
                    
                    class_id = class_id_list[i]
                    #print(class_id)
                    useMultiNomal = True
                    if useMultiNomal:
                        
                        L_single = likelihood_multinomial(D[i_prior],d_obs, np.array(class_id), class_is_idx=class_is_idx)
                        #print(L_single[0])
                    L[i] = L_single           
                    t.append(time.time()-t0)

                else: 
                    # noise model not regcognized
                    # L_single = -1
                    pass
            else:
                L[i] = np.zeros(N)
                    
        t0=time.time()

        # NOw we have all the likelihoods for all data types. Copmbine them into ooe
        L_single = L
        L = np.sum(L_single, axis=0)
        

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
            p=P_acc/np.sum(P_acc)
            i_use = np.random.choice(N, nr, p=p)
        except:
            print('Error in np.random.choice for ip=%d' % ip)   
            i_use = np.random.choice(N, nr)
        
        if useRandomData:
            # get the correct index of the subset used
            i_use = idx[i_use]

        t.append(time.time()-t0)        

    

        # Compute the evidence
        maxlogL = np.nanmax(L)
        exp_logL = np.exp(L - maxlogL)
        EV = maxlogL + np.log(np.nansum(exp_logL)/len(L))

        # Compute 'posterior evidence' - mean posterior likelihood
        EV_post = np.nanmean(exp_logL)
        #EV_post2 = np.nanmean(L[i_use])
        #print('EV=%f, EV_post=%f, EV_post2=%f' % (EV, EV_post, EV_post2))
        
        t.append(time.time()-t0)

        pltDegug = 0
        if pltDegug>0:
            import matplotlib.pyplot as plt
            plt.semilogy(d_obs, 'k', linewidth=4)
            plt.semilogy(D[0][i_use].T, 'r', linewidth=1)
            plt.show()
            print(D[0][10])

        i_use_all[j] = i_use
        T_all[j] = T
        EV_all[j] = EV
        EV_post_all[j] = EV_post
        # find the number of unique indexes
        N_UNIQUE_all[j] = len(np.unique(i_use))

        if showInfo>2:
            for i in range(len(t)):
                if i<Ndt:
                    print(' Time id%d: %f - %s' % (i,t[i],noise_model[i]))
                else:
                    print(' Time id%d, sampling: %f' % (i,t[i]))
            print('Time total: %f' % np.sum(t))
        
    return i_use_all, T_all, EV_all, EV_post_all, N_UNIQUE_all, ip_range



def integrate_posterior_main(ip_chunks, D, DATA, idx, N_use, id_use, autoT, T_base, nr, Ncpu, use_N_best):
    """
    Coordinate parallel processing of posterior sampling across multiple chunks.
    
    This function manages the parallel execution of rejection sampling by distributing
    data point chunks across multiple CPU cores. It handles shared memory management
    for efficient data transfer between processes and aggregates results from all chunks.
    
    Parameters
    ----------
    ip_chunks : list
        List of data point index chunks for parallel processing.
    D : list
        List of forward modeled data arrays shared across processes.
    DATA : dict
        Dictionary containing observed data structures.
    idx : list
        Indices of prior samples to use for inversion.
    N_use : int
        Maximum number of prior samples per chunk.
    id_use : list
        List of data identifiers for likelihood calculation.
    autoT : int
        Automatic temperature estimation flag.
    T_base : float
        Base temperature for rejection sampling.
    nr : int
        Number of posterior samples to retain per data point.
    Ncpu : int
        Number of CPU cores to use for parallel processing.
    use_N_best : int
        Flag to use only the N best-fitting samples.
    
    Returns
    -------
    tuple
        Tuple containing aggregated results from all chunks:
        - i_use_all : ndarray, shape (Ndp, nr)
            Indices of accepted posterior samples for all data points
        - T_all : ndarray, shape (Ndp,)
            Temperature values used for all data points
        - EV_all : ndarray, shape (Ndp,)
            Evidence values for all data points
        - EV_post_all : ndarray, shape (Ndp,)
            Posterior evidence values for all data points
        - N_UNIQUE_all : ndarray, shape (Ndp,)
            Number of unique samples for all data points
    
    Notes
    -----
    This function uses shared memory to minimize data copying overhead during parallel processing.
    Shared memory is automatically cleaned up after processing completion.
    
    The function creates a process pool with the specified number of CPUs and distributes
    the work chunks across them. Each worker process operates on a subset of data points
    and returns its results, which are then aggregated by the main process.
    """
    #import integrate as ig
    from multiprocessing import Pool

    #shared_memory_refs = create_shared_memory(D)
    shared_memory_refs, shm_objects = create_shared_memory(D)
    #reconstructed_arrays = reconstruct_shared_arrays(shared_memory_refs)

    #with Pool(Ncpu) as p:
    try:
        with Pool(Ncpu) as p:
            # New implementation with shared memory
            results = p.map(integrate_posterior_chunk, [(i, ip_chunks, DATA, idx,  N_use, id_use, shared_memory_refs, autoT, T_base, nr, use_N_best) for i in range(len(ip_chunks))])
            # Old implementation where D was copied to each process
            #results = p.map(integrate_posterior_chunk, [(i, ip_chunks, D, DATA, idx,  N_use, id_use, shared_memory_refs, autoT, T_base, nr, use_N_best) for i in range(len(ip_chunks))])
    finally:
        # Always clean up shared memory
        if shm_objects:
            cleanup_shared_memory(shm_objects)

    # Cleanup shared memory
    #cleanup_shared_memory(shared_memory_refs)
    cleanup_shared_memory(shm_objects)
    

    # Get sample size N from f_prior_h5
    N=D[0].shape[0]
    
    # Get number of data points from, f_data_h5
    Ndp = DATA['d_obs'][0].shape[0]

    i_use_all = np.random.randint(0, N, (Ndp, nr))
    T_all = np.zeros(Ndp)*np.nan
    EV_all = np.zeros(Ndp)*np.nan
    EV_post_all = np.zeros(Ndp)*np.nan
    N_UNIQUE_all = np.zeros(Ndp)*np.nan
    
    for i, (i_use, T, EV, EV_post, N_UNIQUE, ip_range) in enumerate(results):
        for i in range(len(ip_range)):
                ip = ip_range[i]
                #print('ip=%d, i=%d' % (ip,i))
                i_use_all[ip] = i_use[i]
                T_all[ip] = T[i]
                EV_all[ip] = EV[i]
                EV_post_all[ip] = EV_post[i]
                N_UNIQUE_all[ip] = N_UNIQUE[i]

    return i_use_all, T_all, EV_all, EV_post_all, N_UNIQUE_all



def integrate_posterior_chunk(args):
    """
    Process a single chunk of data points for parallel rejection sampling.
    
    This function is called by each worker process in the parallel processing pool.
    It reconstructs shared data arrays, processes the assigned chunk of data points
    using rejection sampling, and returns results for aggregation by the main process.
    
    Parameters
    ----------
    args : tuple
        Tuple containing chunk parameters:
        - i_chunk : int
            Index of the current chunk being processed
        - ip_chunks : list
            List of all data point index chunks
        - DATA : dict
            Dictionary containing observed data structures
        - idx : list
            Indices of prior samples to use for inversion
        - N_use : int
            Maximum number of prior samples to evaluate
        - id_use : list
            List of data identifiers for likelihood calculation
        - shared_memory_refs : list
            References to shared memory segments containing forward modeled data
        - autoT : int
            Automatic temperature estimation flag
        - T_base : float
            Base temperature for rejection sampling
        - nr : int
            Number of posterior samples to retain per data point
        - use_N_best : int
            Flag to use only the N best-fitting samples
    
    Returns
    -------
    tuple
        Tuple containing chunk results:
        - i_use : ndarray, shape (nump, nr)
            Indices of accepted posterior samples for the chunk
        - T : ndarray, shape (nump,)
            Temperature values used for the chunk
        - EV : ndarray, shape (nump,)
            Evidence values for the chunk
        - EV_post : ndarray, shape (nump,)
            Posterior evidence values for the chunk
        - N_UNIQUE : ndarray, shape (nump,)
            Number of unique samples for the chunk
        - ip_range : ndarray
            Range of data point indices that were processed
    
    Notes
    -----
    This function runs in a separate process and communicates with the main process
    through shared memory for data arrays and return values through the process pool.
    
    The function first reconstructs the shared data arrays from memory references,
    then calls integrate_rejection_range to perform the actual rejection sampling
    on the assigned chunk of data points.
    """
    #import integrate as ig
    
    # New implementation with shared memory
    i_chunk, ip_chunks, DATA, idx, N_use, id_use, shared_memory_refs, autoT, T_base, nr, use_N_best = args
    # Old implementation where D was copied to each process
    #i_chunk, ip_chunks, D, DATA, idx, N_use, id_use, shared_memory_refs, autoT, T_base, nr, use_N_best = args
    #D=reconstruct_shared_arrays(shared_memory_refs)
    
    # Reconstruct shared arrays without copying - returns tuple (arrays, shm_objects)
    D, worker_shm_objects = reconstruct_shared_arrays(shared_memory_refs)

    try:
        # Perhaps truncat according to N_use
        #for i in len(D)
        #    D[i] = D[i][:N_use]

        ip_range = ip_chunks[i_chunk]

        #print(f'Chunk {i_chunk+1}/{len(ip_chunks)}, ndp={len(ip_range)}')

        i_use, T, EV, EV_post, N_UNIQUE, ip_range = integrate_rejection_range(
            D,
            DATA,
            idx,
            N_use=N_use,
            id_use=id_use,
            ip_range=ip_range,
            autoT=autoT,
            T_base=T_base,
            nr=nr,     
            use_N_best=use_N_best, 
        )

        return i_use, T, EV, EV_post, N_UNIQUE, ip_range
    
    finally:
        # Clean up worker's shared memory references
        for shm in worker_shm_objects:
            try:
                shm.close()
            except Exception as e:
                logger.debug(f"Error closing shared memory in worker: {e}")


def select_subset_for_inversion(dd, N_app):
    """
    Select a subset of indices for inversion based on the sum of absolute values.

    This function calculates the sum of absolute values along the specified axis
    for each row in the input array `dd`. It then selects the indices of the 
    `N_app` smallest sums.

    Parameters
    ----------
    dd : numpy.ndarray
        A 2D array of data from which to select the subset.
    N_app : int
        The number of indices to select based on the smallest sums.

    Returns
    -------
    idx : numpy.ndarray
        An array of indices corresponding to the `N_app` smallest sums.
    nsum : numpy.ndarray
        An array containing the sum of absolute values for each row in `dd`.

    Notes
    -----
    This function uses `np.nansum` to ignore NaNs in the summation and 
    `np.argpartition` for efficient selection of the smallest sums.
    """
    nsum = np.nansum(np.abs(dd), axis=1)
    idx = np.argpartition(nsum, N_app)[:N_app]
    return idx, nsum


def likelihood_gaussian_diagonal(D, d_obs, d_std, N_app=0):
    """
    Compute the Gaussian likelihood for a diagonal covariance matrix.
    
    This function calculates the likelihood of observed data given a set of predicted data
    and standard deviations, assuming a Gaussian distribution with a diagonal covariance matrix.
    
    Parameters
    ----------
    D : ndarray, shape (n_samples, n_features)
        Predicted data array containing forward model predictions.
    d_obs : ndarray, shape (n_features,)
        Observed data array containing measured values.
    d_std : ndarray, shape (n_features,)
        Standard deviation array containing measurement uncertainties.
    N_app : int, optional
        Number of data points to use for approximation. If 0, uses all data.
        Default is 0.
    
    Returns
    -------
    ndarray, shape (n_samples,)
        Log-likelihood values for each sample, computed as:
        L[i] = -0.5 * sum((D[i] - d_obs)**2 / d_std**2)
    
    Notes
    -----
    The function assumes independent Gaussian errors with diagonal covariance matrix.
    The log-likelihood is computed using vectorized operations for efficiency.
    
    When N_app > 0, only the N_app samples with smallest residuals are evaluated,
    and the remaining samples are assigned a very low likelihood (-1e15).
    """
    
    # Compute the likelihood
    # Sequential
    #L = np.zeros(D.shape[0])
    #for i in range(D.shape[0]):
    #    L[i] = -0.5 * np.nansum(dd[i]**2 / d_std**2)
    # Vectorized
    dd = D - d_obs
    
    if N_app > 0:
       L = np.ones(D.shape[0])*-1e+15
       idx = select_subset_for_inversion(dd, N_app)[0] 
       #L_small = -0.5 * np.nansum(dd[idx]**2 / d_std**2, axis=1)
       L_small = likelihood_gaussian_diagonal(D[idx], d_obs, d_std,0)
       L[idx]=L_small
       
    else:
        L = -0.5 * np.nansum(dd**2 / d_std**2, axis=1)

    return L

def likelihood_gaussian_full(D, d_obs, Cd, N_app=0, checkNaN=True, useVectorized=False):
    """
    Calculate the Gaussian likelihood with full covariance matrix.
    
    This function computes likelihood values for model predictions given observed data
    and a full covariance matrix, handling NaN values appropriately.
    
    Parameters
    ----------
    D : ndarray, shape (n_samples, n_features)
        Model predictions containing forward model results.
    d_obs : ndarray, shape (n_features,)
        Observed data containing measured values.
    Cd : ndarray, shape (n_features, n_features)
        Full covariance matrix of observed data uncertainties.
    N_app : int, optional
        Number of data points to use for approximation. If 0, uses all data.
        Default is 0.
    checkNaN : bool, optional
        If True, handles NaN values in d_obs by ignoring them in calculations.
        Default is True.
    useVectorized : bool, optional
        If True, uses vectorized computation for better performance.
        Default is False.
    
    Returns
    -------
    ndarray, shape (n_samples,)
        Log-likelihood values for each sample, computed as:
        L[i] = -0.5 * (D[i] - d_obs)^T * Cd^(-1) * (D[i] - d_obs)
    
    Notes
    -----
    The function handles full covariance matrices accounting for correlated errors.
    When checkNaN=True, only non-NaN data points are used in the likelihood calculation.
    
    The vectorized implementation uses einsum for efficient matrix operations.
    When N_app > 0, only the N_app samples with smallest residuals are evaluated.
    
    TODO: Check that this works when D has NaN values and determine why they occur.
    """
    
    if checkNaN:
        # find index of non-nan values in d_obs or non-nan values in np.sum(Cd, axis=0)
        #ind = np.where(~np.isnan(d_obs))[0]
        ind = np.where(~np.isnan(d_obs) & ~np.isnan(np.sum(Cd, axis=0)))[0]
        # Exclude also all data for which one Nan Is available.. This is probably not ideal
        ind = np.where(~np.isnan(d_obs) & ~np.isnan(np.sum(Cd, axis=0)) & ~np.isnan(np.sum(D, axis=0)) )[0]
        dd = D[:,ind] - d_obs[ind]
        iCd = np.linalg.inv(Cd[np.ix_(ind, ind)])
    else:    
        dd = D - d_obs
        iCd = np.linalg.inv(Cd)
            
    if N_app > 0:
        L = np.ones(D.shape[0])*-1e+15
        idx = select_subset_for_inversion(dd, N_app)[0] 
        if useVectorized:
            L_small = -.5 * np.einsum('ij,ij->i', dd[idx] @ iCd, dd[idx])
        else:
            L_small = np.zeros(idx.shape[0])
            for i in range(idx.shape[0]):
                L_small[i] = -.5 * np.nansum(dd[idx[i]].T @ iCd @ dd[idx[i]])
        L[idx] = L_small
    
        return L
    
    if useVectorized:
        # vectorized    
        L = -.5 * np.einsum('ij,ij->i', dd @ iCd, dd)        
    else:   
        # non-vectorized
        L = np.zeros(D.shape[0])
        for i in range(D.shape[0]):
            L[i] = -.5 * np.nansum(dd[i].T @ iCd @ dd[i])
        
    return L




def likelihood_multinomial(D, P_obs, class_id=None, class_is_idx=False, entropyFilter=False, entropyThreshold=0.99):
    """
    Calculate log-likelihood of multinomial distribution for discrete data.
    
    This function computes the log-likelihood of multinomial distribution for discrete data
    using direct indexing and optimized array operations for efficient computation.
    
    Parameters
    ----------
    D : ndarray, shape (N, n_features)
        Matrix of observed discrete data, where each element represents a class ID.
    P_obs : ndarray, shape (n_classes, n_features)
        Matrix of probabilities, where each column represents probability distribution over classes.
    class_id : ndarray, optional
        Array of unique class IDs corresponding to rows in P_obs. 
        If None, extracted from unique values in D.
        Default is None.
    class_is_idx : bool, optional
        If True, class_id is already an index. If False, computes index from class_id array.
        Default is False.
    entropyFilter : bool, optional
        If True, applies entropy filtering to select features.
        Default is False.
    entropyThreshold : float, optional
        Threshold for entropy filtering. Features with entropy below this value are selected.
        Default is 0.99.
    
    Returns
    -------
    ndarray, shape (N,)
        Log-likelihood values for each sample, computed using log base 10 with
        normalization by maximum probability for numerical stability.
    
    Notes
    -----
    The function creates a mapping from class IDs to indices, converts test data to
    corresponding indices, and retrieves probabilities using advanced indexing.
    
    The log-likelihood is calculated using max normalization for numerical stability:
    logL[i] = sum(log10(p[i] / p_max))
    
    When entropyFilter is True, only features with entropy below the threshold
    are used in the likelihood calculation, which can improve computational efficiency
    for datasets with many uninformative features.
    
    Examples
    --------
    >>> D = np.array([[1, 2], [2, 1]])  # Sample data with class IDs
    >>> P_obs = np.array([[0.3, 0.7], [0.7, 0.3]])  # Class probabilities
    >>> logL = likelihood_multinomial(D, P_obs)
    """
    
    from scipy.stats import entropy
    
    if class_id is None:
        class_id =  np.arange(len(np.unique(D))).astype(int)
        class_id =  np.unique(D).astype(int)
    
    D=np.atleast_2d(D)

    if entropyFilter:
        H=entropy(P_obs.T)
        used = np.where(H<entropyThreshold)[0] 
        if len(used)==0:
            used = np.arange(1)
        D = D[:,used]
        P_obs = P_obs[:,used]

    N, nm = D.shape
    logL = np.zeros((N))
    class_id = class_id.astype(int)
    p_max = np.max(P_obs, axis=0)

    # Create mapping from class_id to index
    class_to_idx = {cid: idx for idx, cid in enumerate(class_id)}
    #print('class_id',class_id)  
    #print('class_to_idx',class_to_idx)

    for i in range(N):
        # Convert test data to indices using the mapping
        if class_is_idx:
            i_test = D[i]
        else:
            i_test = np.array([class_to_idx[cls] for cls in D[i]])

        # Get probabilities directly using advanced indexing
        p = P_obs[i_test, np.arange(nm)]
        # Calculate log likelihood
        logL[i] = np.sum(np.log10(p/p_max))

       
    return logL


def create_shared_memory(arrays):
    """
    Create shared memory segments for arrays.
    
    This function creates shared memory segments for a list of numpy arrays,
    allowing them to be accessed efficiently across multiple processes without
    copying data. Returns both memory references and objects for cleanup.
    
    Parameters
    ----------
    arrays : list
        List of numpy arrays to place in shared memory.
    
    Returns
    -------
    tuple
        Tuple containing (shared_memories, shm_objects) where:
        - shared_memories : list
            List of (name, shape, dtype) tuples identifying shared memory segments
        - shm_objects : list
            List of SharedMemory objects for cleanup
    
    Notes
    -----
    The returned shm_objects must be cleaned up using cleanup_shared_memory()
    to prevent memory leaks. This should be done in a finally block.
    
    If an error occurs during creation, any successfully created memory segments
    are automatically cleaned up before raising the exception.
    """
    shared_memories = []
    shm_objects = []
    
    try:
        for array in arrays:
            shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
            shared_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
            shared_array[:] = array[:]
            shared_memories.append((shm.name, array.shape, array.dtype))
            shm_objects.append(shm)
            logger.debug(f"Created shared memory: {shm.name}")
        return shared_memories, shm_objects
    except Exception as e:
        logger.error(f"Error creating shared memory: {e}")
        # Clean up any created memory segments before raising
        for shm in shm_objects:
            shm.close()
            shm.unlink()
        raise

def reconstruct_shared_arrays(shared_memory_refs):
    """
    Reconstruct arrays from shared memory references.
    
    This function takes shared memory references (created by create_shared_memory)
    and reconstructs the original numpy arrays by accessing the shared memory
    segments. Used by worker processes to access shared data.
    
    Parameters
    ----------
    shared_memory_refs : list
        List of (name, shape, dtype) tuples identifying shared memory segments.
    
    Returns
    -------
    tuple
        Tuple containing:
        - reconstructed_arrays : list
            List of numpy arrays reconstructed from shared memory
        - shm_objects : list
            List of shared memory objects that must be closed after use
    
    Warnings
    --------
    The reconstructed arrays are views into shared memory. Modifications
    will affect the shared data across all processes. Do NOT modify these arrays.
    The shared memory objects must be closed after use to prevent leaks.
    
    Notes
    -----
    If an error occurs during reconstruction, any successfully opened shared memory
    objects are automatically closed before raising the exception.
    """
    reconstructed_arrays = []
    shm_objects = []
    for shm_name, shape, dtype in shared_memory_refs:
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
            array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            reconstructed_arrays.append(array)
            shm_objects.append(shm)
        except Exception as e:
            logger.error(f"Error reconstructing array: {e}")
            # Clean up any successfully opened shared memory objects
            for opened_shm in shm_objects:
                opened_shm.close()
            raise
    return reconstructed_arrays, shm_objects

def cleanup_shared_memory(shm_objects):
    """
    Clean up shared memory segments.
    
    This function properly closes and unlinks shared memory objects created during
    parallel processing to prevent memory leaks and system resource exhaustion.
    It handles cleanup gracefully by catching and ignoring errors for objects
    that may have already been cleaned up.
    
    Parameters
    ----------
    shm_objects : list
        List of shared memory objects to clean up.
    
    Returns
    -------
    None
    
    Notes
    -----
    This function should always be called in a finally block or similar
    error-safe context to ensure cleanup occurs even if exceptions are raised.
    
    Each shared memory object is both closed (to release the local reference)
    and unlinked (to remove it from the system). Errors during cleanup are
    silently ignored to prevent cascading failures.
    
    Examples
    --------
    >>> shared_memories, shm_objects = create_shared_memory(arrays)
    >>> try:
    ...     # Use shared memory
    ...     pass
    ... finally:
    ...     cleanup_shared_memory(shm_objects)
    """
    if not shm_objects:
        return
    
    for shm in shm_objects:
        try:
            shm.close()
            shm.unlink()
            logger.debug(f"Cleaned up shared memory: {shm.name}")

        except Exception as e:
            #logger.error(f"Error cleaning up shared memory: {e}")
            pass