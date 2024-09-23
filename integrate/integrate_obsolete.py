
def integrate_rejection(f_prior_h5='DJURSLAND_P01_N0010000_NB-13_NR03_PRIOR.h5',
                            f_data_h5='tTEM-Djursland.h5',
                            f_post_h5='',
                            autoT=1,
                            N_use=1000000,
                            ns=400,
                            parallel=1, 
                            updatePostStat= True,
                            **kwargs):
    r"""
    Perform rejection-based integration of data and prior simulations.

    :param f_prior_h5: Path to the prior simulations HDF5 file.
    :type f_prior_h5: str
    :param f_data_h5: Path to the data HDF5 file.
    :type f_data_h5: str
    :param f_post_h5: Path to the output posterior HDF5 file. If not provided, a default filename will be generated.
    :type f_post_h5: str
    :param autoT: Auto-tuning parameter.
    :type autoT: int
    :param N_use: Number of prior simulations to use.
    :type N_use: int
    :param ns: Number of samples.
    :type ns: int
    :param parallel: Parallelization mode. 1 for parallel processing, 2 for executing the script from the command line, and any other value for sequential processing.
    :type parallel: int
    :param updatePostStat: Flag indicating whether to update posterior statistics.
    :type updatePostStat: bool
    :param \**kwargs: Additional keyword arguments.
    :returns: Path to the output posterior HDF5 file.
    :rtype: str
    :raises FileNotFoundError: If the prior simulations or data file does not exist.
    """
    
    import h5py
    import numpy as np
    from datetime import datetime   
    import argparse
    from tqdm import tqdm
    from functools import partial
    import multiprocessing
    from multiprocessing import Pool
    #from multiprocessing.dummy import Pool
    import os

    id=1

    chunksize = kwargs.get('chunksize', 1)
    Nproc = kwargs.get('Nproc', 0)
    showInfo = kwargs.get('showInfo', 0)
    if showInfo>0:
        print('Running: integrate_rejection.py %s %s --autoT %d --N_use %d --ns %d -parallel %d --updatePostStat %d' % (f_prior_h5,f_data_h5,autoT,N_use,ns,parallel,updatePostStat))

    #% Check that hdf5 files exists
    import os.path
    if not os.path.isfile(f_prior_h5):
        print('File %s does not exist' % f_prior_h5)
        exit()  
    if not os.path.isfile(f_data_h5):
        print('File %s does not exist' % f_data_h5)
        exit()
 
    with h5py.File(f_data_h5, 'r') as f:
        d_obs = f['/D1/d_obs']
        nd = d_obs.shape[1]
        nsoundings = d_obs.shape[0]

    data_str = '/D%d' % id
    with h5py.File(f_prior_h5, 'r') as f:
        d_sim = f[data_str]
        N = d_sim.shape[0]
    N_use = min([N_use, N])

    with h5py.File(f_prior_h5, 'r') as f:
        d_sim = f[data_str][:N_use,:]
        #d_sim = f[data_str][:,:N_use]


    # Create shared memory block
    shm = shared_memory.SharedMemory(create=True, size=d_sim.nbytes)
    d_sim_shared = np.ndarray(d_sim.shape, dtype=d_sim.dtype, buffer=shm.buf)
    np.copyto(d_sim_shared, d_sim)

    print(shm.name)

    #shm.close()
    #shm.unlink()
    
    if len(f_post_h5)==0:
        f_post_h5 = "POST_%s_%s_Nu%d_aT%d.h5" % (os.path.splitext(f_data_h5)[0],os.path.splitext(f_prior_h5)[0],N_use,autoT)
        #f_post_h5 = f"{f_prior_h5[:-3]}_POST_Nu{N_use}_aT{autoT}.h5"

    # Check that f_post_h5 allready exists, and warn the user   
    if os.path.isfile(f_post_h5):
        print('File %s allready exists' % f_post_h5)
        print('Overwriting...')    
        
    

    if showInfo>0:
        print('nsoundings:%d, N_use:%d, nd:%d' % (nsoundings,N_use,nd))
        print('Writing results to ',f_post_h5)
    
    # remaining code...

    date_start = str(datetime.now())
    t_start = datetime.now()
    i_use_all = np.zeros((ns,nsoundings), dtype=int)
    POST_T = np.ones(nsoundings) * np.nan
    POST_EV = np.ones(nsoundings) * np.nan

    if parallel==1:
        ## % PARALLEL IN SCRIPT
        # Parallel
        if Nproc < 1 :
            Nproc =  int(multiprocessing.cpu_count()/2)
            #Nproc =  int(multiprocessing.cpu_count())
        if (showInfo>-1):
            print("Using %d parallel threads." % (Nproc))
            # print("nsoundings: %d" % nsoundings)
        
        # Create a list of tuples where each tuple contains the arguments for a single call to sample_from_posterior_shared
        args_list = [(is_, shm.name, d_sim.shape, d_sim.dtype, f_data_h5, N_use, autoT, ns) for is_ in range(nsoundings)]
        # Create a multiprocessing pool and compute D for each chunk of C
        with Pool(Nproc) as p:
            out = list(tqdm(p.imap(sample_from_posterior_shared, args_list, chunksize=chunksize), total=nsoundings, mininterval=1))

        for output in out:
            i_use = output[0]
            T = output[1]
            EV = output[2]
            is_ = output[3]
            POST_T[is_] = T
            POST_EV[is_] = EV
            i_use_all[:,is_] = i_use

    elif parallel==2:
        print('CALL SCRIPT FROM COMMANDLINE!!!!')
        cmd = 'python integrate_rejection.py %s %s --N_use=%d --autoT=%d --ns=%d --updatePostStat=%d' % (f_prior_h5,f_data_h5,N_use,autoT,ns,updatePostStat) 
        print('Executing "%s"'%(cmd))
        import os
        os.system(cmd)

    else:
        # SEQUENTIAL        
        for is_ in tqdm(range(nsoundings)):
            i_use, T, EV, is_out = sample_from_posterior(is_,d_sim,f_data_h5, N_use,autoT,ns)
            #i_use, T, EV, is_out = sample_from_posterior_shared(0, shm.name, d_sim.shape, d_sim.dtype,f_data_h5, N_use,autoT,ns)            
            POST_T[is_] = T
            POST_EV[is_] = EV
            i_use_all[:,is_] = i_use
    
        date_end = str(datetime.now())
        t_end = datetime.now()
        t_elapsed = (t_end - t_start).total_seconds()
        t_per_sounding = t_elapsed / nsoundings

        print('T_av=%3.1f ' % (np.nanmean(POST_T)))
        
    # Close and release shared memory block
    shm.close()
    shm.unlink()
        
    date_end = str(datetime.now())
    t_end = datetime.now()
    t_elapsed = (t_end - t_start).total_seconds()
    t_per_sounding = t_elapsed / nsoundings
    if (showInfo>-1):
        print('T_av=%3.1f, Time=%5.1fs/%d soundings ,%4.3fms/sounding' % (np.nanmean(POST_T),t_elapsed,nsoundings,t_per_sounding*1000))

    if showInfo>0:
        print('Writing to file: ',f_post_h5)
    with h5py.File(f_post_h5, 'w') as f:
        f.create_dataset('i_use', data=i_use_all.T)
        f.create_dataset('T', data=POST_T.T)
        f.create_dataset('EV', data=POST_EV.T)
        f.attrs['date_start'] = date_start
        f.attrs['date_end'] = date_end
        f.attrs['inv_time'] = t_elapsed
        f.attrs['f5_prior'] = f_prior_h5
        f.attrs['f5_data'] = f_data_h5
        f.attrs['N_use'] = N_use

    if updatePostStat:
        integrate_posterior_stats(f_post_h5, **kwargs)
    
    return f_post_h5



#def sample_from_posterior_shared(is_, shm_name, shape, dtype,f_data_h5='tTEM-Djursland.h5', N_use=1000000, autoT=1, ns=400):
def sample_from_posterior_shared(args):
    # Unpack tuple
    is_, shm_name, shape, dtype, f_data_h5, N_use, autoT, ns = args
    # Recreate the numpy array from shared memory
    shm = shared_memory.SharedMemory(name=shm_name)
    d_sim = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    
    with h5py.File(f_data_h5, 'r') as f:
        d_obs = f['/D1/d_obs'][is_,:]
        d_std = f['/D1/d_std'][is_,:]
    
    i_use = np.where(~np.isnan(d_obs) & (np.abs(d_obs) > 0))[0]
    d_obs = d_obs[i_use]
    d_var = d_std[i_use]**2

    dd = (d_sim[:, i_use] - d_obs)**2
    logL = -.5*np.sum(dd/d_var, axis=1)

    if autoT == 1:
        T = logl_T_est(logL)
    else:
        T = 1
    maxlogL = np.nanmax(logL)
    
    exp_logL = np.exp(logL - maxlogL)
    i_use, P_acc = lu_post_sample_logl(logL, ns, T)
    EV = maxlogL + np.log(np.nansum(exp_logL)/len(logL))
    return i_use, T, EV, is_


def sample_from_posterior_old(is_, d_sim, f_data_h5='tTEM-Djursland.h5', N_use=1000000, autoT=1, ns=400):
            
    # This is extremely memory efficicent, but perhaps not CPU efficiwent, when lookup table is small?
    d_obs = h5py.File(f_data_h5, 'r')['/D1/d_obs'][is_,:]
    d_std = h5py.File(f_data_h5, 'r')['/D1/d_std'][is_,:]
    i_use = np.where(~np.isnan(d_obs) & (np.abs(d_obs) > 0))[0]
    
    d_obs = d_obs[i_use]
    d_var = d_std[i_use]**2

    logL = np.zeros(N_use)
    for i in range(N_use):
        dd = (d_sim[i,i_use] - d_obs)**2
        logL[i] = -.5*np.sum(dd/d_var)

    if autoT == 1:
        T = logl_T_est(logL)
    else:
        T = 1
    maxlogL = np.nanmax(logL)
    
    i_use, P_acc = lu_post_sample_logl(logL, ns, T)
    EV=maxlogL + np.log(np.nansum(np.exp(logL-maxlogL))/len(logL))
    return i_use, T, EV, is_
