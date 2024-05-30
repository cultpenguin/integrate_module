import h5py
import numpy as np
import os.path
import subprocess
from sys import exit
from multiprocessing import Pool
from multiprocessing import shared_memory
from functools import partial


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

import numpy as np

def logl_T_est(logL, N_above=10, P_acc_lev=0.2):
    """
    Estimate a temperature (T_est) based on a given logarithmic likelihood (logL), 
    a number (N_above), and an acceptance level (P_acc_lev).

    :param logL: An array of logarithmic likelihoods.
    :type logL: numpy.ndarray
    :param N_above: The number of elements above which to consider in the sorted logL array. Default is 10.
    :type N_above: int, optional
    :param P_acc_lev: The acceptance level for the calculation. Default is 0.2.
    :type P_acc_lev: float, optional
    :return: The estimated temperature. It's either a positive number or infinity.
    :rtype: float

    note: The function sorts the logL array in ascending order after normalizing the data by subtracting the maximum value from each element.
    It then removes any NaN values from the sorted array.
    If the sorted array is not empty, it calculates T_est based on the N_above+1th last element in the sorted array and the natural logarithm of P_acc_lev.
    If the sorted array is empty, it sets T_est to infinity.
    """
    sorted_logL = np.sort(logL - np.nanmax(logL))
    sorted_logL = sorted_logL[~np.isnan(sorted_logL)]
    
    if sorted_logL.size > 0:
        logL_lev = sorted_logL[-N_above-1]
        T_est = logL_lev / np.log(P_acc_lev)
        T_est = np.nanmax([1, T_est])
    else:
        T_est = np.inf

    return T_est

# def logl_T_est(logL, N_above=10, P_acc_lev=0.2):
#     sorted_logL = np.sort(logL - np.nanmax(logL))
#     sorted_logL = sorted_logL[~np.isnan(sorted_logL)]
    
#     if sorted_logL.size > 0:
#         logL_lev = sorted_logL[-N_above-1]
#         T_est = logL_lev / np.log(P_acc_lev)
#         T_est = np.nanmax([1, T_est])
#     else:
#         T_est = np.inf
    
#     return T_est

def lu_post_sample_logl(logL, ns=1, T=1):
    """
    Perform a likelihood-utility (LU) post-sampling on the given logarithmic likelihoods (logL).

    Parameters:

    :logL: (numpy.ndarray): An array of logarithmic likelihoods.
    :ns: (int, optional): The number of samples to generate. Default is 1.
    :T: (float, optional): The temperature parameter for the acceptance probability calculation. Default is 1.

    Returns:
    tuple: A tuple containing two numpy arrays. The first array contains the indices of the selected samples. 
           The second array contains the calculated acceptance probabilities for each likelihood.

    Note:
    The function calculates the acceptance probability for each likelihood by exponentiating the likelihood divided by the temperature T.
    It then generates a cumulative distribution function (CDF) from these probabilities.
    The function generates ns samples by selecting the index where the CDF first exceeds a randomly generated number.
    """
    N = len(logL)
    P_acc = np.exp((1/T) * (logL - np.nanmax(logL)))
    P_acc[np.isnan(P_acc)] = 0

    Cum_P = np.cumsum(P_acc)
    Cum_P = Cum_P / np.nanmax(Cum_P)
    dp = 1 / N
    p = np.array([i * dp for i in range(1, N+1)])

    i_use_all = np.zeros(ns, dtype=int)
    for is_ in range(ns):
        r = np.random.rand()
        i_use = np.where(Cum_P > r)[0][0]
        i_use_all[is_] = i_use+1
    
    return i_use_all, P_acc

def integrate_update_prior_attributes(f_prior_h5, **kwargs):
    """
    Update the 'is_discrete' attribute of datasets in an HDF5 file.

    This function iterates over all datasets in the provided HDF5 file. 
    If a dataset's name starts with 'M', the function checks if the dataset 
    has an 'is_discrete' attribute. If not, it checks if the dataset appears 
    to represent discrete data by sampling the first 1000 elements and checking 
    how many unique values there are. If there are fewer than 20 unique values, 
    it sets 'is_discrete' to 1; otherwise, it sets 'is_discrete' to 0. 
    The 'is_discrete' attribute is then added to the dataset.

    Parameters
    ----------
    f_prior_h5 : str
        The path to the HDF5 file to process.
    """
    
    showInfo = kwargs.get('showInfo', 0)
    
    # Check that hdf5 files exists
    if not os.path.isfile(f_prior_h5):
        print('File %s does not exist' % f_prior_h5)
        exit()  

    with h5py.File(f_prior_h5, 'a') as f:  # open file in append mode
        for name, dataset in f.items():
            if name.upper().startswith('M'):
                # Check if the attribute 'is_discrete' exists
                if 'is_discrete' in dataset.attrs:
                    if (showInfo>0):
                        print('%s: %s.is_discrete=%d' % (f_prior_h5,name,dataset.attrs['is_discrete']))
                else:
                    # Check if M is discrete
                    M_sample = dataset[:1000]  # get the first 1000 elements
                    class_id = np.unique(M_sample)
                    if len(class_id) < 20:
                        is_discrete = 1
                        dataset.attrs['class_id'] = class_id
                        ## convert class_id to an array of strings and save it as an attribute if the attribute does not 
                        ## already exist
                        if 'class_name' not in dataset.attrs:
                            dataset.attrs['class_name'] = np.array([str(x) for x in class_id])
                        
                    else:
                        is_discrete = 0

                    if (showInfo>0):
                        print(f'Setting is_discrete={is_discrete}, for {name}')
                    dataset.attrs['is_discrete'] = is_discrete

                if dataset.attrs['is_discrete']==1:
                    if not ('class_id' in dataset.attrs):
                        M_sample = dataset[:1000]  # get the first 1000 elements
                        class_id = np.unique(M_sample)
                        dataset.attrs['class_id'] = class_id
                    if not ('class_name' in dataset.attrs):
                        # Convert class_id to an array of strings and save it as an attribute if the attribute does not
                        class_id = dataset.attrs['class_id']
                        dataset.attrs['class_name'] = [str(x) for x in class_id]


def integrate_posterior_stats(f_post_h5='DJURSLAND_P01_N0100000_NB-13_NR03_POST_Nu1000_aT1.h5', **kwargs):
    import h5py
    import numpy as np
    import integrate
    import scipy as sp
    from tqdm import tqdm

    showInfo = kwargs.get('showInfo', 0)

    #f_post_h5='DJURSLAND_P01_N0100000_NB-13_NR03_POST_Nu50000_aT1.h5'
    # Check if f_prior_h5 attribute exists in the HDF5 file
    with h5py.File(f_post_h5, 'r') as f:
        if 'f5_prior' in f.attrs:
            f_prior_h5 = f.attrs['f5_prior']
        else:
            raise ValueError(f"'f5_prior' attribute does not exist in {f_post_h5}")

    integrate.integrate_update_prior_attributes(f_prior_h5, **kwargs)


    # Load 'i_use' data from the HDF5 file
    try:
        with h5py.File(f_post_h5, 'r') as f:
            i_use = f['i_use'][:]
    except KeyError:
        print(f"Could not read 'i_use' from {f_post_h5}")
        #return
    
    # Process each dataset in f_prior_h5
    with h5py.File(f_prior_h5, 'r') as f_prior, h5py.File(f_post_h5, 'a') as f_post:
        for name, dataset in f_prior.items():
                
            if name.upper().startswith('M') and 'is_discrete' in dataset.attrs and dataset.attrs['is_discrete'] == 0:
                if showInfo>0:
                    print('%s: CONTINUOUS' % name)

                nm = dataset.shape[1]
                nsounding, nr = i_use.shape
                m_post = np.zeros((nm, nr))

                M_mean = np.zeros((nsounding,nm))
                M_std = np.zeros((nsounding,nm))
                M_median = np.zeros((nsounding,nm))

                # Create datasets
                for stat in ['Mean', 'Median', 'Std']:
                    if stat not in f_post:
                        dset = '/%s/%s' % (name,stat)
                        if dset not in f_post:
                            print('Creating %s' % dset)
                            f_post.create_dataset(dset, (nsounding,nm))

                #if dataset.size <= 1e6:  # arbitrary threshold for loading all data into memory
                M_all = dataset[:]

                for iid in tqdm(range(nsounding), mininterval=1):
                    ir = np.int64(i_use[iid,:]-1)
                    m_post = M_all[ir,:]

                    m_mean = np.exp(np.mean(np.log(m_post), axis=0))
                    m_median = np.median(m_post, axis=0)
                    m_std = np.std(np.log10(m_post), axis=0)

                    M_mean[iid,:] = m_mean
                    M_median[iid,:] = m_median
                    M_std[iid,:] = m_std


                f_post['/%s/%s' % (name,'Mean')][:] = M_mean
                f_post['/%s/%s' % (name,'Median')][:] = M_median
                f_post['/%s/%s' % (name,'Std')][:] = M_std

            elif name.upper().startswith('M') and 'is_discrete' in dataset.attrs and dataset.attrs['is_discrete'] == 1:
                
                nm = dataset.shape[1]
                nsounding, nr = i_use.shape
                nsounding, nr = i_use.shape
                nm = dataset.shape[1]
                # Get number of classes for name    
                class_id = f_prior[name].attrs['class_id']
                n_classes = len(class_id)
                
                if showInfo>0:
                    print('%s: DISCRETE, N_classes =%d' % (name,n_classes))    

                M_mode = np.zeros((nsounding,nm))
                M_entropy = np.zeros((nsounding,nm))
                M_P= np.zeros((nsounding,n_classes,nm))

                # Create datasets in h5 file
                for stat in ['Mode', 'Entropy']:
                    if stat not in f_post:
                        dset = '/%s/%s' % (name,stat)
                        if dset not in f_post:
                            print('Creating %s' % dset)
                            f_post.create_dataset(dset, (nsounding,nm))
                for stat in ['Mode', 'P']:
                    if stat not in f_post:
                        dset = '/%s/%s' % (name,stat)
                        if dset not in f_post:
                            print('Creating %s' % dset)
                            f_post.create_dataset(dset, (nsounding,n_classes,nm))

                M_all = dataset[:]

                for iid in tqdm(range(nsounding), mininterval=1):

                    # Get the indices of the rows to use
                    ir = np.int64(i_use[iid,:]-1)
                    
                    # Load ALL DATA AND EXTRACT
                    # load from all models in memory
                    #m_post = dataset[:][ir,:]
                    m_post = M_all[ir,:]
                    # Load only the needed data
                    #m_post = np.zeros((nr,nm))
                    #for j in range(nr):
                    #    m_post[j,:] = dataset[ir[j],:]
                    

                    # Compute the class probability
                    n_count = np.zeros((n_classes,nm))
                    for ic in range(n_classes):
                        n_count[ic,:]=np.sum(class_id[ic]==m_post, axis=0)/nr    
                    M_P[iid,:,:] = n_count

                    # Compute the mode
                    M_mode[iid,:] = class_id[np.argmax(n_count, axis=0)]

                    # Compute the entropy
                    M_entropy[iid,:]=sp.stats.entropy(n_count, base=n_classes)

                f_post['/%s/%s' % (name,'Mode')][:] = M_median
                f_post['/%s/%s' % (name,'Entropy')][:] = M_entropy
                f_post['/%s/%s' % (name,'P')][:] = M_P


            else: 
                if (showInfo>0):
                    print('%s: NOT RECOGNIZED' % name.upper())
                
            
                

    return None

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


def sample_from_posterior(is_, d_sim, f_data_h5='tTEM-Djursland.h5', N_use=1000000, autoT=1, ns=400):
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



#def sample_from_posterior_chunk(is_,d_sim,f_data_h5, N_use,autoT,ns):
#    return sample_from_posterior(is_,d_sim,f_data_h5, N_use,autoT,ns) 

def integrate_rejection(f_prior_h5='DJURSLAND_P01_N0010000_NB-13_NR03_PRIOR.h5',
                            f_data_h5='tTEM-Djursland.h5',
                            f_post_h5='',
                            autoT=1,
                            N_use=1000000,
                            ns=400,
                            parallel=1, 
                            updatePostStat= True,
                            **kwargs):

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
            out = list(tqdm(p.imap(sample_from_posterior_shared, args_list, chunksize=1), total=nsoundings, mininterval=1))

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

#%% integrate_prior_data: updates PRIOR strutcure with DATA
def prior_data(f_prior_in_h5, f_forward_h5, id=1, im=1, doMakePriorCopy=0, parallel=True):
    # Check if at least two inputs are provided
    if f_prior_in_h5 is None or f_forward_h5 is None:
        print(f'{__name__}: Use at least two inputs to')
        help(__name__)
        return ''

    # Open HDF5 files
    with h5py.File(f_forward_h5, 'r') as f:
        # Check type=='TDEM'
        if 'type' in f.attrs:
            data_type = f.attrs['type']
        else:
            data_type = 'TDEM'

    f_prior_h5 = ''
    if data_type.lower() == 'tdem':
        # TDEM
        with h5py.File(f_forward_h5, 'r') as f:
            if 'method' in f.attrs:
                method = f.attrs['method']
            else:
                print(f'{__name__}: "TDEM/{method}" not supported')
                return

        if method.lower() == 'ga-aem':
            f_prior_h5, id, im = integrate_prior_data_gaaem(f_prior_in_h5, f_forward_h5, id, im, doMakePriorCopy)
        else:
            print(f'{__name__}: "TDEM/{method}" not supported')
            return
    elif data_type.lower() == 'identity':
        f_prior_h5, id, im = integrate_prior_data_identity(f_prior_in_h5, f_forward_h5, id, im, doMakePriorCopy)
    else:
        print(f'{__name__}: "{data_type}" not supported')
        return

    # update prior data with an attribute defining the prior
    with h5py.File(f_prior_h5, 'a') as f:
        f.attrs[f'/D{id}'] = 'f5_forward'

    return f_prior_h5


'''
Forward simulation
'''

def forward_gaaem(C=np.array(()), thickness=np.array(()), GEX={}, file_gex='', stmfiles=[], showtime=False, **kwargs):
    """
    Perform forward modeling using the GAAEM method.

    :param C: Conductivity array, defaults to np.array(())
    :type C: numpy.ndarray, optional
    :param thickness: Thickness array, defaults to np.array(())
    :type thickness: numpy.ndarray, optional
    :param GEX: GEX dictionary, defaults to {}
    :type GEX: dict, optional
    :param file_gex: Path to GEX file, defaults to ''
    :type file_gex: str, optional
    :param stmfiles: List of STM files, defaults to []
    :type stmfiles: list, optional
    :param showtime: Flag to display execution time, defaults to False
    :type showtime: bool, optional
    :param **kwargs: Additional keyword arguments
    :returns: Forward data as a numpy.ndarray
    :raises ValueError: If the thickness array does not match the number of layers minus 1
    :todo: Allow using only 1 moment/STM file at a time
    """

    from gatdaem1d import Earth;
    from gatdaem1d import Geometry;
    # Next should probably only be loaded if the DLL is not allready loaded!!!
    from gatdaem1d import TDAEMSystem; # loads the DLL!!
    import integrate as ig
    import time 
    from tqdm import tqdm

    showInfo = kwargs.get('showInfo', 0)
    doCompress = kwargs.get('doCompress', True)

    if (len(stmfiles)>0) and (file_gex != '') and (len(GEX)==0):
        # GEX FILE and STM FILES
        if (showInfo)>0:
            print('Using submitted GEX file (%s)' % (file_gex))
        GEX =   ig.read_gex(file_gex)
    elif (len(stmfiles)==0) and (file_gex != '') and (len(GEX)==0):
        # ONLY GEX FILE
        stmfiles, GEX = ig.gex_to_stm(file_gex, **kwargs)
    elif (len(stmfiles)>0) and (file_gex == '') and (len(GEX)>0):
        # Using GEX dict and STM FILES
        a = 1
    elif (len(GEX)>0) and (len(stmfiles)>1):
        # using the GEX file in stmfiles
        print('Using submitted GEX and STM files')
    elif (len(GEX)>0) and (len(stmfiles)==0):
        # using GEX file and writing STM files
        print('Using submitted GEX and writing STM files')
        stmfiles = ig.write_stm_files(GEX, **kwargs)
    elif (len(GEX)==0) and (len(stmfiles)>1):
        if (file_gex == ''):
            print('Error: file_gex not provided')
            return -1
        else:
            print('Converting STM files to GEX')
            GEX =   ig.read_gex(file_gex)
    elif (len(GEX)>0) and (len(stmfiles)==0):
        stmfiles, GEX = ig.gex_to_stm(file_gex, **kwargs)
    elif (file_gex != ''):
        a=1
        #stmfiles, GEX = ig.gex_to_stm(file_gex, **kwargs)
    else:   
        print('Error: No GEX or STM files provided')
        return -1

    if (showInfo>0):
        print('Using GEX file: ', GEX['filename'])

    nstm=len(stmfiles)
    if (showInfo>0):
        for i in range(len(stmfiles)):
            print('Using MOMENT:', stmfiles[i])

    if C.ndim==1:
        nd=1
        nl=C.shape[0]
    else:
        nd,nl=C.shape

    nt = thickness.shape[0]
    if nt != (nl-1):
        raise ValueError('Error: thickness array (nt=%d) does not match the number of layers minus 1(nl=%d)' % (nt,nl))

    if (showInfo>0):
        print('nd=%s, nl=%d,  nstm=%d' %(nd,nl,nstm))

    # SETTING UP t1=time.time()
    t1=time.time()
    #S=[]
    #for i in range(nstm):
    #    S.append = TDAEMSystem(stmfiles[i])
    
    S_LM = TDAEMSystem(stmfiles[0])
    if nstm>1:
        S_HM = TDAEMSystem(stmfiles[1])
        S=[S_LM, S_HM]
    else:
        S=[S_LM]
    t2=time.time()
    t_system = 1000*(t2-t1)
    if showtime:
        print("Time, Setting up systems = %4.1fms" % t_system)

    # Setting up geometry
    GEX = ig.read_gex(file_gex)
    txrx_dx = float(GEX['General']['RxCoilPosition1'][0])-float(GEX['General']['TxCoilPosition1'][0])
    txrx_dy = float(GEX['General']['RxCoilPosition1'][1])-float(GEX['General']['TxCoilPosition1'][1])
    txrx_dz = float(GEX['General']['RxCoilPosition1'][2])-float(GEX['General']['TxCoilPosition1'][2])
    tx_height = -float(GEX['General']['TxCoilPosition1'][2])
    #G = Geometry(tx_height=tx_height, txrx_dx = -txrx_dx, txrx_dz = -txrx_dz)
    #G = Geometry(tx_height=.01, txrx_dx = -12.62, txrx_dz = +2.16)
    G = Geometry(tx_height=tx_height, txrx_dx = txrx_dx, txrx_dy = txrx_dy, txrx_dz = txrx_dz)
    if (showInfo>0):
        print('tx_height=%f, txrx_dx=%f, txrx_dy=%f, txrx_dz=%f' % (tx_height, txrx_dx, txrx_dy, txrx_dz))
    
    ng0 = GEX['Channel1']['NoGates']-GEX['Channel1']['RemoveInitialGates'][0]
    if nstm>1:
        ng1 = GEX['Channel2']['NoGates']-GEX['Channel2']['RemoveInitialGates'][0]
    else:
        ng1 = 0

    
    ng = int(ng0+ng1)
    D = np.zeros((nd,ng))

    # Compute forward data
    t1=time.time()
    for i in tqdm(range(nd), mininterval=1):
        if C.ndim==1:
            # Only one model
            conductivity = C
        else:
            conductivity = C[i]

        #doCompress=True
        if doCompress:
            i_change=np.where(np.diff(conductivity) != 0 )[0]+1
            n_change = len(i_change)
            conductivity_compress = np.zeros(n_change+1)+conductivity[0]
            thickness_compress = np.zeros(n_change)
            for il in range(n_change):
                conductivity_compress[il+1] = conductivity[i_change[il]]
                if il==0:
                    thickness_compress[il]=np.sum(thickness[0:i_change[il]])
                else:   
                    i1=i_change[il-1]
                    i2=i_change[il]
                    #print("i1: %d, i2: %d" % (i1, i2))
                    thickness_compress[il]=np.sum(thickness[i1:i2]) 
            E = Earth(conductivity_compress,thickness_compress)
        else:   
            E = Earth(conductivity,thickness)

        fm0 = S[0].forwardmodel(G,E)
        d = -fm0.SZ
        if nstm>1:
            fm1 = S[1].forwardmodel(G,E)
            d1 = -fm1.SZ
            d = np.concatenate((d,d1))    

        D[i] = d    

        '''
        fm_lm = S_LM.forwardmodel(G,E)
        fm_hm = S_HM.forwardmodel(G,E)
        # combine -fm_lm.SZ and -fm_hm.SZ
        d = np.concatenate((-fm_lm.SZ,-fm_hm.SZ))
        d_ref = D[i]
        '''
        
    t2=time.time()
    if showtime:
        print("Time = %4.1fms per model and %d model tests" % (1000*(t2-t1)/nd, nd))

    return D



import time

def forward_gaaem_chunk(C_chunk, thickness, stmfiles, file_gex, Nhank, Nfreq, **kwargs):
    # pause for random time
    # time.sleep(np.random.rand()*10)
    return forward_gaaem(C=C_chunk, thickness=thickness, stmfiles=stmfiles, file_gex=file_gex, Nhank=Nhank, Nfreq=Nfreq, parallel=False, **kwargs)

def prior_data_gaaem(f_prior_h5, file_gex, doMakePriorCopy=True, im=1, id=1, Nhank=280, Nfreq=12, parallel=True, **kwargs):
    """
    Generate prior data for the ga-aem method.

    Parameters:
    - f_prior_h5 (str): Path to the prior data file in HDF5 format.
    - file_gex (str): Path to the file containing geophysical exploration data.
    - doMakePriorCopy (bool): Flag indicating whether to make a copy of the prior file (default: True).
    - im (int): Index of the model (default: 1).
    - id (int): Index of the data (default: 1).
    - Nhank (int): Number of Hankel transform quadrature points (default: 18).
    - Nfreq (int): Number of frequencies (default: 5).
    - parallel (bool): Flag indicating whether multiprocessing is used
    - Nproc (int): Number of processes to use (default: ncpus).
    
    Returns:
    f_prior_data: filename of hdf5 fille containing the updated prior data
    """
    import shutil
    import integrate as ig
    import multiprocessing
    from multiprocessing import Pool
    import time

    type = 'TDEM'
    method = 'ga-aem'
    showInfo = kwargs.get('showInfo', 0)
    Nproc = kwargs.get('Nproc', 0)
    
    if doMakePriorCopy:
        f_prior_data_h5 = '%s_%s_Nh%d_Nf%d.h5' % (os.path.splitext(f_prior_h5)[0], os.path.splitext(file_gex)[0], Nhank, Nfreq)
        if (showInfo>-1):
            print("Creating a copy of %s as %s" % (f_prior_h5, f_prior_data_h5))
        # make a copy of the prior file
        shutil.copy(f_prior_h5, f_prior_data_h5)
    else:
        f_prior_data_h5 = f_prior_h5

    Mname = '/M%d' % im
    Dname = '/D%d' % id
    f_prior = h5py.File(f_prior_data_h5, 'a')

    # Get thickness
    if 'x' in f_prior[Mname].attrs:
        z = f_prior[Mname].attrs['x']
    else:
        z = f_prior[Mname].attrs['z']
    thickness = np.diff(z)

    # Get conductivity
    if Mname in f_prior.keys():
        C = 1 / f_prior[Mname][:]
    else:
        print('Could not load %s from %s' % (Mname, f_prior_data_h5))

    N = f_prior[Mname].shape[0]
    t1 = time.time()
    if not parallel:
        # Sequential
        D = ig.forward_gaaem(C=C, thickness=thickness, file_gex=file_gex, Nhank=Nhank, Nfreq=Nfreq, parallel=parallel, **kwargs)
    else:
    
        # Make sure STM files are only written once!!! (need for multihreading)
        # D = ig.forward_gaaem(C=C[0:1,:], thickness=thickness, file_gex=file_gex, Nhank=Nhank, Nfreq=Nfreq, parallel=False, **kwargs)
        stmfiles, GEX = ig.gex_to_stm(file_gex, Nhank=Nhank, Nfreq=Nfreq, **kwargs)

        # Parallel
        if Nproc < 1 :
            Nproc =  int(multiprocessing.cpu_count()/2)
            Nproc =  int(multiprocessing.cpu_count())
        if (showInfo>0):
            print("Using %d parallel threads." % (Nproc))

        # 1: Define a function to compute a chunk
        ## OUTSIDE
        # 2: Create chunks
        C_chunks = np.array_split(C, Nproc)

        # 3: Compute the chunks in parallel
        forward_gaaem_chunk_partial = partial(forward_gaaem_chunk, thickness=thickness, stmfiles=stmfiles, file_gex=file_gex, Nhank=Nhank, Nfreq=Nfreq, **kwargs)

        # Create a multiprocessing pool and compute D for each chunk of C
        with Pool() as p:
            D_chunks = p.map(forward_gaaem_chunk_partial, C_chunks)
        
        #useIterative=0
        #if useIterative==1:
        #    D_chunks = []
        #    for C_chunk in C_chunks:    
        #        D_chunk = ig.forward_gaaem(C=C_chunk, thickness=thickness, file_gex=file_gex, Nhank=Nhank, Nfreq=Nfreq, parallel=False, **kwargs)
        #        D_chunks.append(D_chunk)

        # 4: Combine the chunks into D
        print('Concatenating D_chunks')
        D = np.concatenate(D_chunks)
        print("D.shape", D.shape)

        # D = ig.forward_gaaem(C=C, thickness=thickness, file_gex=file_gex, Nhank=Nhank, Nfreq=Nfreq, parallel=parallel, **kwargs)

    t2 = time.time()
    t_elapsed = t2 - t1
    print('Time elapsed: %5.1f s, for %d soundings. %4.3f ms/sounding. %4.1fit/s' % (t_elapsed, N, 1000*t_elapsed/N,N/t_elapsed))
    
    # Write D to f_prior['/D1']
    f_prior[Dname] = D

    # Add method, type, file_ex, and im as attributes to '/D1'
    f_prior[Dname].attrs['method'] = method
    f_prior[Dname].attrs['type'] = type
    f_prior[Dname].attrs['im'] = im
    f_prior[Dname].attrs['Nhank'] = Nhank
    f_prior[Dname].attrs['Nfreq'] = Nfreq

    f_prior.close()

    return f_prior_data_h5
# %%





def prior_model_layered(lay_dist='uniform', dz = 1, z_max = 90, NLAY_min=3, NLAY_max=6, NLAY_deg=6, rho_dist='log-uniform', RHO_min=0.1, RHO_max=100, RHO_MEAN=100, RHO_std=80, N=100000):
    """
    Generate a prior model with layered structure.

    Args:
        lay_dist (str): Distribution of the number of layers. Options are 'chi2' and 'uniform'. Default is 'chi2'.
        NLAY_min (int): Minimum number of layers. Default is 3.
        NLAY_max (int): Maximum number of layers. Default is 6.
        NLAY_deg (int): Degrees of freedom for chi-square distribution. Only applicable if lay_dist is 'chi2'. Default is 6.
        rho_dist (str): Distribution of resistivity within each layer. Options are 'log-uniform', 'uniform', 'normal', and 'lognormal'. Default is 'log-uniform'.
        RHO_min (float): Minimum resistivity value. Default is 0.1.
        RHO_max (float): Maximum resistivity value. Default is 100.
        RHO_MEAN (float): Mean resistivity value. Only applicable if rho_dist is 'normal' or 'lognormal'. Default is 100.
        RHO_std (float): Standard deviation of resistivity value. Only applicable if rho_dist is 'normal' or 'lognormal'. Default is 80.
        N (int): Number of prior models to generate. Default is 100000.

    Returns:
        str: Filepath of the saved prior model.

    """
    
    from tqdm import tqdm

    if NLAY_max < NLAY_min:
        #raise ValueError('NLAY_max must be greater than or equal to NLAY_min.')
        NLAY_max = NLAY_min

    if NLAY_min < 1:
        #raise ValueError('NLAY_min must be greater than or equal to 1.')
        NLAY_min = 1
        
    if lay_dist == 'uniform':
        NLAY = np.random.randint(NLAY_min, NLAY_max+1, N)
        f_prior_h5 = 'PRIOR_UNIFORM_NL_%d-%d_%s_N%d.h5' % (NLAY_min, NLAY_max, rho_dist, N)

    elif lay_dist == 'chi2':
        NLAY = np.random.chisquare(NLAY_deg, N)
        NLAY = np.ceil(NLAY).astype(int)    
        f_prior_h5 = 'PRIOR_CHI2_NF_%d_%s_N%d.h5' % (NLAY_deg, rho_dist, N)

    # Force NLAY to be a 2 dimensional numpy array
    NLAY = NLAY[:, np.newaxis]
    
    z_min = 0
    z = np.arange(z_min, z_max, dz)
    nz= len(z)
    M_rho = np.zeros((N, nz))

    # save to hdf5 file
    
    #% simulate the number of layers as in integer
    for i in tqdm(range(N), mininterval=1):
    
        i_boundaries = np.sort(np.random.choice(nz, NLAY[i]-1, replace=False))        

        ### simulate the resistivity in each layer
        if rho_dist=='log-normal':
            rho_all=np.random.lognormal(mean=np.log10(RHO_MEAN), sigma=np.log10(RHO_std), size=NLAY[i])
        elif rho_dist=='normal':
            rho_all=np.random.normal(mean=RHO_MEAN, sigma=RHO_std, size=NLAY[i])
        elif rho_dist=='log-uniform':
            rho_all=np.exp(np.random.uniform(np.log(RHO_min), np.log(RHO_max), NLAY[i]))
        elif rho_dist=='uniform':
            rho_all=np.random.uniform(RHO_min, RHO_max, NLAY[i])

        rho = np.zeros(nz)+rho_all[0]
        for j in range(len(i_boundaries)):
            rho[i_boundaries[j]:] = rho_all[j+1]

        M_rho[i]=rho        


    print("Saving prior model to %s" % f_prior_h5)
    f_prior = h5py.File(f_prior_h5, 'w')
    f_prior.create_dataset('/M1', data=M_rho)
    f_prior['/M1'].attrs['is_discrete'] = 0
    f_prior['/M1'].attrs['z'] = z
    f_prior['/M1'].attrs['x'] = z
    f_prior.create_dataset('/M2', data=NLAY)
    f_prior['/M2'].attrs['is_discrete'] = 0
    f_prior['/M2'].attrs['z'] = z
    f_prior['/M2'].attrs['x'] = z
    f_prior.close()    

    return f_prior_h5

def prior_model_workbench(N=100000, rho_dist='log-uniform', z1=0, z2= 100, nlayers=30, p=2, rho_min = 1, rho_max= 300, rho_mean=180, rho_std=80, chi2_deg= 100):
    """

    Generate a prior model with increasingly thick layers
 
    Args:
        N (int): Number of prior models to generate. Default is 100000.
        rho_dist (str): Distribution of resistivity within each layer. Options are 'log-uniform', 'uniform', 'normal', 'lognormal', and 'chi2'. Default is 'log-uniform'.
        # rho_dist='uniform', 'log-uniform'
        rho_min (float): Minimum resistivity value. Default is 0.1.
        rho_max (float): Maximum resistivity value. Default is 100.
        # rho_dist='normal', 'log-normal'
        rho_mean (float): Mean resistivity value. Only applicable if rho_dist is 'normal' or 'lognormal'. Default is 100.
        rho_std (float): Standard deviation of resistivity value. Only applicable if rho_dist is 'normal' or 'lognormal'. Default is 80.
        # rho_dist='chi2'
        chi2_def (int): Degrees of freedom for chi2 distribution. Only applicable if rho_dist is 'chi2'. Default is 100.

    Returns:
        str: Filepath of the saved prior model.


    """
    import numpy as np
    import matplotlib.pyplot as plt
    import h5py
    from tqdm import tqdm
    

    f_prior_h5 = 'PRIOR_WB%d_N%d_%s' % (nlayers,N,rho_dist)

    z= z1 + (z2 - z1) * np.linspace(0, 1, nlayers) ** p

    nz = len(z)
    
    if rho_dist=='uniform':
        M_rho = np.random.uniform(low=rho_min, high = rho_max, size=(N, nz))
        f_prior_h5 = '%s_R%g_%g.h5' % (f_prior_h5, rho_min, rho_max)
    elif rho_dist=='log-uniform':
        M_rho = np.exp(np.random.uniform(low=np.log(rho_min), high = np.log(rho_max), size=(N, nz)))
        f_prior_h5 = '%s_R%g_%g.h5' % (f_prior_h5, rho_min, rho_max)
    elif rho_dist=='normal':
        M_rho = np.random.normal(loc=rho_mean, scale = rho_std, size=(N, nz))
        f_prior_h5 = '%s_R%g_%g.h5' % (f_prior_h5, rho_mean, rho_std)
    elif rho_dist=='log-normal':
        M_rho = np.random.lognormal(mean=np.log(rho_mean), sigma = rho_std/rho_mean, size=(N, nz))
        f_prior_h5 = '%s_R%g_%g.h5' % (f_prior_h5, rho_mean, rho_std)
    elif rho_dist=='chi2':
        M_rho = np.random.chisquare(df = chi2_deg, size=(N, nz))
        f_prior_h5 = '%s_deg%d.h5' % (f_prior_h5,chi2_deg)

    f_prior_h5 = f_prior_h5 + '.h5'

    print("Saving prior model to %s" % f_prior_h5)
    f_prior = h5py.File(f_prior_h5, 'w')
    f_prior.create_dataset('/M1', data=M_rho)
    f_prior['/M1'].attrs['is_discrete'] = 0
    f_prior['/M1'].attrs['z'] = z
    f_prior['/M1'].attrs['x'] = z

    #plt.figure()
    #plt.subplot(121)
    #plt.hist(M_rho.flatten())
    #plt.subplot(122)
    #plt.hist(np.log10(M_rho.flatten()))

    # return the full filepath to f_prior_h5
    
    return f_prior_h5

    
