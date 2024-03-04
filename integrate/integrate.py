import h5py
import numpy as np
import os.path
import subprocess
from sys import exit



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

def logl_T_est(logL, N_above=10, P_acc_lev=0.2):
    """
    Estimate a temperature (T_est) based on a given logarithmic likelihood (logL), 
    a number (N_above), and an acceptance level (P_acc_lev).

    Parameters:
    logL (numpy.ndarray): An array of logarithmic likelihoods.
    N_above (int, optional): The number of elements above which to consider in the sorted logL array. Default is 10.
    P_acc_lev (float, optional): The acceptance level for the calculation. Default is 0.2.

    Returns:
    float: The estimated temperature. It's either a positive number or infinity.

    Note:
    The function sorts the logL array in ascending order after normalizing the data by subtracting the maximum value from each element.
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
    logL (numpy.ndarray): An array of logarithmic likelihoods.
    ns (int, optional): The number of samples to generate. Default is 1.
    T (float, optional): The temperature parameter for the acceptance probability calculation. Default is 1.

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

def integrate_update_prior_attributes(f_prior_h5):
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
    
    # Check that hdf5 files exists
    if not os.path.isfile(f_prior_h5):
        print('File %s does not exist' % f_prior_h5)
        exit()  
    with h5py.File(f_prior_h5, 'a') as f:  # open file in append mode
        for name, dataset in f.items():
            if name.upper().startswith('M'):
                # Check if the attribute 'is_discrete' exists
                if 'is_discrete' in dataset.attrs:
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


def integrate_posterior_stats(f_post_h5='DJURSLAND_P01_N0100000_NB-13_NR03_POST_Nu1000_aT1.h5'):
    import h5py
    import numpy as np
    import integrate
    from tqdm import tqdm

    #f_post_h5='DJURSLAND_P01_N0100000_NB-13_NR03_POST_Nu50000_aT1.h5'
    # Check if f_prior_h5 attribute exists in the HDF5 file
    with h5py.File(f_post_h5, 'r') as f:
        if 'f5_prior' in f.attrs:
            f_prior_h5 = f.attrs['f5_prior']
        else:
            raise ValueError(f"'f5_prior' attribute does not exist in {f_post_h5}")

    integrate.integrate_update_prior_attributes(f_prior_h5)


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
            print(name.upper())
                
            if name.upper().startswith('M') and 'is_discrete' in dataset.attrs and dataset.attrs['is_discrete'] == 0:
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

                for iid in tqdm(range(nsounding)):
                    ir = np.int64(i_use[iid,:]-1)
                    #if dataset.size <= 1e6:
                    m_post = M_all[ir,:]
                    #else:
                    #    for j in range(len(ir)):
                    #        m_post[:, j] = dataset[:, ir[j]]

                    m_mean = np.mean(m_post, axis=0)
                    m_median = np.median(m_post, axis=0)
                    m_std = np.std(m_post, axis=0)

                    M_mean[iid,:] = m_mean
                    M_median[iid,:] = m_median
                    M_std[iid,:] = m_std

                    #¤f_post['/%s/%s' % (name,'Mean')][iid,:] = m_mean
                    #f_post['/%s/%s' % (name,'Median')][iid,:] = m_median
                    #f_post['/%s/%s' % (name,'Std')][iid,:] = m_std

                f_post['/%s/%s' % (name,'Mean')][:] = M_mean
                f_post['/%s/%s' % (name,'Median')][:] = M_median
                f_post['/%s/%s' % (name,'Std')][:] = M_std

    return 1

def sample_from_posterior(is_, d_sim, f_data_h5='tTEM-Djursland.h5', N_use=1000000, autoT=1, ns=400):
            
    d_obs = h5py.File(f_data_h5, 'r')['/D1/d_obs'][is_,:]
    d_std = h5py.File(f_data_h5, 'r')['/D1/d_std'][is_,:]
    i_use = np.where(~np.isnan(d_obs) & (np.abs(d_obs) > 0))[0]
    
    d_obs = d_obs[i_use]
    d_var = d_std[i_use]**2

    logL = np.zeros(N_use)
    for i in range(N_use):
        dd = (d_sim[is_,i_use] - d_obs)**2
        logL[i] = -.5*np.sum(dd/d_var)

    if autoT == 1:
        T = logl_T_est(logL)
    else:
        T = 1
    maxlogL = np.nanmax(logL)
    
    i_use, P_acc = lu_post_sample_logl(logL, ns, T)
    EV=maxlogL + np.log(np.nansum(np.exp(logL-maxlogL))/len(logL))
    return i_use, T, EV, is_





def integrate_rejection(f_prior_h5='DJURSLAND_P01_N0010000_NB-13_NR03_PRIOR.h5',
                            f_data_h5='tTEM-Djursland.h5',
                            autoT=1,
                            N_use=1000000,
                            ns=400,
                            parallel=0):

    import h5py
    import numpy as np
    from datetime import datetime   
    import argparse
    from tqdm import tqdm
    from functools import partial
    from multiprocessing import Pool
 

    print('Running: integrate_rejection.py %s %s --autoT %d --N_use %d --ns %d -parallel %d' % (f_prior_h5,f_data_h5,autoT,N_use,ns,parallel))

    #% Check that hdf5 files exists
    import os.path
    if not os.path.isfile(f_prior_h5):
        print('File %s does not exist' % f_prior_h5)
        exit()  
    if not os.path.isfile(f_data_h5):
        print('File %s does not exist' % f_data_h5)
        exit()

    f_post_h5=False 

    with h5py.File(f_data_h5, 'r') as f:
        d_obs = f['/D1/d_obs']
        nd = d_obs.shape[1]
        nsoundings = d_obs.shape[0]

    data_str = '/D1'
    with h5py.File(f_prior_h5, 'r') as f:
        d_sim = f[data_str]
        N = d_sim.shape[0]
    N_use = min([N_use, N])
    # Read 'd' for to get the correct subset
    with h5py.File(f_prior_h5, 'r') as f:
        d_sim = f[data_str][:, :N_use]

    if not f_post_h5:
        f_post_h5 = f"{f_prior_h5[:-9]}_POST_Nu{N_use}_aT{autoT}.h5"

    print('nsoundings:%d, N_use:%d, nd:%d' % (nsoundings,N_use,nd))
    print('Writing results to ',f_post_h5)

    # remaining code...

    date_start = str(datetime.now())
    t_start = datetime.now()
    i_use_all = np.zeros((ns,nsoundings), dtype=int)
    POST_T = np.ones(nsoundings) * np.nan
    POST_EV = np.ones(nsoundings) * np.nan

    if parallel:
        # # % PARALLEL
        print('CALL SCRIPT FROM COMMANDLINE!!!!')
        cmd = 'python integrate_rejection.py %s %s --N_use=%d --autoT=%d --ns=%d' % (f_prior_h5,f_data_h5,N_use,autoT,ns) 
        print('Executing "%s"'%(cmd))
        import os
        os.system(cmd)
    else:
        # SEQUENTIAL
        for is_ in tqdm(range(nsoundings)):
            i_use, T, EV, is_out = sample_from_posterior(is_,d_sim,f_data_h5, N_use,autoT,ns)            
            POST_T[is_] = T
            POST_EV[is_] = EV
            i_use_all[:,is_] = i_use
    
        date_end = str(datetime.now())
        t_end = datetime.now()
        t_elapsed = (t_end - t_start).total_seconds()
        t_per_sounding = t_elapsed / nsoundings

        print('Average temperature, T=%3.1f ' % (np.nanmean(POST_T)))
        print('Time elapsed: %5.1f s, for %d soundings' % (t_elapsed,nsoundings))
        print('Time per sounding: %4.3f ms' % (t_per_sounding*1000))

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

    doPosteriorStats = True
    #doPosteriorStats = False
    if doPosteriorStats:
        print('Running posterior stats')
        integrate_posterior_stats(f_post_h5)
    
    return f_post_h5

#%% integrate_prior_data: updates PRIOR strutcure with DATA
def integrate_prior_data(f_prior_in_h5, f_forward_h5, id=1, im=1, doMakePriorCopy=0):
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
