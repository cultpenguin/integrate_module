import h5py
import numpy as np
import os.path
import subprocess
from sys import exit
import multiprocessing
from multiprocessing import Pool
from multiprocessing import shared_memory
from multiprocessing import get_context
from functools import partial
import time

# %% Set up logging.. USed to test creation and use of shared memory
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING) # For production
#logger.setLevel(logging.DEBUG)  # For debugging    
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def is_notebook():
    """
    Check if the code is running in a Jupyter notebook or IPython shell.

    Returns:
        bool: True if running in a Jupyter notebook or IPython shell, False otherwise.
    """
    try:
        # Get the shell type from IPython
        shell = get_ipython().__class__.__name__
        
        if shell == 'ZMQInteractiveShell':
            # Additional check for VS Code
            import sys
            if 'vscode' in sys.modules:
                return False
            return True
        else:
            return False
            
    except NameError:  # If get_ipython is not defined (standard Python)
        return False


def use_parallel(**kwargs):
    """
    Determine if parallel processing can be used based on the environment.
    This function checks if the code is running in a Jupyter notebook or on a 
    POSIX system (e.g., Linux). If either condition is met, parallel processing 
    is considered OK. Otherwise, it is not recommended unless the primary script 
    is embedded in an `if __name__ == "__main__":` block.
    :param kwargs: Additional keyword arguments.
        - showInfo (int): If greater than 0, prints information about the 
          environment and parallel processing status. Default is 0.
    :return: 
        bool: True if parallel processing is OK, False otherwise.
    """
    import os
    showInfo = kwargs.get('showInfo', 0)
    
    parallel = True
    if is_notebook():
        # Then it is always OK to use parallel processing
        if showInfo>0:
            print('Notebook detected. Parallel processing is OK')
        parallel = True
    
    else:
        # if os is Linux, when default is spawn, then parallel processing is OK
        if os.name == 'posix':
            if os.uname().sysname == 'Darwin':
                if showInfo>0:
                    # MacOS use fork, which is not OK
                    print('MacOS detected. Parallel processing is not OK')        
                parallel = False
            else:
                if showInfo>0:
                    print('Posix system detected. Parallel processing is OK')        
                parallel = True            
        else:
            parallel = False
        if not parallel:    
            if showInfo>0:
                print('Non posix system detected. Parallel processing is not OK')        
                print('If parallel processing is needed, make sure to embed you primary script in a :if __name__ == "__main__": block')        
            parallel = False

    return parallel

    


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


def lu_post_sample_logl(logL, ns=1, T=1):
    """
    Perform LU post-sampling log-likelihood calculation.

    :param logL: Array of log-likelihood values.
    :type logL: array-like
    :param ns: Number of samples to generate. Defaults to 1.
    :type ns: int, optional
    :param T: Temperature parameter. Defaults to 1.
    :type T: float, optional

    :return: A tuple containing the generated samples and the acceptance probabilities.
    :rtype: tuple
        - i_use_all (numpy darray): Array of indices of the selected samples.
        - P_acc (numpy array): Array of acceptance probabilities.
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
        i_use_all[is_] = i_use
    
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

    :param f_prior_h5: The path to the HDF5 file to process.
    :type f_prior_h5: str
    """
    
    showInfo = kwargs.get('showInfo', 0)
    
    # Check that hdf5 files exists
    if not os.path.isfile(f_prior_h5):
        print('File %s does not exist' % f_prior_h5)
        exit()  

    with h5py.File(f_prior_h5, 'a') as f:  # open file in append mode
        for name, dataset in f.items():
            if showInfo>0:
                print("integrate_update_prior_attributes: Checking %s" % (name))
            if name.upper().startswith('M'):
                # Check if the attribute 'is_discrete' exists
                if 'x' in dataset.attrs:
                    pass
                else:
                    if 'z' in dataset.attrs:
                        dataset.attrs['x'] = dataset.attrs['z']
                    else:
                        x = np.arange(dataset.shape[1])
                        dataset.attrs['x'] = x
                        print(dataset.attrs)
                        #if 'M1' in f.keys():
                        #    if 'x' in f['/M1'].attrs.keys():
                        #        f[name].attrs['x'] = f['/M1'].attrs['x']
                        #        print('Setting %s/x = /M1/x ' % name)
                        #    else:
                        #        print('No x attribute found in %s' % name)    
                
                if 'is_discrete' in dataset.attrs:
                    if (showInfo>0):
                        print('%s: %s.is_discrete=%d' % (f_prior_h5,name,dataset.attrs['is_discrete']))
                else:
                    # Check if M is discrete
                    M_sample = dataset[:1000]  # get the first 1000 elements
                    class_id = np.unique(M_sample)
                    print(class_id)
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


def integrate_posterior_stats(f_post_h5='POST.h5', **kwargs):
    """
    Compute posterior statistics for datasets in an HDF5 file.

    This function computes various statistics for datasets in an HDF5 file based on the posterior samples.
    The statistics include mean, median, standard deviation for continuous datasets, and mode, entropy, and class probabilities for discrete datasets.
    The computed statistics are stored in the same HDF5 file.

    :param f_post_h5: The path to the HDF5 file to process.
    :type f_post_h5: str
    :param usePrior: Flag indicating whether to use the prior samples. Default is False.
    :type usePrior: bool
    :param kwargs: Additional keyword arguments.
    :type kwargs: dict
    """
    import h5py
    import numpy as np
    import integrate
    import scipy as sp
    from tqdm import tqdm

    showInfo = kwargs.get('showInfo', 0)
    if showInfo<0:
        disableTqdm=True
    else:
        disableTqdm=False
    usePrior = kwargs.get('usePrior', False)
    updateGeometryFromData = kwargs.get('updateGeometryFromData', True)

    #f_post_h5='DJURSLAND_P01_N0100000_NB-13_NR03_POST_Nu50000_aT1.h5'
    # Check if f_prior_h5 attribute exists in the HDF5 file
    with h5py.File(f_post_h5, 'r') as f:
        if 'f5_prior' in f.attrs:
            f_prior_h5 = f.attrs['f5_prior']
        else:
            f_prior_h5 = None
            raise ValueError(f"'f5_prior' attribute does not exist in {f_post_h5}")

    # Check if f5_data attribute exists in the HDF5 file
    with h5py.File(f_post_h5, 'r') as f:
        if 'f5_data' in f.attrs:
            f_data_h5 = f.attrs['f5_data']
        else:
            f_data_h5 = None
            raise ValueError(f"'f5_data' attribute does not exist in {f_post_h5}")

    # update Geometry from f_data_h5
    if (updateGeometryFromData)&(f_data_h5 is not None):
        with h5py.File(f_data_h5, 'r') as f_data, h5py.File(f_post_h5, 'a') as f_post:
            if '/UTMX' in f_data:
                if '/UTMX' not in f_post:
                    f_data.copy('/UTMX', f_post)
            if '/UTMY' in f_data:
                if '/UTMY' not in f_post:
                    f_data.copy('/UTMY', f_post)
            if '/LINE' in f_data:
                if '/LINE' not in f_post:
                    f_data.copy('/LINE', f_post)
            if '/ELEVATION' in f_data:
                if '/ELEVATION' not in f_post:
                    f_data.copy('/ELEVATION', f_post)
    
    # Load 'i_use' data from the HDF5 file
    try:
        with h5py.File(f_post_h5, 'r') as f:
            i_use = f['i_use'][:]
    except KeyError:
        print(f"Could not read 'i_use' from {f_post_h5}")
        #return
    
    if usePrior:
        with h5py.File(f_prior_h5, 'r') as f_prior:
            N = f_prior['/M1'].shape[0]
            nr=i_use.shape[1]
            nd=i_use.shape[0]
            # compute i_use of  (nd,nr), with random integer numbers between 0 and N-1
            i_use = np.random.randint(0, N, (nd,nr))
    
    # Process each dataset in f_prior_h5
    with h5py.File(f_prior_h5, 'r') as f_prior, h5py.File(f_post_h5, 'a') as f_post:
        for name, dataset in f_prior.items():
                
            if name.upper().startswith('M') and 'is_discrete' in dataset.attrs and dataset.attrs['is_discrete'] == 0:
                if showInfo>0:
                    print('%s: CONTINUOUS' % name)

                nm = dataset.shape[1]
                nsounding, nr = i_use.shape
                m_post = np.zeros((nm, nr))

                M_logmean = np.zeros((nsounding,nm))
                M_mean = np.zeros((nsounding,nm))
                M_std = np.zeros((nsounding,nm))
                M_median = np.zeros((nsounding,nm))

                if showInfo>0:
                    print('nm=%d, nsounding=%d, nr=%d' % (nm, nsounding, nr))
                    print('M_mean.shape=%s' % str(M_mean.shape))

                # Create datasets
                for stat in ['Mean', 'Median', 'Std','LogMean']:
                    if stat not in f_post:
                        dset = '/%s/%s' % (name,stat)
                        if dset not in f_post:
                            if (showInfo>0):
                                print('Creating %s in %s' % (dset,f_post_h5 ))
                            f_post.create_dataset(dset, (nsounding,nm))

                #if dataset.size <= 1e6:  # arbitrary threshold for loading all data into memory
                M_all = dataset[:]

                for iid in tqdm(range(nsounding), mininterval=1, disable=disableTqdm, desc='poststat', leave=False):
                    ir = np.int64(i_use[iid,:])
                    m_post = M_all[ir,:]

                    m_logmean = np.exp(np.mean(np.log(m_post), axis=0))
                    m_mean = np.mean(m_post, axis=0)
                    m_median = np.median(m_post, axis=0)
                    # Handle negative and zero values when taking log10 for standard deviation
                    with np.errstate(invalid='ignore', divide='ignore'):
                        m_std = np.std(np.log10(np.maximum(m_post, 1e-10)), axis=0)

                    M_logmean[iid,:] = m_logmean
                    M_mean[iid,:] = m_mean
                    M_median[iid,:] = m_median
                    M_std[iid,:] = m_std

                f_post['/%s/%s' % (name,'LogMean')][:] = M_logmean
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
                            if (showInfo>0):
                                print('Creating %s in %s' % (dset,f_post_h5 ))
                            f_post.create_dataset(dset, (nsounding,nm))
                for stat in ['Mode', 'P']:
                    if stat not in f_post:
                        dset = '/%s/%s' % (name,stat)
                        if dset not in f_post:
                            if (showInfo>0):
                                print('Creating %s' % dset)
                            f_post.create_dataset(dset, (nsounding,n_classes,nm))

                M_all = dataset[:]

                for iid in tqdm(range(nsounding), mininterval=1, disable=disableTqdm, desc='poststat', leave=False):

                    # Get the indices of the rows to use
                    ir = np.int64(i_use[iid,:])
                    
                    m_post = M_all[ir,:]
                    
                    # Compute the class probability
                    n_count = np.zeros((n_classes,nm))
                    for ic in range(n_classes):
                        n_count[ic,:]=np.sum(class_id[ic]==m_post, axis=0)/nr    
                    M_P[iid,:,:] = n_count

                    # Compute the mode
                    M_mode[iid,:] = class_id[np.argmax(n_count, axis=0)]

                    # Compute the entropy
                    M_entropy[iid,:]=sp.stats.entropy(n_count, base=n_classes)

                f_post['/%s/%s' % (name,'Mode')][:] = M_mode
                f_post['/%s/%s' % (name,'Entropy')][:] = M_entropy
                f_post['/%s/%s' % (name,'P')][:] = M_P


            else: 
                if (showInfo>0):
                    print('%s: NOT RECOGNIZED' % name.upper())
                
            
                

    return None


def sample_from_posterior(is_, d_sim, f_data_h5='tTEM-Djursland.h5', N_use=1000000, autoT=1, ns=400):
    r"""
    Sample from the posterior distribution.

    Parameters:
    - is\_ (int): Index of data f_data_h5.
    - d_sim (ndarray): Simulated data.
    - f_data_h5 (str): Filepath of the data file (default: 'tTEM-Djursland.h5').
    - N_use (int): Number of samples to use (default: 1000000).
    - autoT (int): Flag indicating whether to estimate temperature (default: 1).
    - ns (int): Number of samples to draw from the posterior (default: 400).

    Returns:
    - i_use (ndarray): Indices of the samples used.
    - T (float): Temperature.
    - EV (float): Expected value.
    - is\_ (int): Index of the posterior sample.
    """
    with h5py.File(f_data_h5, 'r') as f:
        d_obs = f['/D1/d_obs'][is_,:]
        d_std = f['/D1/d_std'][is_,:]
    
    i_use = np.where(~np.isnan(d_obs) & (np.abs(d_obs) > 0))[0]
    d_obs = d_obs[i_use]
    d_var = d_std[i_use]**2

    dd = (d_sim[:, i_use] - d_obs)**2
    #logL = -.5*np.sum(dd/d_var, axis=1)
    logL = np.sum(-0.5 * dd / d_var, axis=1)

    # Compute the annealing temperature
    if autoT == 1:
        T = logl_T_est(logL)
    else:
        T = 1
    maxlogL = np.nanmax(logL)
    
    # Find ns realizations of the posterior, using the log-likelihood values logL, and the annealing tempetrature T 
    i_use, P_acc = lu_post_sample_logl(logL, ns, T)
    
    # Compute the evidence
    exp_logL = np.exp(logL - maxlogL)
    EV = maxlogL + np.log(np.nansum(exp_logL)/len(logL))
    return i_use, T, EV, is_




#def sample_from_posterior_chunk(is_,d_sim,f_data_h5, N_use,autoT,ns):
#    return sample_from_posterior(is_,d_sim,f_data_h5, N_use,autoT,ns) 

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


    integrate_update_prior_attributes(f_prior_h5)

    return f_prior_h5


'''
Forward simulation
'''

def forward_gaaem(C=np.array(()), 
                    thickness=np.array(()), 
                    stmfiles=[], 
                    tx_height=np.array(()), 
                    txrx_dx = -13, 
                    txrx_dy = 0,
                    txrx_dz     = .1,
                    GEX={}, 
                    file_gex='', 
                    showtime=False, 
                    **kwargs):
    """
    Perform forward modeling using the **GAAEM** method.

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
    :param tx_height: Transmitter height array, defaults to np.array(())
    :type tx_height: numpy.ndarray, optional
    :param showtime: Flag to display execution time, defaults to False
    """
    
    from gatdaem1d import Earth;
    from gatdaem1d import Geometry;
    # Next should probably only be loaded if the DLL is not allready loaded!!!
    from gatdaem1d import TDAEMSystem; # loads the DLL!!
    import integrate as ig
    import time 
    from tqdm import tqdm

    showInfo = kwargs.get('showInfo', 0)
    if (showInfo<0):
        disableTqdm=True
    else:
        disableTqdm=False

    doCompress = kwargs.get('doCompress', True)

    if (len(stmfiles)>0) and (file_gex != '') and (len(GEX)==0):
        # GEX FILE and STM FILES
        if (showInfo)>1:
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
            if (showInfo>-1):
                print('Using STM files without GEX file')
            #return -1
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
        print('Using STM files : ')
        print(stmfiles)

    if (showInfo>1):
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
    if len(GEX)>0:
        GEX = ig.read_gex(file_gex)
        if 'TxCoilPosition1' in GEX['General']:
            # Typical for tTEM system
            txrx_dx = float(GEX['General']['RxCoilPosition1'][0])-float(GEX['General']['TxCoilPosition1'][0])
            txrx_dy = float(GEX['General']['RxCoilPosition1'][1])-float(GEX['General']['TxCoilPosition1'][1])
            txrx_dz = float(GEX['General']['RxCoilPosition1'][2])-float(GEX['General']['TxCoilPosition1'][2])
            if len(tx_height)==0:
                tx_height = -float(GEX['General']['TxCoilPosition1'][2])
                tx_height=np.array([tx_height])

        else:
            # Typical for SkyTEM system
            txrx_dx = float(GEX['General']['RxCoilPosition1'][0])
            txrx_dy = float(GEX['General']['RxCoilPosition1'][1])
            txrx_dz = float(GEX['General']['RxCoilPosition1'][2])
            if len(tx_height)==0:
                tx_height=np.array([40])
    

        # Set geometru once, if tx_height has one value
        if len(tx_height)==1:
            if (showInfo>1):
                print('Using tx_height=%f' % tx_height[0])
            G = Geometry(tx_height=tx_height, txrx_dx = txrx_dx, txrx_dy = txrx_dy, txrx_dz = txrx_dz)
        if (showInfo>1):
            print('tx_height=%f, txrx_dx=%f, txrx_dy=%f, txrx_dz=%f' % (tx_height[0], txrx_dx, txrx_dy, txrx_dz))
        
        ng0 = GEX['Channel1']['NoGates']-GEX['Channel1']['RemoveInitialGates'][0]
        if nstm>1:
            ng1 = GEX['Channel2']['NoGates']-GEX['Channel2']['RemoveInitialGates'][0]
        else:
            ng1 = 0
        ng = int(ng0+ng1)
    
    else:
        if len(tx_height)==0:
            tx_height=np.array([0])
        G = Geometry(tx_height=tx_height, txrx_dx = txrx_dx, txrx_dy = txrx_dy, txrx_dz = txrx_dz)
        ng = 28

    #print(tx_height)

    D = np.zeros((nd,ng))

    # Compute forward data
    t1=time.time()
    for i in tqdm(range(nd), mininterval=1, disable=disableTqdm, desc='gatdaem1d', leave=False):
        if C.ndim==1:
            # Only one model
            conductivity = C
        else:
            conductivity = C[i]

        # Update geomtery, tx_heigght is changing!
        if len(tx_height)>1:
            if (showInfo>1):
                print('Using tx_height=%f' % tx_height[i])
            G = Geometry(tx_height=tx_height[i], txrx_dx = txrx_dx, txrx_dy = txrx_dy, txrx_dz = txrx_dz)
    
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

def forward_gaaem_chunk(C_chunk, tx_height_chunk, thickness, stmfiles, file_gex, Nhank, Nfreq, **kwargs):
    """
    Perform forward modeling using the GAAEM method on a chunk of data.

    :param C_chunk: The chunk of data to be processed.
    :type C_chunk: numpy.ndarray
    :param tx_height_chunk: The transmitter heights for this chunk.
    :type tx_height_chunk: numpy.ndarray 
    :param thickness: The thickness of the model.
    :type thickness: float
    :param stmfiles: A list of STM files.
    :type stmfiles: list
    :param file_gex: The path to the GEX file.
    :type file_gex: str
    :param Nhank: The number of Hankel functions.
    :type Nhank: int
    :param Nfreq: The number of frequencies.
    :type Nfreq: int
    :param kwargs: Additional keyword arguments.
    :type kwargs: dict

    :return: The result of the forward modeling.
    :rtype: numpy.ndarray
    """
    return forward_gaaem(C=C_chunk, 
                        thickness=thickness, 
                        tx_height=tx_height_chunk,
                        stmfiles=stmfiles, 
                        file_gex=file_gex, 
                        Nhank=Nhank, 
                        Nfreq=Nfreq, 
                        parallel=False, 
                        **kwargs)

# %% PRIOR DATA GENERATORS

# Add this function to check current handle count (Windows only)
def get_process_handle_count():
    """Return the number of handles used by the current process (Windows only)"""
    import psutil
    import os
    return psutil.Process(os.getpid()).num_handles()

def prior_data_gaaem(f_prior_h5, file_gex, N=0, doMakePriorCopy=True, im=1, id=1, im_height=0, Nhank=280, Nfreq=12, is_log=False, parallel=True, **kwargs):
    """
    Generate prior data for the ga-aem method.

    :param f_prior_h5: Path to the prior data file in HDF5 format.
    :type f_prior_h5: str
    :param file_gex: Path to the file containing geophysical exploration data.
    :type file_gex: str
    :param N: Number of soundings to consider (default: 0).
    :type N: int
    :param doMakePriorCopy: Flag indicating whether to make a copy of the prior file (default: True).
    :type doMakePriorCopy: bool
    :param im: Index of the model (default: 1).
    :type im: int
    :param id: Index of the data (default: 1).
    :type id: int
    :param Nhank: Number of Hankel transform quadrature points (default: 280).
    :type Nhank: int
    :param Nfreq: Number of frequencies (default: 12).
    :type Nfreq: int
    :param parallel: Flag indicating whether multiprocessing is used (default: True).
    :type parallel: bool
    :param Ncpu: Number of cpus/threads used (default: 0 - all).
    :type Ncpu: int
    :param im_height: Index of the model for height (default: 0).
    :type im_height: int
    :param kwargs: Additional keyword arguments.
    :type kwargs: dict
    :param Ncpu: Number of CPUs to use (default: 0->all).
    :type Ncpu: int
    
    :return: Filename of the HDF5 file containing the updated prior data.
    :rtype: str
    """
    import integrate as ig
    import os 
    type = 'TDEM'
    method = 'ga-aem'
    showInfo = kwargs.get('showInfo', 0)
    Ncpu = kwargs.get('Ncpu', 0)
    # of 'Nproc' is set in kwargs use it 
    Ncpu = kwargs.get('Nproc', Ncpu)

    if showInfo>0:
        print('prior_data_gaaem: %s/%s -- starting' % (type, method))

    # Force open/close of hdf5 file
    if showInfo>0:
        print('Forcing open and close of %s' % (f_prior_h5))
    with h5py.File(f_prior_h5, 'r') as f:
        # open and close
        pass

    with h5py.File(f_prior_h5, 'r') as f:
        N_in = f['M1'].shape[0]
    if N==0: 
        N = N_in     
    if N>N_in:
        N=N_in

    if not os.path.isfile(file_gex):
        print("ERRROR: file_gex=%s does not exist in the current folder." % file_gex)

    if doMakePriorCopy:
        file_gex_basename = os.path.splitext(os.path.basename(file_gex))[0]

        if N < N_in:
            f_prior_data_h5 = '%s_%s_N%d_Nh%d_Nf%d.h5' % (os.path.splitext(f_prior_h5)[0], os.path.splitext(file_gex_basename)[0], N, Nhank, Nfreq)
        else:
            f_prior_data_h5 = '%s_%s_Nh%d_Nf%d.h5' % (os.path.splitext(f_prior_h5)[0], os.path.splitext(file_gex_basename)[0], Nhank, Nfreq)
            
        
        if (showInfo>-1):
            print("Creating a copy of %s" % (f_prior_h5))
            print("                as %s" % (f_prior_data_h5))
        if (showInfo>0):
                print('  using N=%d of N_in=%d data' % (N,N_in))
        
        # make a copy of the prior file
        ig.copy_hdf5_file(f_prior_h5, f_prior_data_h5,N,showInfo=showInfo)
            
    else:
        f_prior_data_h5 = f_prior_h5

    
    Mname = '/M%d' % im
    Mheight = '/M%d' % im_height
    Dname = '/D%d' % id


    f_prior = h5py.File(f_prior_data_h5, 'a')

    if im_height>0:    
        if (showInfo>1):
            print('Using M%d for height' % im_height)
        tx_height = f_prior[Mheight][:]

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
        if (showInfo>-1):
            print("prior_data_gaaem: Using 1 thread /(sequential).")
        # Sequential
        if im_height>0:
            print('Using tx_height')
            D = ig.forward_gaaem(C=C, thickness=thickness, tx_height=tx_height, file_gex=file_gex, Nhank=Nhank, Nfreq=Nfreq, parallel=parallel, **kwargs)
        else:
            D = ig.forward_gaaem(C=C, thickness=thickness, file_gex=file_gex, Nhank=Nhank, Nfreq=Nfreq, parallel=parallel, **kwargs)
        if is_log:
            D = np.log10(D)
    else:

        # Make sure STM files are only written once!!! (need for multihreading)
        # D = ig.forward_gaaem(C=C[0:1,:], thickness=thickness, file_gex=file_gex, Nhank=Nhank, Nfreq=Nfreq, parallel=False, **kwargs)
        stmfiles, GEX = ig.gex_to_stm(file_gex, Nhank=Nhank, Nfreq=Nfreq, **kwargs)

        # Parallel
        if Ncpu < 1 :
            #Ncpu =  int(multiprocessing.cpu_count()/2)
            Ncpu =  int(multiprocessing.cpu_count())
        if (showInfo>-1):
            print("prior_data_gaaem: Using %d parallel threads." % (Ncpu))

        # 1: Define a function to compute a chunk
        ## OUTSIDE
        # 2: Create chunks
        C_chunks = np.array_split(C, Ncpu)
        
        if im_height>0:    
            tx_height_chunks = np.array_split(tx_height, Ncpu)
            
        else:
            # create tx_height_chunks as a list of length Ncpu, where each entry is tx_height=np.array(())
            tx_height_chunks = [np.array(())]*Ncpu


        import os
        if os.name == 'nt':  # Windows
            # Log handle count before creating pool
            handle_count_before = get_process_handle_count()
            print(f"Handle count before pool: {handle_count_before}")

        # 3: Compute the chunks in parallel
        forward_gaaem_chunk_partial = partial(forward_gaaem_chunk, thickness=thickness, stmfiles=stmfiles, file_gex=file_gex, Nhank=Nhank, Nfreq=Nfreq, **kwargs)

        # Use spawn context on Windows for better handle management
        if os.name == 'nt':
            ctx = multiprocessing.get_context("spawn")
            Ncpu = min(Ncpu, 60)  # Set to a safe limit below 63
            with ctx.Pool(processes=Ncpu) as p:
                D_chunks = p.starmap(forward_gaaem_chunk_partial, zip(C_chunks, tx_height_chunks))
        else:
            # On non-Windows platforms, use regular Pool
            with Pool(processes=Ncpu) as p:
                D_chunks = p.starmap(forward_gaaem_chunk_partial, zip(C_chunks, tx_height_chunks))

  
        D = np.concatenate(D_chunks)
        #print("D.shape", D.shape)
        if is_log:
            D = np.log10(D)

        if os.name == 'nt' and 'get_process_handle_count' in globals():
            # Log handle count after pool is closed
            handle_count_after = get_process_handle_count()
            print(f"Handle count after pool: {handle_count_after}")


        # D = ig.forward_gaaem(C=C, thickness=thickness, file_gex=file_gex, Nhank=Nhank, Nfreq=Nfreq, parallel=parallel, **kwargs)

    t2 = time.time()
    t_elapsed = t2 - t1
    if (showInfo>-1):
        print('prior_data_gaaem: Time=%5.1fs/%d soundings. %4.1fms/sounding, %3.1fit/s' % (t_elapsed, N, 1000*t_elapsed/N,N/t_elapsed))
    
    # Write D to f_prior['/D1']
    f_prior[Dname] = D

    # Add method, type, file_ex, and im as attributes to '/D1'
    f_prior[Dname].attrs['method'] = method
    f_prior[Dname].attrs['type'] = type
    f_prior[Dname].attrs['im'] = im
    f_prior[Dname].attrs['Nhank'] = Nhank
    f_prior[Dname].attrs['Nfreq'] = Nfreq

    f_prior.close()

    integrate_update_prior_attributes(f_prior_data_h5)
    
    return f_prior_data_h5


def prior_data_identity(f_prior_h5, id=0, im=1, N=0, doMakePriorCopy=False, **kwargs):
    '''
    Generate data D%id from model M%im in the prior file f_prior_h5 as an identity of M%im.

    :param f_prior_h5: Path to the prior data file in HDF5 format.
    :type f_prior_h5: str
    :param id: Index of the data (default: 0). if id=0, the next available data id is used
    :type id: int
    :param im: Index of the model (default: 1).
    :type im: int
    :param N: Number of soundings to consider (default: 0).
    :type N: int
    :param doMakePriorCopy: Flag indicating whether to make a copy of the prior file (default: False).
    :type doMakePriorCopy: bool
    :param kwargs: Additional keyword arguments.
    :type kwargs: dict

    '''
    import integrate as ig
    import time
    
    type = 'idenity'
    method = '--'
    showInfo = kwargs.get('showInfo', 0)
    forceDeleteExisting = kwargs.get('forceDeleteExisting', True)


    # check keys for the data with max id form 'D1', 'D2', 'D3', ...
    if id==0:
        with h5py.File(f_prior_h5, 'a') as f_prior:
            id = 1
            for id_test in range(15):
                key = '/D%d' % id_test
                if key in f_prior.keys():
                    print('Checking key EXISTS: %s' % key)
                    id = id_test+1
                else:                    
                    pass
            print('using id = %d' % id)    
        
    
    with h5py.File(f_prior_h5, 'a') as f:
        N_in = f['M1'].shape[0]
    if N==0: 
        N = N_in     
    if N>N_in:
        N=N_in

    print('N=%d, N_in=%d' % (N,N_in))
    if doMakePriorCopy:
        if N < N_in:
            f_prior_data_h5 = '%s_N%s_IDEN_im%d_id%d.h5' % (os.path.splitext(f_prior_h5)[0], N, im, id)
        else:
            f_prior_data_h5 = '%s_IDEN_im%d_id%d.h5' % (os.path.splitext(f_prior_h5)[0], im, id)
        if (showInfo>0):
            print("Creating a copy of %s as %s" % (f_prior_h5, f_prior_data_h5))
        ig.copy_hdf5_file(f_prior_h5, f_prior_data_h5,N)
        
    else:
        f_prior_data_h5 = f_prior_h5

    Mname = '/M%d' % im
    Dname = '/D%d' % id

    # copy f_prior[Mname] to Dname
    print('Copying %s to %s in filename=%s' % (Mname, Dname, f_prior_data_h5))

    # f_prior = h5py.File(f_prior_data_h5, 'r+')
    with h5py.File(f_prior_data_h5, 'a') as f:
        D = f[Mname]
        # check if Dname exists, if so, delete it
        if Dname in f.keys():
            if forceDeleteExisting:
                print('Key %s allready exists -- DELETING !!!!' % Dname)
                del f[Dname]
            else:
                print('Key %s allready exists - doing nothing' % Dname)
                return f_prior_data_h5
        
        dataset = f.create_dataset(Dname, data=D)  # 'i4' represents 32-bit integers
        dataset.attrs['description'] = 'Identiy of %s' % Mname
        dataset.attrs['f5_forward'] = 'none'
        dataset.attrs['with_noise'] = 0
        #f_prior.close()
    
    
    return f_prior_data_h5

# %% PRIOR MODEL GENERATORS
def prior_model_layered(lay_dist='uniform', dz = 1, z_max = 90, 
                        NLAY_min=3, NLAY_max=6, NLAY_deg=6, 
                        RHO_dist='log-uniform', RHO_min=0.1, RHO_max=100, RHO_MEAN=100, RHO_std=80, 
                        N=100000, **kwargs):
    """
    Generate a prior model with layered structure.

    :param lay_dist: Distribution of the number of layers. Options are 'chi2' and 'uniform'. Default is 'chi2'.
    :type lay_dist: str
    :param NLAY_min: Minimum number of layers. Default is 3.
    :type NLAY_min: int
    :param NLAY_max: Maximum number of layers. Default is 6.
    :type NLAY_max: int
    :param NLAY_deg: Degrees of freedom for chi-square distribution. Only applicable if lay_dist is 'chi2'. Default is 6.
    :type NLAY_deg: int
    :param RHO_dist: Distribution of resistivity within each layer. Options are 'log-uniform', 'uniform', 'normal', and 'lognormal'. Default is 'log-uniform'.
    :type RHO_dist: str
    :param RHO_min: Minimum resistivity value. Default is 0.1.
    :type RHO_min: float
    :param RHO_max: Maximum resistivity value. Default is 100.
    :type RHO_max: float
    :param RHO_MEAN: Mean resistivity value. Only applicable if RHO_dist is 'normal' or 'lognormal'. Default is 100.
    :type RHO_MEAN: float
    :param RHO_std: Standard deviation of resistivity value. Only applicable if RHO_dist is 'normal' or 'lognormal'. Default is 80.
    :type RHO_std: float
    :param N: Number of prior models to generate. Default is 100000.
    :type N: int
    :param f_prior_h5: Path to the prior model file in HDF5 format. Default is ''.
    :type f_prior_h5: str

    :return: Filepath of the saved prior model.
    :rtype: str
    """
    
    from tqdm import tqdm
    import integrate as ig

    showInfo = kwargs.get('showInfo', 0)
    f_prior_h5 = kwargs.get('f_prior_h5', '')
 
    if NLAY_max < NLAY_min:
        #raise ValueError('NLAY_max must be greater than or equal to NLAY_min.')
        NLAY_max = NLAY_min

    if NLAY_min < 1:
        #raise ValueError('NLAY_min must be greater than or equal to 1.')
        NLAY_min = 1

    if lay_dist == 'uniform':
        NLAY = np.random.randint(NLAY_min, NLAY_max+1, N)
        if len(f_prior_h5)<1:
            f_prior_h5 = 'PRIOR_UNIFORM_NL_%d-%d_%s_N%d.h5' % (NLAY_min, NLAY_max, RHO_dist, N)

    elif lay_dist == 'chi2':
        NLAY = np.random.chisquare(NLAY_deg, N)
        NLAY = np.ceil(NLAY).astype(int)    
        if len(f_prior_h5)<1:
            f_prior_h5 = 'PRIOR_CHI2_NF_%d_%s_N%d.h5' % (NLAY_deg, RHO_dist, N)

    # Force NLAY to be a 2 dimensional numpy array
    NLAY = NLAY[:, np.newaxis]
    
    z_min = 0
    z = np.arange(z_min, z_max, dz)
    nz= len(z)
    M_rho = np.zeros((N, nz))

    #% simulate the number of layers as in integer
    for i in tqdm(range(N), mininterval=1, disable=(showInfo<0), desc='prior_layered', leave=False):
    
        i_boundaries = np.sort(np.random.choice(nz, NLAY[i]-1, replace=False))        

        ### simulate the resistivity in each layer
        if RHO_dist=='log-normal':
            rho_all=np.random.lognormal(mean=np.log10(RHO_MEAN), sigma=np.log10(RHO_std), size=NLAY[i])
        elif RHO_dist=='normal':
            rho_all=np.random.normal(mean=RHO_MEAN, sigma=RHO_std, size=NLAY[i])
        elif RHO_dist=='log-uniform':
            rho_all=np.exp(np.random.uniform(np.log(RHO_min), np.log(RHO_max), NLAY[i]))
        elif RHO_dist=='uniform':
            rho_all=np.random.uniform(RHO_min, RHO_max, NLAY[i])

        rho = np.zeros(nz)+rho_all[0]
        for j in range(len(i_boundaries)):
            rho[i_boundaries[j]:] = rho_all[j+1]

        M_rho[i]=rho        

    if (showInfo>0):
        print("prior_model_layered: Saving prior model to %s" % f_prior_h5)
    
    # save to hdf5 file
    
    if (showInfo>1):
        print("Saving '/M1' prior model  %s" % f_prior_h5)

    ig.save_prior_model(f_prior_h5,M_rho.astype(np.float32),
                im=1,
                name='resistivity',
                is_discrete = 0, 
                x = z,
                z = z,
                delete_if_exist = True,
                force_replace=True,
                showInfo=showInfo,
                )

    if (showInfo>1):
        print("Saving '/M2' prior model  %s" % f_prior_h5)
    ig.save_prior_model(f_prior_h5,NLAY.astype(np.float32),
                        im=2,
                        name = 'Number of layers',
                        is_discrete=0, 
                        x=np.array([0]), 
                        z=np.array([0]),
                        force_replace=True, 
                        showInfo=showInfo,
                )

    return f_prior_h5

def prior_model_workbench_direct(N=100000, RHO_dist='log-uniform', z1=0, z_max= 100, 
                          nlayers=0, p=2, NLAY_min=3, NLAY_max=6,
                          RHO_min = 1, RHO_max= 300, RHO_mean=180, RHO_std=80, chi2_deg= 100, **kwargs):
    """
    Generate a prior model with increasingly thick layers.
    ALl models have the same number of layers!
    See also: prior_model_workbench
 
    :param N: Number of prior models to generate. Default is 100000.
    :type N: int
    :param RHO_dist: Distribution of resistivity within each layer. Options are 'log-uniform', 'uniform', 'normal', 'lognormal', and 'chi2'. Default is 'log-uniform'.
    :type RHO_dist: str
    :param z1: Minimum depth value. Default is 0.
    :type z1: float
    :param z2: Maximum depth value. Default is 100.
    :type z2: float
    :param nlayers: Number of layers. Default is 30.
    :type nlayers: int
    :param p: Power parameter for thickness increase. Default is 2.
    :type p: int
    :param RHO_min: Minimum resistivity value. Default is 1.
    :type RHO_min: float
    :param RHO_max: Maximum resistivity value. Default is 300.
    :type RHO_max: float
    :param RHO_mean: Mean resistivity value. Only applicable if RHO_dist is 'normal' or 'lognormal'. Default is 180.
    :type RHO_mean: float
    :param RHO_std: Standard deviation of resistivity value. Only applicable if RHO_dist is 'normal' or 'lognormal'. Default is 80.
    :type RHO_std: float
    :param chi2_deg: Degrees of freedom for chi2 distribution. Only applicable if RHO_dist is 'chi2'. Default is 100.
    :type chi2_deg: int
    :param f_prior_h5: Path to the prior model file in HDF5 format. Default is ''.
    :type f_prior_h5: str

    :return: Filepath of the saved prior model.
    :rtype: str
    """

    import integrate as ig

    showInfo = kwargs.get('showInfo', 0)
    f_prior_h5 = kwargs.get('f_prior_h5', '')

    if nlayers<1:
        nlayers = 30
    
    z2=z_max
    z= z1 + (z2 - z1) * np.linspace(0, 1, nlayers) ** p

    nz = len(z)
    
    if RHO_dist=='uniform':
        M_rho = np.random.uniform(low=RHO_min, high = RHO_max, size=(N, nz))
        if len(f_prior_h5)<1:
            f_prior_h5 = '%s_R%g_%g.h5' % (f_prior_h5, RHO_min, RHO_max)
    elif RHO_dist=='log-uniform':
        M_rho = np.exp(np.random.uniform(low=np.log(RHO_min), high = np.log(RHO_max), size=(N, nz)))
        if len(f_prior_h5)<1:
            f_prior_h5 = '%s_R%g_%g.h5' % (f_prior_h5, RHO_min, RHO_max)
    elif RHO_dist=='normal':
        M_rho = np.random.normal(loc=RHO_mean, scale = RHO_std, size=(N, nz))
        if len(f_prior_h5)<1:
            f_prior_h5 = '%s_R%g_%g.h5' % (f_prior_h5, RHO_mean, RHO_std)
    elif RHO_dist=='log-normal':
        M_rho = np.random.lognormal(mean=np.log(RHO_mean), sigma = RHO_std/RHO_mean, size=(N, nz))
        if len(f_prior_h5)<1:
            f_prior_h5 = '%s_R%g_%g.h5' % (f_prior_h5, RHO_mean, RHO_std)
    elif RHO_dist=='chi2':
        M_rho = np.random.chisquare(df = chi2_deg, size=(N, nz))
        if len(f_prior_h5)<1:
            f_prior_h5 = '%s_deg%d.h5' % (f_prior_h5,chi2_deg)
    else:
        raise ValueError('RHO_dist=%s not supported' % RHO_dist)
    
    
    if (showInfo>0):
        print("prior_model_workbench_direct: Saving prior model to %s" % f_prior_h5)


    if (showInfo>1):
        print("Saving '/M1' prior model  %s" % f_prior_h5)
    ig.save_prior_model(f_prior_h5,M_rho.astype(np.float32),
                im=1,
                name='Resistivity',
                is_discrete = 0, 
                x = z,
                z = z,
                delete_if_exist = True,
                force_replace=True,
                showInfo=showInfo,
                )


    return f_prior_h5


def prior_model_workbench(N=100000, p=2, z1=0, z_max= 100, dz=1,
                          lay_dist='uniform', nlayers=0, NLAY_min=3, NLAY_max=6, NLAY_deg=5,
                          RHO_dist='log-uniform', 
                          RHO_min = 1, RHO_max= 300, RHO_mean=180, RHO_std=80, chi2_deg= 100, **kwargs):
    """
    Generate a prior model with increasingly thick layers
 
    :param lay_dist: Distribution of the number of layers. Options are 'chi2' and 'uniform'. Default is 'chi2'.
    :type lay_dist: str:param N: Number of prior models to generate. Default is 100000.
    :type N: int
    :param RHO_dist: Distribution of resistivity within each layer. Options are 'log-uniform', 'uniform', 'normal', 'lognormal', and 'chi2'. Default is 'log-uniform'.
    :type RHO_dist: str
    :param z1: Minimum depth value. Default is 0.
    :type z1: float
    :param z2: Maximum depth value. Default is 100.
    :type z2: float
    :param nlayers: Number of layers. Default is 30.
    :type nlayers: int
    :param NLAY_min: Minimum number of layers. Default is 3.
    :type NLAY_min: int
    :param NLAY_max: Maximum number of layers. Default is 6.
    :type NLAY_max: int
    :param NLAY_deg: Degrees of freedom for chi-square distribution. Only applicable if lay_dist is 'chi2'. Default is 6.
    :type NLAY_deg: int
    :param p: Power parameter for thickness increase. Default is 2.
    :type p: int
    :param RHO_min: Minimum resistivity value. Default is 1.
    :type RHO_min: float
    :param RHO_max: Maximum resistivity value. Default is 300.
    :type RHO_max: float
    :param RHO_mean: Mean resistivity value. Only applicable if RHO_dist is 'normal' or 'lognormal'. Default is 180.
    :type RHO_mean: float
    :param RHO_std: Standard deviation of resistivity value. Only applicable if RHO_dist is 'normal' or 'lognormal'. Default is 80.
    :type RHO_std: float
    :param chi2_deg: Degrees of freedom for chi2 distribution. Only applicable if RHO_dist is 'chi2'. Default is 100.
    :type chi2_deg: int
    :param f_prior_h5: Path to the prior model file in HDF5 format. Default is ''.
    :type f_prior_h5: str    

    :return: Filepath of the saved prior model.
    :rtype: str
    """
    from tqdm import tqdm
    import integrate as ig

    f_prior_h5 = kwargs.get('f_prior_h5', '')
    showInfo = kwargs.get('showInfo', 0)
    if nlayers>0:
        NLAY_min = nlayers
        NLAY_max = nlayers

    if NLAY_max < NLAY_min:
        #raise ValueError('NLAY_max must be greater than or equal to NLAY_min.')
        NLAY_max = NLAY_min

    if NLAY_min < 1:
        #raise ValueError('NLAY_min must be greater than or equal to 1.')
        NLAY_min = 1

    
    if lay_dist == 'chi2':
        NLAY = np.random.chisquare(NLAY_deg, N)
        NLAY = np.ceil(NLAY).astype(int)    
        if len(f_prior_h5)<1:
            f_prior_h5 = 'PRIOR_WB_CHI2_NF_%d_%s_N%d.h5' % (NLAY_deg, RHO_dist, N)
    elif lay_dist == 'uniform':
        NLAY = np.random.randint(NLAY_min, NLAY_max+1, N)
        if NLAY_min == NLAY_max:
            nlayers = NLAY_min
            if len(f_prior_h5)<1:
                f_prior_h5 = 'PRIOR_WB_UNIFORM_%d_N%d_%s' % (nlayers,N,RHO_dist)
        else:   
            if len(f_prior_h5)<1:
                f_prior_h5 = 'PROPR_WB_UNIFORM_%d-%d_N%d_%s' % (NLAY_min,NLAY_max,N,RHO_dist)
        


    # Force NLAY to be a 2 dimensional numpy array (for when exporting to HDF5)
    NLAY = NLAY[:, np.newaxis]
    
    z_min = 0
    z = np.arange(z_min, z_max, dz)
    nz= len(z)
    if showInfo>1:
        print('z_min, z_max, dz, nz = %g, %g, %g, %d' % (z_min, z_max, dz, nz))
    M_rho = np.zeros((N, nz))

    for i in tqdm(range(N), mininterval=1, disable=(showInfo<0), desc='prior_workbench', leave=False):
        nlayers = NLAY[i][0]
        #print(nlayers)
        z2=z_max
        z_single= z1 + (z2 - z1) * np.linspace(0, 1, nlayers) ** p

        if RHO_dist=='uniform':
            M_rho_single = np.random.uniform(low=RHO_min, high = RHO_max, size=(1, nz))
        elif RHO_dist=='log-uniform':
            M_rho_single = np.exp(np.random.uniform(low=np.log(RHO_min), high = np.log(RHO_max), size=(1, nz)))
        elif RHO_dist=='normal':
            M_rho_single = np.random.normal(loc=RHO_mean, scale = RHO_std, size=(1, nz))
        elif RHO_dist=='log-normal':
            M_rho_single = np.random.lognormal(mean=np.log(RHO_mean), sigma = RHO_std/RHO_mean, size=(1, nz))
        elif RHO_dist=='chi2':
            M_rho_single = np.random.chisquare(df = chi2_deg, size=(1, nz))

        for j in range(nlayers):
            ind = np.where(z>=z_single[j])[0]
            M_rho[i,ind]= M_rho_single[0,j]

    if (showInfo>0):
        print("prior_model_workbench: Saving prior model to %s" % f_prior_h5)

    if (showInfo>1):
        print("Saving '/M1' prior model  %s" % f_prior_h5)
    ig.save_prior_model(f_prior_h5,M_rho.astype(np.float32),
                im=1,
                name='Resistivity',
                is_discrete = 0, 
                x = z,
                z = z,
                delete_if_exist = True,
                force_replace=True,
                showInfo=showInfo,
                )

    if (showInfo>1):
        print("Saving '/M2' prior model  %s" % f_prior_h5)
    ig.save_prior_model(f_prior_h5,NLAY.astype(np.float32),
                        im=2,
                        name = 'Number of layers',
                        is_discrete=0, 
                        x=np.array([0]), 
                        z=np.array([0]),
                        force_replace=True, 
                        showInfo=showInfo,
                )

    # return the full filepath to f_prior_h5
    return f_prior_h5



def posterior_cumulative_thickness(f_post_h5, im=2, icat=[0], usePrior=False, **kwargs):
    """
    Calculate the posterior cumulative thickness based on the given inputs.

    :param f_post_h5: Path to the input h5 file.
    :type f_post_h5: str
    :param im: Index of model parameter number, M[im].
    :type im: int
    :param icat: List of category indices.
    :type icat: list
    :param usePrior: Flag indicating whether to use prior.
    :type usePrior: bool
    :param kwargs: Additional keyword arguments.
    :returns: 
        - thick_mean (ndarray): Array of mean cumulative thickness.
        - thick_median (ndarray): Array of median cumulative thickness.
        - thick_std (ndarray): Array of standard deviation of cumulative thickness.
        - class_out (list): List of class names.
        - X (ndarray): Array of X values.
        - Y (ndarray): Array of Y values.
    :rtype: tuple
    """

    import h5py
    import integrate as ig

    if isinstance(icat, int):
        icat = np.array([icat])

    with h5py.File(f_post_h5,'r') as f_post:
        f_prior_h5 = f_post['/'].attrs['f5_prior']
        f_data_h5 = f_post['/'].attrs['f5_data']

    X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)

    Mstr = '/M%d' % im
    with h5py.File(f_prior_h5,'r') as f_prior:
        if not Mstr in f_prior.keys():
            print('No %s found in %s' % (Mstr, f_prior_h5))
            return -1
        if not f_prior[Mstr].attrs['is_discrete']:
            print('M%d is not discrete' % im)
            return -1



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

    with h5py.File(f_post_h5,'r') as f_post:
        #P=f_post[Mstr+'/P'][:]
        i_use = f_post['/i_use'][:]

    ns,nr=i_use.shape

    if usePrior:
        for i in range(ns):
            i_use[i,:]=np.arange(nr)
        

    f_prior = h5py.File(f_prior_h5,'r')
    M_prior = f_prior[Mstr][:]
    f_prior.close()
    nz = M_prior.shape[1]

    thick_mean = np.zeros((ns))
    thick_median = np.zeros((ns))
    thick_std = np.zeros((ns))


    thick = np.diff(z)

    for i in range(ns):

        jj = i_use[i,:].astype(int)-1
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

    return thick_mean, thick_median, thick_std, class_out, X, Y


def create_shared_memory(arrays):
    """Create shared memory segments for arrays."""
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
    """Reconstruct arrays from shared memory references."""
    reconstructed_arrays = []
    for shm_name, shape, dtype in shared_memory_refs:
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
            array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            reconstructed_arrays.append(array.copy())
            shm.close()
        except Exception as e:
            logger.error(f"Error reconstructing array: {e}")
            raise
    return reconstructed_arrays

def cleanup_shared_memory(shm_objects):
    """Clean up shared memory segments."""
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


# # Create shared memory arrays
# def create_shared_memory(arrays):
#     shared_memories = []
#     for array in arrays:
#         shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
#         # Copy the data into shared memory
#         shared_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
#         shared_array[:] = array[:]
#         # Store the shared memory name, array shape, and dtype
#         shared_memories.append((shm.name, array.shape, array.dtype))
#     return shared_memories

# # Function to reconstruct the list D from shared memory
# def reconstruct_shared_arrays(shared_memory_refs):
#     reconstructed_arrays = []
#     for shm_name, shape, dtype in shared_memory_refs:
#         shm = shared_memory.SharedMemory(name=shm_name)
#         array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
#         reconstructed_arrays.append(array.copy())  # Copy the data to avoid leaving a view
#         shm.close()
#     return reconstructed_arrays

# # Cleanup function for shared memory
# def cleanup_shared_memory(shared_memory_refs):
#     for shm_name, _, _ in shared_memory_refs:
#         shm = shared_memory.SharedMemory(name=shm_name)
#         shm.close()
#         shm.unlink()

# START OF REJECTION SAMPLING

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
    from datetime import datetime   
    #from multiprocessing import Pool
    #import multiprocessing
    import integrate as ig
    #import numpy as np
    #import h5py

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
        print('integrate_rejection: Time=%5.1fs/%d soundings, %4.1fms/sounding, %3.1fit/s' % (t_elapsed,Ndp_invert,t_per_sounding*1000,Ndp_invert/t_elapsed))
    if (showInfo>-1):
        print('integrate_rejection: T_av=%3.1f' % (np.nanmean(T_all)))
        print('integrate_rejection: EV_av=%3.1f' % (np.nanmean(EV_all)))

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
        integrate_posterior_stats(f_post_h5, **kwargs)

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
    

    from tqdm import tqdm
    import numpy as np
    import h5py
    import time
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
    #import integrate as ig
    from multiprocessing import Pool, shared_memory

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
    #import integrate as ig
    
    # New implementation with shared memory
    i_chunk, ip_chunks, DATA, idx, N_use, id_use, shared_memory_refs, autoT, T_base, nr, use_N_best = args
    # Old implementation where D was copied to each process
    #i_chunk, ip_chunks, D, DATA, idx, N_use, id_use, shared_memory_refs, autoT, T_base, nr, use_N_best = args
    #D=reconstruct_shared_arrays(shared_memory_refs)
    D=reconstruct_shared_arrays(shared_memory_refs)

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


# END OF REJECTION SAMPLING


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
    This function calculates the likelihood of observed data `d_obs` given 
    a set of predicted data `D` and standard deviations `d_std` assuming 
    a Gaussian distribution with a diagonal covariance matrix.
    Parameters
    ----------
    D : numpy.ndarray
        Predicted data array of shape (n_samples, n_features).
    d_obs : numpy.ndarray
        Observed data array of shape (n_features,).
    d_std : numpy.ndarray
        Standard deviation array of shape (n_features,).
    Returns
    -------
    numpy.ndarray
        Likelihood array of shape (n_samples,).
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
    Calculate the Gaussian likelihood for a given dataset.
    Parameters
    ----------
    D : numpy.ndarray
        The model predictions, with shape (n_samples, n_features).
    d_obs : numpy.ndarray
        The observed data, with shape (n_features,).
    Cd : numpy.ndarray
        The covariance matrix of the observed data, with shape (n_features, n_features).
    checkNaN : bool, optional
        If True, the function will handle NaN values in `d_obs` by ignoring them in the calculations. 
        Default is True.
    Returns
    -------
    numpy.ndarray
        The Gaussian likelihood for each sample, with shape (n_samples,).
    TODO
        We need to check that this works when D has NAN value.. (and Why does it ever?)
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
    This function computes the log-likelihood of multinomial distribution for given 
    discrete data using direct indexing and optimized array operations.
    Parameters
    ----------
    D : numpy.ndarray
        Matrix of observed discrete data with shape (N, n_features), where N is the
        number of samples and each element represents a class ID.
    P_obs : numpy.ndarray
        Matrix of probabilities with shape (n_classes, n_features), where each column
        represents probability distribution over classes for a feature.
    class_id : numpy.ndarray
        Array of unique class IDs corresponding to rows in P_obs.
        If class=None, then the id class id's are extracted from the unique values in D.
    class_is_idx=False, optional
        If True, class_id is already an index. 
        Default is False, and then the id is computed from the class_id array.
    entropyFilter : bool, optional
        If True, applies entropy filtering to select features. Default is False.
    entropyThreshold : float, optional
        Threshold value for entropy filtering, features with entropy below this value
        are selected. Default is 0.99.
        Log-likelihood values for each sample, shape (N,).        
    Returns
    -------
    numpy.ndarray
        Log-likelihood values for each sample, shape (N, 1).
    Notes
    -----
    The function:
        1. Creates a mapping from class IDs to indices
        2. Converts test data to corresponding indices
        3. Retrieves probabilities using advanced indexing
        4. Calculates log likelihood using max normalization for numerical stability

    """
    
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



# %% Synthetic data

def synthetic_case(case='Wedge', **kwargs):
    """
    Generate synthetic geological models for different cases.
    Parameters
    ----------
    case : str, optional
        The type of synthetic case to generate. Options are 'Wedge' and '3Layer'. Default is 'Wedge'.
    **kwargs : dict, optional
        Additional parameters for the synthetic case generation.
        Common parameters:
        - showInfo : int, optional
            If greater than 0, print information about the generated case. Default is 0.
        Parameters for 'Wedge' case:
        - x_max : int, optional
            Maximum x-dimension size. Default is 1000.
        - dx : float, optional
            Step size in the x-dimension. Default is 1000./x_max.
        - z_max : int, optional
            Maximum z-dimension size. Default is 90.
        - dz : float, optional
            Step size in the z-dimension. Default is 1.
        - z1 : float, optional
            Depth at which the wedge starts. Default is z_max/10.
        - rho : list of float, optional
            Density values for different layers. Default is [100, 200, 120].
        - wedge_angle : float, optional
            Angle of the wedge in degrees. Default is 1.
        Parameters for '3Layer' case:
        - x_max : int, optional
            Maximum x-dimension size. Default is 100.
        - x_range : float, optional
            Range in the x-dimension for the cosine function. Default is x_max/4.
        - dx : float, optional
            Step size in the x-dimension. Default is 1.
        - z_max : int, optional
            Maximum z-dimension size. Default is 60.
        - dz : float, optional
            Step size in the z-dimension. Default is 1.
        - z1 : float, optional
            Depth at which the first layer ends. Default is z_max/3.
        - z_thick : float, optional
            Thickness of the second layer. Default is z_max/2.
        - rho1_1 : float, optional
            Density at the start of the first layer. Default is 120.
        - rho1_2 : float, optional
            Density at the end of the first layer. Default is 10.
        - rho2_1 : float, optional
            Density at the start of the second layer. Default is rho1_2.
        - rho2_2 : float, optional
            Density at the end of the second layer. Default is rho1_1.
        - rho3 : float, optional
            Density of the third layer. Default is 120.
    Returns
    -------
    M : numpy.ndarray
        The generated synthetic model.
    x : numpy.ndarray
        The x-coordinates of the model.
    z : numpy.ndarray
        The z-coordinates of the model.
    """
    
    showInfo = kwargs.get('showInfo', 0)

    if case.lower() == 'wedge':
        # Create synthetic wedhge model
        
        # variables
        x_max = kwargs.get('x_max', 1000)
        dx = kwargs.get('dx', 1000./x_max)
        z_max = kwargs.get('z_max', 90)
        dz = kwargs.get('dz', 1)
        z1 = kwargs.get('z1', z_max/10)
        rho = kwargs.get('rho', [100,200,120])
        wedge_angle = kwargs.get('wedge_angle', 1)

        if showInfo>0:
            print('Creating synthetic %s case with wedge angle=%f' % (case,ẃedge_angle))

        z = np.arange(0,z_max,dz)
        x = np.arange(0,x_max,dx)

        nx = x.shape[0]
        nz = z.shape[0]

        M = np.zeros((nx,nz))+rho[0]
        # set M=rho[3] of all iz> (z==z1)
        iz = np.where(z>=z1)[0]
        M[:,iz] = rho[2]
        for ix in range(nx):
            wedge_angle_rad = np.deg2rad(wedge_angle)
            z2 = z1 + x[ix]*np.tan(wedge_angle_rad)            
            #find iz where  (z>=z1) and (z<=z2)
            iz = np.where((z>=z1) & (z<=z2))[0]
            #print(z[iz[0]])
            M[ix,iz] = rho[1]

        return M, x, z

    elif case.lower() == '3layer':
        # Create synthetic 3 layer model

        # variables
        x_max = kwargs.get('x_max', 100)
        x_range = kwargs.get('x_range', x_max/4)
        dx = kwargs.get('dx', 1)
        z_max = kwargs.get('z_max', 60)
        dz = kwargs.get('dz', 1)
        z1 = kwargs.get('z1', z_max/3)
        z_thick = kwargs.get('z_thick', z_max/2)
        

        rho1_1 = kwargs.get('rho1_1', 120)
        rho1_2 = kwargs.get('rho1_2', 10)
        rho2_1 = kwargs.get('rho1_2', rho1_2)
        rho2_2 = kwargs.get('rho2_2', rho1_1)
        rho3 = kwargs.get('rho3', 120)

        if showInfo>0:
            print('Creating synthetic %s case with wedge angle=%f' % (case,ẃedge_angle))

        z = np.arange(0,z_max,dz)
        x = np.arange(0,x_max,dx)

        nx = x.shape[0]
        nz = z.shape[0]

        M = np.zeros((nx,nz))+rho3
        iz1 = np.where(z<=z1)[0]
        for ix in range(nx):
            rho1 = rho1_1 + (rho1_2 - rho1_1) * x[ix]/x_max
            rho2 = rho2_1 + (rho2_2 - rho2_1) * x[ix]/x_max
            M[ix,iz1] = rho1
            z2 = z1 + z_thick*0.5*(1+np.cos(x[ix]/(x_range)*np.pi))            
            rho2 = rho2_1 + (rho2_2 - rho2_1) * x[ix]/x_max
            iz2 = np.where((z>=z1) & (z<=z2))[0]
            M[ix,iz2] = rho2

        return M, x, z



def get_weight_from_position(f_data_h5,x_well=0,y_well=0, i_ref=-1, r_dis = 400, r_data=2, useLog=True, doPlot=False, plFile=None):
    """Calculate weights based on distance and data similarity to a reference point.

    This function computes three sets of weights:
    1. Combined weights based on both spatial distance and data similarity
    2. Distance-based weights
    3. Data similarity weights

    Parameters
    ----------
    f_data_h5 : str
        Path to HDF5 file containing geometry and observed data
    x_well : float, optional
        X coordinate of reference point (well), by default 0
    y_well : float, optional 
        Y coordinate of reference point (well), by default 0
    i_ref : int, optional
        Index of reference point, by default -1 (auto-calculated as closest to x_well,y_well)
    r_dis : float, optional
        Distance range parameter for spatial weighting, by default 400
    r_data : float, optional
        Data similarity range parameter for data weighting, by default 2

    Returns
    -------
    tuple
        (w_combined, w_dis, w_data) where:
        - w_combined: Combined weights from distance and data similarity
        - w_dis: Distance-based weights
        - w_data: Data similarity-based weights

    Notes
    -----
    The weights are calculated using Gaussian functions:
    - Distance weights use exp(-dis²/r_dis²)
    - Data weights use exp(-sum_dd²/r_data²)
    where dis is spatial distance and sum_dd is cumulative data difference
    """
    import integrate as ig
    import numpy as np
    import matplotlib.pyplot as plt
    X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)
    DATA = ig.load_data(f_data_h5)
    id=0
    d_obs = DATA['d_obs'][id]
    d_std = DATA['d_std'][id]
    # index if position in X and Y with smallets distance to well
    if i_ref == -1:
        i_ref = np.argmin((X-x_well)**2 + (Y-y_well)**2)

    # select gates to use 
    # find the number of data points for each gate that has non-nan values
    n_not_nan = np.sum(~np.isnan(d_obs), axis=0)
    n_not_nan_freq = n_not_nan/d_obs.shape[0]
    # use the data for which n_not_nan_freq>0.8
    i_use = np.where(n_not_nan_freq>0.8)[0]
    # only use i_use values that are not nan
    i_use = i_use[~np.isnan(d_obs[i_ref,i_use])]
    # select gates to use, manually
    if useLog:
        d_ref = np.log10(d_obs[i_ref,i_use])
        d_test = np.log10(d_obs[:,i_use])
    else:
        d_ref = d_obs[i_ref,i_use]
        d_test =d_obs[:,i_use]
    dd = np.abs(d_test - d_ref)
    sum_dd = np.sum(dd, axis=1)

    w_data = np.exp(-1*sum_dd**2/r_data**2)

    # COmpute the distance from d_ref to all other points
    dis = np.sqrt((X-X[i_ref])**2 + (Y-Y[i_ref])**2)
    w_dis = np.exp(-1*dis**2/r_dis**2)

    w_combined = w_data * w_dis

    cmap = 'hot_r'
    #cmap = 'jet'

    if doPlot:
        plt.figure(figsize=(15,5))
        for i in range(3):
            plt.subplot(1,3,i+1)
            plt.plot(X,Y,'.', markersize=1.02, color='lightgray') 
            #plt.scatter(X[i_use], Y[i_use], c=w[i_use], cmap='jet', s=1, zorder=3, vmin=0, vmax=1, marker='.')
                 
            if i==0:
                i_use = np.where(w_combined>0.001)[0]
                plt.scatter(X[i_use],Y[i_use],c=w_combined[i_use], s=1, cmap=cmap, vmin=0, vmax=1, marker='.', zorder=3)  
                plt.title('Combined weights')          
            elif i==1:
                i_use = np.where(w_dis>0.001)[0]                
                plt.scatter(X[i_use],Y[i_use],c=w_dis[i_use], s=1, cmap=cmap, vmin=0, vmax=1, marker='.', zorder=3)
                plt.title('XY distance weights')
            elif i==2:
                i_use = np.where(w_data>0.001)[0]                                
                plt.scatter(X[i_use],Y[i_use],c=w_data[i_use], s=0.2, cmap=cmap, vmin=0, vmax=1, marker='.', zorder=3)
                plt.title('Data distance weights')
            plt.axis('equal')
            plt.colorbar()
            plt.grid()
            plt.plot(x_well,y_well,'wo', zorder=6, markersize=2)
            plt.plot(x_well,y_well,'ko', zorder=5, markersize=4)
            plt.plot(x_well,y_well,'wo', zorder=4, markersize=6)
    

        plt.suptitle('Weights')
        plt.xlabel('X')
        plt.ylabel('Y')
        if plFile is None:
            plFile = 'weights_%d_%d_%d_rdis%d_rdata%d.png' % (x_well,y_well,i_ref,r_dis,r_data)
        plt.savefig(plFile, dpi=300)

    return w_combined, w_dis, w_data, i_ref

####################################
## MISC 

def comb_cprob(pA, pAgB, pAgC, tau=1.0):
    """
    Combine conditional probabilities based on permanence of updating ratios.

    This function implements the probability combination method described in 
    Journel's "An Alternative to Traditional Data Independence Hypotheses" 
    (Math Geology, 2004).

    Parameters:
    -----------
    pA : array_like
        Probability of event A
    pAgB : array_like
        Conditional probability of A given B
    pAgC : array_like
        Conditional probability of A given C
    tau : float, optional
        Combination parameter controlling the ratio permanence (default=1.0)

    Returns:
    --------
    ndarray
        Combined conditional probability Prob(A|B,C)

    References:
    -----------
    Journel, An Alternative to Traditional Data Independence Hypotheses, 
    Mathematical Geology, 2002
    """
    # Compute odds ratios
    a = (1 - pA) / pA
    b = (1 - pAgB) / pAgB
    c = (1 - pAgC) / pAgC
    
    # Compute combined probability
    pAgBC = 1 / (1 + b * (c / a) ** tau)
    
    return pAgBC

def entropy(P, base = None):
    """
    Calculate the entropy of a discrete probability distribution.

    The entropy is calculated using the formula:
    H(P) = -sum(P_i * log_b(P_i))

    Parameters
    ----------
    P : numpy.ndarray
        Probability distribution. Can be a 1D or 2D array.
        If 2D, each row represents a different distribution.
    base : int, optional
        The logarithm base to use. If None, uses the number of elements
        in the probability distribution (P.shape[1]). Default is None.

    Returns
    -------
    numpy.ndarray
        The entropy value(s). If input P is 2D, returns an array with
        entropy for each row distribution.

    Notes
    -----
    - Input probabilities are assumed to be normalized (sum to 1)
    - Zero probabilities are handled by numpy's log function
    - For 2D input, entropy is calculated row-wise

    Examples
    --------
    >>> P = np.array([0.5, 0.5])
    >>> entropy(P)
    1.0

    >>> P = np.array([[0.5, 0.5], [0.1, 0.9]])
    >>> entropy(P)
    array([1.0, 0.469])
    """
    P = np.atleast_2d(P)    
    if base is None:
        base = P.shape[1]
    H = -np.sum(P*np.log(P)/np.log(base), axis=1)
    return H

def class_id_to_idx(D, class_id=None):
    """
    Convert class identifiers to indices.

    This function takes an array of class identifiers and converts them to 
    corresponding indices. If no class identifiers are provided, it will 
    automatically determine the unique class identifiers from the input array.

    :param D: numpy.ndarray
        Array containing class identifiers.
    :param class_id: numpy.ndarray, optional
        Array of unique class identifiers. If None, unique class identifiers 
        will be determined from the input array `D`.
    :return: tuple
        A tuple containing:
        - D_idx (numpy.ndarray): Array with class identifiers converted to indices.
        - class_id (numpy.ndarray): Array of unique class identifiers.
    """

    if class_id is None:
        class_id = np.unique(D)
    D_idx = np.zeros(D.shape)
    for i in range(len(class_id)):
        D_idx[D==class_id[i]]=i
    # Make sure the indices are integers
    D_idx = D_idx.astype(int)
    class_id_out = np.unique(D_idx)
    
    return D_idx, class_id, class_id_out
