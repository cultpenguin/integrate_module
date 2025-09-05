"""
INTEGRATE Core Module - Probabilistic Geophysical Data Integration

This module implements rejection sampling algorithms for Bayesian inversion and
probabilistic data integration in geophysics, with particular focus on electromagnetic
(EM) data analysis. The module provides comprehensive tools for prior model generation,
forward modeling, likelihood computation, and posterior sampling.

Key Features:
    - Rejection sampling for Bayesian inversion
    - Parallel processing with shared memory optimization
    - Temperature annealing for improved sampling efficiency
    - Support for multiple data types (TDEM, multinomial, etc.)
    - Integration with GA-AEM electromagnetic forward modeling
    - Automatic temperature estimation and adaptive sampling

Main Functions:
    - integrate_rejection(): Main rejection sampling workflow (now in integrate_rejection module)
    - prior_data(): Integration of forward modeling with prior structure
    - forward_gaaem(): Electromagnetic forward modeling interface
    - likelihood_*(): Various likelihood calculation functions (now in integrate_rejection module)
    - posterior_*(): Posterior analysis and statistics

Author: Thomas Mejer Hansen
Email: tmeha@geo.au.dk
"""

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

    Returns
    -------
    bool
        True if running in a Jupyter notebook or IPython shell, False otherwise.
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
    is considered safe. Otherwise, it is not recommended unless the primary script 
    is embedded in an `if __name__ == "__main__":` block.
    
    Parameters
    ----------
    **kwargs : dict
        Additional keyword arguments including showInfo for verbosity control.
        showInfo : int, optional
            If greater than 0, prints information about the environment and 
            parallel processing status. Default is 0.
    
    Returns
    -------
    bool
        True if parallel processing is safe, False otherwise.
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

    Parameters
    ----------
    logL : numpy.ndarray
        An array of logarithmic likelihoods.
    N_above : int, optional
        The number of elements above which to consider in the sorted logL array. 
        Default is 10.
    P_acc_lev : float, optional
        The acceptance level for the calculation. Default is 0.2.
    
    Returns
    -------
    float
        The estimated temperature. It's either a positive number or infinity.

    Notes
    -----
    The function sorts the logL array in ascending order after normalizing the data 
    by subtracting the maximum value from each element. It then removes any NaN values 
    from the sorted array. If the sorted array is not empty, it calculates T_est based 
    on the N_above+1th last element in the sorted array and the natural logarithm of 
    P_acc_lev. If the sorted array is empty, it sets T_est to infinity.
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

    Parameters
    ----------
    logL : array-like
        Array of log-likelihood values.
    ns : int, optional
        Number of samples to generate. Defaults to 1.
    T : float, optional
        Temperature parameter. Defaults to 1.

    Returns
    -------
    tuple
        A tuple containing the generated samples and the acceptance probabilities.
        
        i_use_all : numpy.ndarray
            Array of indices of the selected samples.
        P_acc : numpy.ndarray
            Array of acceptance probabilities.
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

    Parameters
    ----------
    f_prior_h5 : str
        The path to the HDF5 file to process.
    **kwargs : dict
        Additional keyword arguments.
        showInfo : int, optional
            Level of verbosity for output (default is 0).
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

    This function computes various statistics for datasets in an HDF5 file based 
    on the posterior samples. The statistics include mean, median, standard 
    deviation for continuous datasets, and mode, entropy, and class probabilities 
    for discrete datasets. The computed statistics are stored in the same HDF5 file.

    Parameters
    ----------
    f_post_h5 : str, optional
        The path to the HDF5 file to process. Default is 'POST.h5'.
    **kwargs : dict
        Additional keyword arguments.
        usePrior : bool, optional
            Flag indicating whether to use the prior samples. Default is False.
        showInfo : int, optional
            Level of verbosity for output.
        updateGeometryFromData : bool, optional
            Whether to update geometry from data file. Default is True.
    
    Returns
    -------
    None
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
    """
    Sample from the posterior distribution.

    Parameters
    ----------
    is_ : int
        Index of data f_data_h5.
    d_sim : ndarray
        Simulated data.
    f_data_h5 : str, optional
        Filepath of the data file. Default is 'tTEM-Djursland.h5'.
    N_use : int, optional
        Number of samples to use. Default is 1000000.
    autoT : int, optional
        Flag indicating whether to estimate temperature. Default is 1.
    ns : int, optional
        Number of samples to draw from the posterior. Default is 400.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        
        i_use : ndarray
            Indices of the samples used.
        T : float
            Temperature.
        EV : float
            Expected value.
        is_ : int
            Index of the posterior sample.
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
    """
    Update prior structure with forward modeled data.
    
    This function integrates forward modeling results into the prior data structure,
    supporting different data types including TDEM (time-domain electromagnetic) data
    with GA-AEM forward modeling and identity transforms.
    
    Parameters
    ----------
    f_prior_in_h5 : str
        Path to input prior HDF5 file containing prior models.
    f_forward_h5 : str
        Path to forward modeling results HDF5 file.
    id : int, optional
        Data identifier for the prior structure. Default is 1.
    im : int, optional
        Model identifier for the prior structure. Default is 1.
    doMakePriorCopy : int, optional
        Flag to create a copy of the prior file (0=no copy, 1=copy). Default is 0.
    parallel : bool, optional
        Enable parallel processing for forward modeling. Default is True.
    
    Returns
    -------
    str
        Path to the updated prior HDF5 file containing integrated data.
    
    Notes
    -----
    The function automatically detects the data type from the forward modeling file
    and calls appropriate integration methods (GA-AEM for TDEM, identity for direct data).
    Prints error messages for unsupported data types or methods.
    """
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
                    stmfiles=None, 
                    tx_height=np.array(()), 
                    txrx_dx = -13, 
                    txrx_dy = 0,
                    txrx_dz     = .1,
                    GEX={}, 
                    file_gex=None, 
                    showtime=False, 
                    **kwargs):
    """
    Perform forward modeling using the GA-AEM method.

    Parameters
    ----------
    C : numpy.ndarray, optional
        Conductivity array. Default is np.array(()).
    thickness : numpy.ndarray, optional
        Thickness array. Default is np.array(()).
    stmfiles : list, optional
        List of STM files. Default is None.
    tx_height : numpy.ndarray, optional
        Transmitter height array. Default is np.array(()).
    txrx_dx : float, optional
        X-distance between transmitter and receiver. Default is -13.
    txrx_dy : float, optional
        Y-distance between transmitter and receiver. Default is 0.
    txrx_dz : float, optional
        Z-distance between transmitter and receiver. Default is 0.1.
    GEX : dict, optional
        GEX dictionary. Default is {}.
    file_gex : str, optional
        Path to GEX file. Default is None.
    showtime : bool, optional
        Flag to display execution time. Default is False.
    **kwargs : dict
        Additional keyword arguments.
        showInfo : int, optional
            Level of verbosity for output.
        doCompress : bool, optional
            Flag to enable layer compression. Default is True.
    
    Returns
    -------
    numpy.ndarray
        Forward modeled data array.
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

    # Handle None defaults
    if stmfiles is None:
        stmfiles = []
    if file_gex is None:
        file_gex = ''

    #print(stmfiles)
    #print(file_gex)

    if (len(stmfiles)>0) and (file_gex != '') and (len(GEX)==0):
        # GEX FILE and STM FILES
        if (showInfo)>1:
            print('Using submitted GEX file (%s)' % (file_gex))
        GEX =   ig.read_gex(file_gex)
    elif (len(stmfiles)>0):
        # USING STM FILES
        if (showInfo)>1:
            print('Using submitted STM files (%s)' % (stmfiles))

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
        if 'filename' in GEX:
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
        # Here we should read the number of gates from the lines in STMFILES that conatin 'NumberOfWindows = 41'
        ng = 41

    # pinrt txrx_dx, txrx_dy, txrx_dz
    if (showInfo>0):
        print('txrx_dx=%f, txrx_dy=%f, txrx_dz=%f' % (txrx_dx, txrx_dy, txrx_dz))
        print('ng=%d' % ng)
        

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
    Perform forward modeling using the GA-AEM method on a chunk of data.

    Parameters
    ----------
    C_chunk : numpy.ndarray
        The chunk of data to be processed.
    tx_height_chunk : numpy.ndarray
        The transmitter heights for this chunk.
    thickness : float
        The thickness of the model.
    stmfiles : list
        A list of STM files.
    file_gex : str
        The path to the GEX file.
    Nhank : int
        The number of Hankel functions.
    Nfreq : int
        The number of frequencies.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    numpy.ndarray
        The result of the forward modeling.
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
    """
    Return the number of handles used by the current process (Windows only).
    
    Returns
    -------
    int
        The number of handles used by the current process.
    """
    import psutil
    import os
    return psutil.Process(os.getpid()).num_handles()

def prior_data_gaaem(f_prior_h5, file_gex=None, stmfiles=None, N=0, doMakePriorCopy=True, im=1, id=1, im_height=0, Nhank=280, Nfreq=12, is_log=False, parallel=True, **kwargs):
    """
    Generate prior data for the GA-AEM method.

    Parameters
    ----------
    f_prior_h5 : str
        Path to the prior data file in HDF5 format.
    file_gex : str
        Path to the file containing geophysical exploration data.
    N : int, optional
        Number of soundings to consider. Default is 0 (use all).
    doMakePriorCopy : bool, optional
        Flag indicating whether to make a copy of the prior file. Default is True.
    im : int, optional
        Index of the model. Default is 1.
    id : int, optional
        Index of the data. Default is 1.
    im_height : int, optional
        Index of the model for height. Default is 0.
    Nhank : int, optional
        Number of Hankel transform quadrature points. Default is 280.
    Nfreq : int, optional
        Number of frequencies. Default is 12.
    is_log : bool, optional
        Flag to apply logarithmic scaling to data. Default is False.
    parallel : bool, optional
        Flag indicating whether multiprocessing is used. Default is True.
    **kwargs : dict
        Additional keyword arguments.
        Ncpu : int, optional
            Number of CPUs to use (default is 0 for all available).
        showInfo : int, optional
            Level of verbosity for output.
    
    Returns
    -------
    str
        Filename of the HDF5 file containing the updated prior data.
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

    # if is not None file_gex
    if (file_gex is not None):
        if not os.path.isfile(file_gex):
            print("ERRROR: file_gex=%s does not exist in the current folder." % file_gex)

    if (stmfiles is not None):
        for i in range(len(stmfiles)):
            if not os.path.isfile(stmfiles[i]):
                print("ERRROR: stmfiles[%d]=%s does not exist in the current folder." % (i,stmfiles[i]))
 

    if doMakePriorCopy:

        # If file_gex is not None, then use it to get the file_base_name
        if (file_gex is not None) and os.path.isfile(file_gex): 
            file_basename = os.path.splitext(os.path.basename(file_gex))[0]
        elif (stmfiles is not None) and (len(stmfiles)>0):
            file_basename = os.path.splitext(os.path.basename(stmfiles[0]))[0]
        else:
            file_basename = 'GAAEM'
        
        print('Using file_basename=%s' % file_basename)

        if N < N_in:
            f_prior_data_h5 = '%s_%s_N%d_Nh%d_Nf%d.h5' % (os.path.splitext(f_prior_h5)[0], os.path.splitext(file_basename)[0], N, Nhank, Nfreq)
        else:
            f_prior_data_h5 = '%s_%s_Nh%d_Nf%d.h5' % (os.path.splitext(f_prior_h5)[0], os.path.splitext(file_basename)[0], Nhank, Nfreq)
        

        if (showInfo>0):
            print("Creating a copy of %s" % (f_prior_h5))
            print("                as %s" % (f_prior_data_h5))
        if (showInfo>1):
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
            if (showInfo>1):
                print('Using tx_height')
            D = ig.forward_gaaem(C=C, 
                                 thickness=thickness, 
                                 tx_height=tx_height, 
                                 file_gex=file_gex, 
                                 stmfiles=stmfiles,
                                 Nhank=Nhank, 
                                 Nfreq=Nfreq, 
                                 parallel=parallel, **kwargs)
        else:
            D = ig.forward_gaaem(C=C, 
                                 thickness=thickness, 
                                 file_gex=file_gex, 
                                 stmfiles=stmfiles,
                                 Nhank=Nhank, 
                                 Nfreq=Nfreq, 
                                 parallel=parallel, **kwargs)
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
            #print(f"Handle count before pool: {handle_count_before}")

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
        
        if is_log:
            D = np.log10(D)

        if os.name == 'nt' and 'get_process_handle_count' in globals():
            # Log handle count after pool is closed
            handle_count_after = get_process_handle_count()
            #   print(f"Handle count after pool: {handle_count_after}")


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
    """
    Generate data D{id} from model M{im} in the prior file f_prior_h5 as an identity of M{im}.

    Parameters
    ----------
    f_prior_h5 : str
        Path to the prior data file in HDF5 format.
    id : int, optional
        Index of the data. If id=0, the next available data id is used. Default is 0.
    im : int, optional
        Index of the model. Default is 1.
    N : int, optional
        Number of soundings to consider. Default is 0 (use all).
    doMakePriorCopy : bool, optional
        Flag indicating whether to make a copy of the prior file. Default is False.
    **kwargs : dict
        Additional keyword arguments.
        showInfo : int, optional
            Level of verbosity for output.
        forceDeleteExisting : bool, optional
            Flag to force deletion of existing data. Default is True.
    
    Returns
    -------
    str
        Path to the HDF5 file containing the updated prior data.
    """
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

    Parameters
    ----------
    lay_dist : str, optional
        Distribution of the number of layers. Options are 'chi2' and 'uniform'. 
        Default is 'uniform'.
    dz : float, optional
        Depth discretization step. Default is 1.
    z_max : float, optional
        Maximum depth. Default is 90.
    NLAY_min : int, optional
        Minimum number of layers. Default is 3.
    NLAY_max : int, optional
        Maximum number of layers. Default is 6.
    NLAY_deg : int, optional
        Degrees of freedom for chi-square distribution. Only applicable if 
        lay_dist is 'chi2'. Default is 6.
    RHO_dist : str, optional
        Distribution of resistivity within each layer. Options are 'log-uniform', 
        'uniform', 'normal', and 'lognormal'. Default is 'log-uniform'.
    RHO_min : float, optional
        Minimum resistivity value. Default is 0.1.
    RHO_max : float, optional
        Maximum resistivity value. Default is 100.
    RHO_MEAN : float, optional
        Mean resistivity value. Only applicable if RHO_dist is 'normal' or 
        'lognormal'. Default is 100.
    RHO_std : float, optional
        Standard deviation of resistivity value. Only applicable if RHO_dist is 
        'normal' or 'lognormal'. Default is 80.
    N : int, optional
        Number of prior models to generate. Default is 100000.
    **kwargs : dict
        Additional keyword arguments.
        f_prior_h5 : str, optional
            Path to the prior model file in HDF5 format. Default is ''.
        showInfo : int, optional
            Level of verbosity for output.

    Returns
    -------
    str
        Filepath of the saved prior model.
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
        NLAY_max = np.max(NLAY)

    # Force NLAY to be a 2 dimensional numpy array
    NLAY = NLAY[:, np.newaxis]
    
    z_min = 0
    z = np.arange(z_min, z_max, dz)
    nz= len(z)
    M_rho = np.zeros((N, nz))
    nm_sparse = NLAY_max+NLAY_max-1
    M_rho_sparse = np.ones((N, nm_sparse))*np.nan
    

    #% simulate the number of layers as in integer
    for i in tqdm(range(N), mininterval=1, disable=(showInfo<0), desc='prior_layered', leave=False):
    

        ## z_boundaries is the depth boundaries of the base of NLAY[i]-1 layers in meters. The values should be between 0 and z_max.
        z_boundaires = np.random.random(NLAY[i]-1)*z_max
        ## Compute i_boundaries as the nearest position of the nearest intefer in z
        i_boundaries = np.searchsorted(z, z_boundaires)
        ## Ensure indices don't exceed array bounds
        i_boundaries = np.minimum(i_boundaries, nz - 1)
        #print("i_boundaries", i_boundaries, "z_boundaires", z_boundaires, "NLAY[i]", NLAY[i])

        ### simulate the resistivity in each layer
        if RHO_dist=='log-normal':
            rho_all=np.random.lognormal(mean=np.log(RHO_MEAN), sigma=np.log(RHO_std), size=NLAY[i])
        elif RHO_dist=='normal':
            rho_all=np.random.normal(loc=RHO_MEAN, scale=RHO_std, size=NLAY[i])
        elif RHO_dist=='log-uniform':
            rho_all=np.exp(np.random.uniform(np.log(RHO_min), np.log(RHO_max), NLAY[i]))
        elif RHO_dist=='uniform':
            rho_all=np.random.uniform(RHO_min, RHO_max, NLAY[i])



        rho = np.zeros(nz)+rho_all[0]
        for j in range(len(i_boundaries)):
            rho[i_boundaries[j]:] = rho_all[j+1]

        M_rho[i]=rho        


        # m_current should be the concatenation of z_boundaires and rho_alls
        m_current = np.concatenate((z_boundaires, rho_all))
        #print("m_current", m_current)
        M_rho_sparse[i,0:len(m_current)] = m_current


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
    ig.save_prior_model(f_prior_h5,M_rho_sparse.astype(np.float32),
                im=2,
                name='sparse - depth-resistivity',
                is_discrete = 0, 
                x = np.arange(0,nm_sparse),
                z = np.arange(0,nm_sparse),
                force_replace=True,
                showInfo=showInfo,
                )


    if (showInfo>1):
        print("Saving '/M3' prior model  %s" % f_prior_h5)
    ig.save_prior_model(f_prior_h5,NLAY.astype(np.float32),
                        im=3,
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
    
    All models have the same number of layers! See also: prior_model_workbench.
 
    Parameters
    ----------
    N : int, optional
        Number of prior models to generate. Default is 100000.
    RHO_dist : str, optional
        Distribution of resistivity within each layer. Options are 'log-uniform', 
        'uniform', 'normal', 'lognormal', and 'chi2'. Default is 'log-uniform'.
    z1 : float, optional
        Minimum depth value. Default is 0.
    z_max : float, optional
        Maximum depth value. Default is 100.
    nlayers : int, optional
        Number of layers. Default is 0 (uses 30 if less than 1).
    p : int, optional
        Power parameter for thickness increase. Default is 2.
    NLAY_min : int, optional
        Minimum number of layers. Default is 3.
    NLAY_max : int, optional
        Maximum number of layers. Default is 6.
    RHO_min : float, optional
        Minimum resistivity value. Default is 1.
    RHO_max : float, optional
        Maximum resistivity value. Default is 300.
    RHO_mean : float, optional
        Mean resistivity value. Only applicable if RHO_dist is 'normal' or 
        'lognormal'. Default is 180.
    RHO_std : float, optional
        Standard deviation of resistivity value. Only applicable if RHO_dist is 
        'normal' or 'lognormal'. Default is 80.
    chi2_deg : int, optional
        Degrees of freedom for chi2 distribution. Only applicable if RHO_dist is 
        'chi2'. Default is 100.
    **kwargs : dict
        Additional keyword arguments.
        f_prior_h5 : str, optional
            Path to the prior model file in HDF5 format. Default is ''.
        showInfo : int, optional
            Level of verbosity for output.

    Returns
    -------
    str
        Filepath of the saved prior model.
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
    Generate a prior model with increasingly thick layers.
 
    Parameters
    ----------
    N : int, optional
        Number of prior models to generate. Default is 100000.
    p : int, optional
        Power parameter for thickness increase. Default is 2.
    z1 : float, optional
        Minimum depth value. Default is 0.
    z_max : float, optional
        Maximum depth value. Default is 100.
    dz : float, optional
        Depth discretization step. Default is 1.
    lay_dist : str, optional
        Distribution of the number of layers. Options are 'chi2' and 'uniform'. 
        Default is 'uniform'.
    nlayers : int, optional
        Number of layers. If greater than 0, sets both NLAY_min and NLAY_max 
        to this value. Default is 0.
    NLAY_min : int, optional
        Minimum number of layers. Default is 3.
    NLAY_max : int, optional
        Maximum number of layers. Default is 6.
    NLAY_deg : int, optional
        Degrees of freedom for chi-square distribution. Only applicable if 
        lay_dist is 'chi2'. Default is 5.
    RHO_dist : str, optional
        Distribution of resistivity within each layer. Options are 'log-uniform', 
        'uniform', 'normal', 'lognormal', and 'chi2'. Default is 'log-uniform'.
    RHO_min : float, optional
        Minimum resistivity value. Default is 1.
    RHO_max : float, optional
        Maximum resistivity value. Default is 300.
    RHO_mean : float, optional
        Mean resistivity value. Only applicable if RHO_dist is 'normal' or 
        'lognormal'. Default is 180.
    RHO_std : float, optional
        Standard deviation of resistivity value. Only applicable if RHO_dist is 
        'normal' or 'lognormal'. Default is 80.
    chi2_deg : int, optional
        Degrees of freedom for chi2 distribution. Only applicable if RHO_dist is 
        'chi2'. Default is 100.
    **kwargs : dict
        Additional keyword arguments.
        f_prior_h5 : str, optional
            Path to the prior model file in HDF5 format. Default is ''.
        showInfo : int, optional
            Level of verbosity for output.

    Returns
    -------
    str
        Filepath of the saved prior model.
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
        NLAY_max = np.max(NLAY)  # Update NLAY_max to accommodate chi2 distribution
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

    nm_sparse = NLAY_max+NLAY_max-1
    if (showInfo>1):
        print("nm_sparse", nm_sparse)
    M_rho_sparse = np.ones((N, nm_sparse))*np.nan
    

    for i in tqdm(range(N), mininterval=1, disable=(showInfo<0), desc='prior_workbench', leave=False):
        nlayers = NLAY[i][0]
        #print(nlayers)
        z2=z_max
        z_single= z1 + (z2 - z1) * np.linspace(0, 1, nlayers) ** p
        
        if RHO_dist=='uniform':
            M_rho_single = np.random.uniform(low=RHO_min, high = RHO_max, size=(1, nlayers))
        elif RHO_dist=='log-uniform':
            M_rho_single = np.exp(np.random.uniform(low=np.log(RHO_min), high = np.log(RHO_max), size=(1, nlayers)))
        elif RHO_dist=='normal':
            M_rho_single = np.random.normal(loc=RHO_mean, scale = RHO_std, size=(1, nlayers))
        elif RHO_dist=='log-normal' or RHO_dist=='lognormal':
            M_rho_single = np.random.lognormal(mean=np.log(RHO_mean), sigma = RHO_std/RHO_mean, size=(1, nlayers))
        elif RHO_dist=='chi2':
            M_rho_single = np.random.chisquare(df = chi2_deg, size=(1, nlayers))
        else:
            # Default to log-uniform if RHO_dist is not recognized
            M_rho_single = np.exp(np.random.uniform(low=np.log(RHO_min), high = np.log(RHO_max), size=(1, nlayers)))

        for j in range(nlayers):
            ind = np.where(z>=z_single[j])[0]
            M_rho[i,ind]= M_rho_single[0,j]


        m_current = np.concatenate((z_single[0:-1].flatten(), M_rho_single.flatten()))
        M_rho_sparse[i,0:len(m_current)] = m_current



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
    ig.save_prior_model(f_prior_h5,M_rho_sparse.astype(np.float32),
                im=2,
                name='sparse - depth-resistivity',
                is_discrete = 0, 
                x = np.arange(0,nm_sparse),
                z = np.arange(0,nm_sparse),
                force_replace=True,
                showInfo=showInfo,
                )

    if (showInfo>1):
        print("Saving '/M3' prior model  %s" % f_prior_h5)
    ig.save_prior_model(f_prior_h5,NLAY.astype(np.float32),
                        im=3,
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

    Parameters
    ----------
    f_post_h5 : str
        Path to the input h5 file.
    im : int, optional
        Index of model parameter number, M[im]. Default is 2.
    icat : list, optional
        List of category indices. Default is [0].
    usePrior : bool, optional
        Flag indicating whether to use prior. Default is False.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        
        thick_mean : ndarray
            Array of mean cumulative thickness.
        thick_median : ndarray
            Array of median cumulative thickness.
        thick_std : ndarray
            Array of standard deviation of cumulative thickness.
        class_out : list
            List of class names.
        X : ndarray
            Array of X values.
        Y : ndarray
            Array of Y values.
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


# # Import rejection sampling functions from separate module
# # Note: These imports work when the package is properly installed
# # For development, you may need to modify paths or use try/except
# try:
#     from integrate.integrate_rejection import (
#         integrate_rejection,
#         integrate_rejection_range, 
#         integrate_posterior_main,
#         integrate_posterior_chunk,
#         likelihood_gaussian_diagonal,
#         likelihood_gaussian_full,
#         likelihood_multinomial,
#         select_subset_for_inversion,
#         create_shared_memory,
#         reconstruct_shared_arrays,
#         cleanup_shared_memory
#     )
# except ImportError:
#     # For development when running directly, try relative import
#     from .integrate_rejection import (
#         integrate_rejection,
#         integrate_rejection_range, 
#         integrate_posterior_main,
#         integrate_posterior_chunk,
#         likelihood_gaussian_diagonal,
#         likelihood_gaussian_full,
#         likelihood_multinomial,
#         select_subset_for_inversion,
#         create_shared_memory,
#         reconstruct_shared_arrays,
#         cleanup_shared_memory
#     )


# Functions moved to integrate_rejection.py
# Functions moved to integrate_rejection.py have been removed

# All rejection sampling related functions have been moved to integrate_rejection.py
# This includes:
# - reconstruct_shared_arrays
# - cleanup_shared_memory
# - integrate_rejection
# - integrate_rejection_range
# - integrate_posterior_main
# - integrate_posterior_chunk
# - likelihood_gaussian_diagonal
# - likelihood_gaussian_full
# - likelihood_multinomial
# - select_subset_for_inversion
# moved to integrate_rejection.py


# %% Synthetic data

def synthetic_case(case='Wedge', **kwargs):
    """
    Generate synthetic geological models for different cases.
    
    This function creates synthetic 2D geological models for testing and validation 
    purposes. Supports 'Wedge' and '3Layer' model types with customizable parameters.
    
    Parameters
    ----------
    case : str, optional
        The type of synthetic case to generate. Options are 'Wedge' and '3Layer'. 
        Default is 'Wedge'.
    **kwargs : dict
        Additional parameters for synthetic case generation.
        
        Common Parameters
        -----------------
        showInfo : int, optional
            If greater than 0, print information about the generated case. Default is 0.
        
        Parameters for 'Wedge' case
        ---------------------------
        x_max : int, optional
            Maximum x-dimension size. Default is 1000.
        dx : float, optional
            Step size in the x-dimension. Default is 1000./x_max.
        z_max : int, optional
            Maximum z-dimension size. Default is 90.
        dz : float, optional
            Step size in the z-dimension. Default is 1.
        z1 : float, optional
            Depth at which the wedge starts. Default is z_max/10.
        rho : list, optional
            Density values for different layers. Default is [100, 200, 120].
        wedge_angle : float, optional
            Angle of the wedge in degrees. Default is 1.
        
        Parameters for '3Layer' case
        ----------------------------
        x_max : int, optional
            Maximum x-dimension size. Default is 100.
        x_range : float, optional
            Range in the x-dimension for the cosine function. Default is x_max/4.
        dx : float, optional
            Step size in the x-dimension. Default is 1.
        z_max : int, optional
            Maximum z-dimension size. Default is 60.
        dz : float, optional
            Step size in the z-dimension. Default is 1.
        z1 : float, optional
            Depth at which the first layer ends. Default is z_max/3.
        z_thick : float, optional
            Thickness of the second layer. Default is z_max/2.
        rho1_1 : float, optional
            Density at the start of the first layer. Default is 120.
        rho1_2 : float, optional
            Density at the end of the first layer. Default is 10.
        rho2_1 : float, optional
            Density at the start of the second layer. Default is rho1_2.
        rho2_2 : float, optional
            Density at the end of the second layer. Default is rho1_1.
        rho3 : float, optional
            Density of the third layer. Default is 120.

    Returns
    -------
    tuple
        A tuple containing (M, x, z) where M is the generated synthetic model, 
        x is x-coordinates, z is z-coordinates.
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

    Parameters
    ----------
    D : numpy.ndarray
        Array containing class identifiers.
    class_id : numpy.ndarray, optional
        Array of unique class identifiers. If None, unique class identifiers 
        will be determined from the input array `D`. Default is None.
    
    Returns
    -------
    tuple
        A tuple containing the following elements:
        
        D_idx : numpy.ndarray
            Array with class identifiers converted to indices.
        class_id : numpy.ndarray
            Array of unique class identifiers.
        class_id_out : numpy.ndarray
            Array of unique output class identifiers.
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



def get_hypothesis_probability(f_post_h5_arr, T=1):
    """
    Calculate hypothesis probabilities and related statistics from posterior files.
    
    This function processes an array of HDF5 file paths containing posterior evidences
    to compute normalized probabilities for each hypothesis, along with evidence values,
    mode hypotheses, and entropy measures.
    
    Parameters
    ----------
    f_post_h5_arr : list of str
        Array of file paths to HDF5 files containing posterior evidence values. 
        Each file should have an '/EV' dataset.
    T : float, optional
        Temperature parameter that applies annealing. Higher temperatures create 
        more uniform distributions. Useful for smoothing distributions from smaller 
        lookup tables. Default is 1.
    
    Returns
    -------
    tuple
        A tuple containing the following elements:
        
        P : numpy.ndarray
            Normalized probabilities for each hypothesis (shape: n_hypothesis, n_samples).
        EV_all : numpy.ndarray
            Evidence values for each hypothesis and sample (shape: n_hypothesis, n_samples).
        MODE_hypothesis : numpy.ndarray
            Index of most probable hypothesis per sample (shape: n_samples).
        ENT_hypothesis : numpy.ndarray
            Entropy of hypothesis distribution per sample, normalized by number 
            of hypotheses (shape: n_samples).
    
    Notes
    -----
    The probability normalization uses the log-sum-exp trick to avoid numerical
    underflow issues when working with evidence values.
    """

    from scipy import stats

    n_hypothesis = len(f_post_h5_arr)
    EV_all = []
    for f_post_h5 in f_post_h5_arr:
        with h5py.File(f_post_h5, 'r') as f:
            EV = f['/EV'][()]
        EV_all.append(EV)
    EV_all = np.array(EV_all)
    # subtract the small value on each column form each column
    P  = np.exp(EV_all - np.max(EV_all, axis=0))**(1/T)
    P  = np.exp((1/T)*(EV_all - np.max(EV_all, axis=0)))
    #P_acc = np.exp((1/T) * (logL - np.nanmax(logL)))
    # Normalize each column to sum to 1 using NumPy broadcasting
    P = P / np.sum(P, axis=0, keepdims=True)

    ENT_hypothesis = np.zeros(P.shape[1])
    MODE_hypothesis = np.zeros(P.shape[1])

    for i in range(P.shape[1]):        
        # get the entropy for each hypothesis
        ENT_hypothesis[i] = stats.entropy(P[:,i], base=n_hypothesis)
        # get the is of the hypothesis with the maximum probability
        MODE_hypothesis[i] = np.argmax(P[:,i], axis=0)

    

    return P, EV_all, MODE_hypothesis, ENT_hypothesis 



def sample_posterior_multiple_hypotheses(f_post_h5_arr, P_hypothesis=None):
    """
    Sample posterior models from multiple hypotheses.
    
    This function samples posterior models from multiple hypotheses stored in HDF5 files,
    according to the given hypothesis probabilities.
    
    Parameters
    ----------
    f_post_h5_arr : list of str
        List of paths to HDF5 files containing posterior models for different hypotheses.
    P_hypothesis : numpy.ndarray, optional
        Array of shape (n_hypotheses, n_soundings) containing probability of each 
        hypothesis for each sounding. If None, uniform probabilities are used. 
        Default is None.
    
    Returns
    -------
    list of numpy.ndarray
        List of posterior model arrays. Each array has shape (n_soundings, n_samples, 
        n_parameters), where n_samples is determined by the first hypothesis's number 
        of samples.
    
    Notes
    -----
    The function combines posterior samples from different hypotheses in proportion to their
    probabilities and ensures the total number of samples equals the first hypothesis's 
    sample count.
    """

    import numpy as np
    import h5py
    import integrate as ig
    
    f_prior_h5_arr = []
    M_all = []
    i_use_all = []
    n_use_all = []
    for ip in range(len(f_post_h5_arr)):
            f_post_h5 = f_post_h5_arr[ip]
            with h5py.File(f_post_h5, 'r') as f_post:
                f_prior_h5 = f_post['/'].attrs['f5_prior']
                f_data_h5 = f_post['/'].attrs['f5_data']
                f_prior_h5_arr.append(f_prior_h5)

                i_use = f_post['/i_use'][:]
                i_use_all.append(i_use)
                n_use_all.append(i_use.shape[1])

                print("loading prior model %s" % f_prior_h5)
                M, idx = ig.load_prior_model(f_prior_h5)
                M_all.append(M)
    print(f_prior_h5_arr)
    i_use_all = np.array(i_use_all)
    n_use_all = np.array(n_use_all)

    if P_hypothesis is None:
        print('Using unform hypothesis probability')
        D = ig.load_data(f_data_h5)
        n_soundings = D['d_obs'][0].shape[0]

        P_hypothesis = np.ones((len(f_post_h5_arr), n_soundings))
        P_hypothesis = P_hypothesis/np.sum(P_hypothesis, axis=0)

    # def sample_posterior_multipleh_hypotheses(f_post_h5_arr, P_hypothesis, n_post=100):
    nsoundings = P_hypothesis.shape[1]
    # If different hypothsis have different number of realizations..
    #for i in range(nsoundings):


    M_post_arr = []

    for im in range(len(M)):
        print("im=%d/%d" % (im+1,len(M)))
        nm = M[im].shape[1]
        M_post = np.zeros((nsoundings, n_use_all[0],nm))

        for i in range(nsoundings):
            # get the probabliity of each hypothesis
            P_hypothesis_is = P_hypothesis[:,i]
            i_use = i_use_all[:,i,:]

            n_use = P_hypothesis_is*n_use_all
            n_use = np.round(n_use).astype(int)
            #print(n_use)
            n_sum = np.sum(n_use)
            # make sure that the sum of n_use is equal to the number of realizations
            delta_n = n_use_all[0]-n_sum
            if delta_n > 0:
                n_use[0] = n_use[0] + delta_n
            elif delta_n < 0:
                pass
                #n_use[0] = n_use[0] - np.abs(delta_n)
            n_sum = np.sum(n_use)
            
            M_dummy = []
            for j in range(len(n_use)):  
                # use the first  realizations from i_use[j]
                #i_use_single = i_use[j,:n_use[j]]
                # use n_use[j] random realizations from iuse[j]
                #print('    j=%d, n_use=%d' % (j,n_use[j]))
                if n_use[j]>0:                    
                    i_use_single = np.random.choice(i_use[j], n_use[j], replace=False)                
                    M_dummy.append(M_all[j][im][i_use_single,:])
                    
            M_sounding = np.concatenate(M_dummy, axis=0)        
            try:
                # take the first n_use_all[0] realizations
                M_post[i]=M_sounding[:n_use_all[0]]
            except:
                print('i=%d, [%d], n_sum=%d, delta_n=%d, n_use_all[0]=%d' % (i,M_sounding.shape[0],n_sum,delta_n,n_use_all[0]))
                try:
                    M_post[i]=M_sounding[:,:n_use[0]]
                except:
                    pass
                
        M_post_arr.append(M_post)    
                
    return M_post_arr


# %% TIMING FUNCTIONS
# Functions moved from integrate_timing_cli.py

def allocate_large_page():
    """
    Allocate a 2MB large page if running on Windows.
    
    Returns
    -------
    int or None
        Pointer to allocated memory on success, None on failure or non-Windows systems.
        
    Notes
    -----
    Large pages can improve performance but require specific Windows privileges.
    """
    import os
    import ctypes
    
    if os.name == "nt":
        kernel32 = ctypes.windll.kernel32
        kernel32.VirtualAlloc.restype = ctypes.c_void_p
        
        LARGE_PAGE_SIZE = 2 * 1024 * 1024  # 2MB
        
        MEM_COMMIT = 0x1000
        MEM_LARGE_PAGES = 0x20000000
        PAGE_READWRITE = 0x04
        
        ptr = kernel32.VirtualAlloc(None, LARGE_PAGE_SIZE, MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE)
        
        if not ptr:
            error_code = ctypes.GetLastError()
            print(f"Failed to allocate large page. Error code: {error_code}")
            return None
        
        print(f"Successfully allocated {LARGE_PAGE_SIZE} bytes at address {hex(ptr)}")
        return ptr
    else:
        print("Large pages are only supported on Windows.")
        return None


def timing_compute(N_arr=[], Nproc_arr=[]):
    """
    Execute timing benchmark for INTEGRATE workflow components.
    
    This function benchmarks the performance of the complete INTEGRATE workflow including
    prior model generation, forward modeling, rejection sampling, and posterior statistics
    computation across different dataset sizes and processor counts.
    
    Parameters
    ----------
    N_arr : array_like, optional
        Array of dataset sizes (number of prior models) to test. 
        Default is [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000].
    Nproc_arr : array_like, optional
        Array of processor counts to test. Default is powers of 2 up to available CPUs.
        
    Returns
    -------
    str
        Filename of the NPZ file containing timing results.
        
    Notes
    -----
    The benchmark tests four main components:
    1. Prior model generation (layered geological models)
    2. Forward modeling using GA-AEM electromagnetic simulation
    3. Rejection sampling for Bayesian inversion  
    4. Posterior statistics computation
    
    Results are saved to an NPZ file with timing arrays and system information.
    The function automatically uses appropriate test data and handles parallel processing
    configuration based on system capabilities.
    """
    import integrate as ig
    # check if parallel computations can be performed
    parallel = ig.use_parallel(showInfo=1)

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import loglog
    import time
    import h5py
    # get name of CPU
    import psutil

    # Get hostname and number of processors
    import socket
    hostname = socket.gethostname()
    import platform
    hostname = platform.node()
    system = platform.system()

    ## Get number of processors
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    Ncpu = physical_cores

    print("# TIMING TEST")
    print("Hostname (system): %s (%s) " % (hostname, system))
    print("Number of processors: %d" % Ncpu)
    
    # SELECT THE CASE TO CONSIDER AND DOWNLOAD THE DATA
    files = ig.get_case_data(showInfo=-1)
    f_data_h5 = files[0]
    file_gex= ig.get_gex_file_from_data(f_data_h5)

    print("Using data file: %s" % f_data_h5)
    print("Using GEX file: %s" % file_gex)

    with h5py.File(f_data_h5, 'r') as f:
        nobs = f['D1/d_obs'].shape[0]


    ## Setup the timing test

    #### Set the size of the data sets to test
    if len(N_arr)==0:
        N_arr = np.array([100,500,1000,5000,10000,50000,100000, 500000, 1000000, 5000000])

    # Set the number of cores to test
    if len(Nproc_arr)==0:
        Nproc_arr=2**(np.double(np.arange(1+int(np.log2(Ncpu)))))   

    n1 = len(N_arr)
    n2 = len(Nproc_arr)    

    print("Testing on %d data sets of size(s):" % len(N_arr), N_arr)
    print("Testing on %d sets of core(s):" % len(Nproc_arr), Nproc_arr)


    file_out  = 'timing_%s-%s-%dcore_Nproc%d_N%d.npz' % (hostname,system,Ncpu,len(Nproc_arr), len(N_arr))
    print("Writing results to %s " % file_out)

    ## TIMING

    showInfo = 0

    T_prior = np.zeros((n1,n2))*np.nan
    T_forward = np.zeros((n1,n2))*np.nan
    T_rejection = np.zeros((n1,n2))*np.nan
    T_poststat = np.zeros((n1,n2))*np.nan

    testRejection = True
    testPostStat = True  
                
    for j in np.arange(n2):
        Ncpu = int(Nproc_arr[j])

        for i in np.arange(len(N_arr)):
            N=int(N_arr[i])
            Ncpu_min = int(np.floor(2**(np.log10(N)-4)))
            
            print('=====================================================')
            print('TIMING: N=%d, Ncpu=%d, Ncpu_min=%d'%(N,Ncpu,Ncpu_min))
            print('=====================================================')
            
            RHO_min = 1
            RHO_max = 800
            z_max = 50 
            useP = 1
            
            if (Ncpu>=Ncpu_min):
                    
                t0_prior = time.time()
                if useP ==1:
                    ## Layered model    
                    f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='chi2', NLAY_deg=5, z_max = z_max, RHO_dist='log-uniform', RHO_min=RHO_min, RHO_max=RHO_max, showInfo=showInfo)
                    #f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='uniform', z_max = z_max, NLAY_min=1, NLAY_max=3, rho_dist='log-uniform', RHO_min=RHO_min, RHO_max=RHO_max)
                    #f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='uniform', z_max = z_max, NLAY_min=1, NLAY_max=8, rho_dist='log-uniform', RHO_min=RHO_min, RHO_max=RHO_max)
                else: 
                    ## N layer model with increasing thickness
                    f_prior_h5 = ig.prior_model_workbench(N=N, z_max = 30, nlayers=20, rho_min = RHO_min, rho_max = RHO_max, showInfo=showInfo)
                #t_prior.append(time.time()-t0_prior)
                T_prior[i,j] = time.time()-t0_prior

            
                #ig.plot_prior_stats(f_prior_h5)
                #% A2. Compute prior DATA
                t0_forward = time.time()
                f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, Ncpu=Ncpu, showInfo=showInfo)
                T_forward[i,j]=time.time()-t0_forward

                #% READY FOR INVERSION
                N_use = 1000000
                t0_rejection = time.time()
                if testRejection:
                    f_post_h5 = ig.integrate_rejection(f_prior_data_h5, f_data_h5, N_use = N_use, parallel=1, updatePostStat=False,  Ncpu=Ncpu, showInfo=showInfo)
                T_rejection[i,j]=time.time()-t0_rejection

                #% Compute some generic statistic of the posterior distribution (Mean, Median, Std)
                t0_poststat = time.time()
                if testPostStat and testRejection:
                    ig.integrate_posterior_stats(f_post_h5,showInfo=showInfo)
                    T_poststat[i,j]=time.time()-t0_poststat

            T_total = T_prior + T_forward + T_rejection + T_poststat
            np.savez(file_out, T_total=T_total, T_prior=T_prior, T_forward=T_forward, T_rejection=T_rejection, T_poststat=T_poststat, N_arr=N_arr, Nproc_arr=Nproc_arr, nobs=nobs)
            
            
    return file_out


def timing_plot(f_timing=''):
    """
    Generate comprehensive timing analysis plots from benchmark results.
    
    This function creates multiple plots analyzing the performance characteristics
    of the INTEGRATE workflow across different dataset sizes and processor counts.
    
    Parameters
    ----------
    f_timing : str
        Path to NPZ file containing timing benchmark results from timing_compute().
        
    Returns
    -------
    None
        Saves multiple PNG files with timing analysis plots.
        
    Notes
    -----
    Generated plots include:
    - Total execution time vs processors and dataset size
    - Forward modeling performance and speedup analysis  
    - Rejection sampling performance and scaling
    - Posterior statistics computation performance
    - Cumulative time breakdowns for different processor counts
    - Comparisons with traditional least squares and MCMC methods
    
    The function handles missing data gracefully and includes reference lines
    for linear scaling to assess parallel efficiency.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    if len(f_timing)==0:
        print('No timing file provided')
        return
    else:
        print('Plotting timing results from %s' % f_timing)

    # file_out is f_timing, without file extension
    file_out = f_timing.split('.')[0]
    
    data = np.load(f_timing)
    T_prior = data['T_prior']
    T_forward = data['T_forward']
    T_rejection = data['T_rejection']
    T_poststat = data['T_poststat']

    N_arr = data['N_arr']
    Nproc_arr = data['Nproc_arr']

    try:
        T_total = data['T_total']
    except:
        T_total = T_prior + T_forward + T_rejection + T_poststat

    try:
        nobs=data['nobs']
    except:
        nobs=11693

    # Plot
    # LSQ, Assumed time, in seconds, for least squares inversion of a single sounding
    t_lsq = 2.0
    # SAMPLING, Assumed time, in seconds, for an McMC inversion of a single sounding
    t_mcmc = 10.0*60.0 

    total_lsq = np.array([nobs*t_lsq, nobs*t_lsq/Nproc_arr[-1]])
    total_mcmc = np.array([nobs*t_mcmc, nobs*t_mcmc/Nproc_arr[-1]])


    # loglog(T_total.T)
    plt.figure(figsize=(6,6))    
    plt.loglog(Nproc_arr, T_total.T, 'o-',  label=N_arr)
    plt.ylabel(r'Total time - $[s]$')
    plt.xlabel('Number of processors')
    plt.grid()
    total_lsq = np.array([nobs*t_lsq, nobs*t_lsq/Nproc_arr[-1]])
    plt.plot([Nproc_arr[0], Nproc_arr[-1]], total_lsq, 'k--', label='LSQ')
    plt.plot([Nproc_arr[0], Nproc_arr[-1]], total_mcmc, 'r--', label='MCMC')
    plt.legend(loc='upper right')
    plt.xticks(ticks=Nproc_arr, labels=[str(int(x)) for x in Nproc_arr])
    plt.ylim(1,1e+8)
    plt.tight_layout()
    plt.savefig('%s_total_sec' % file_out)
    plt.show()
    plt.close()

    plt.figure(figsize=(6,6)) 
    plt.loglog(N_arr, T_total, 'o-', label=[f'{int(x)}' for x in Nproc_arr])
    plt.ylabel(r'Total time - $[s]$')
    plt.xlabel('N-prior')
    plt.grid()
    plt.tight_layout()
    plt.plot([N_arr[0], N_arr[-1]], [nobs*t_lsq, nobs*t_lsq], 'k--', label='LSQ')
    plt.plot([N_arr[0], N_arr[-1]], [nobs*t_mcmc, nobs*t_mcmc], 'r--', label='MCMC')
    plt.legend(loc='upper left')
    #plt.xticks(ticks=N_arr, labels=[str(int(x)) for x in Nproc_arr])
    plt.ylim(1,1e+8)
    plt.savefig('%s_total_sec' % file_out)
    plt.show()
    plt.close()


    #### Plot timing results for forward modeling - GAAEM
    # Average timer per sounding 
    T_forward_sounding = T_forward/N_arr[:,np.newaxis]
    T_forward_sounding_per_sec = N_arr[:,np.newaxis]/T_forward
    T_forward_sounding_per_sec_per_cpu = T_forward_sounding_per_sec/Nproc_arr[np.newaxis,:]
    T_forward_sounding_speedup = T_forward_sounding_per_sec/T_forward_sounding_per_sec[0,0]

    ## Forward time per sounding - CPU
    plt.figure(figsize=(6,6))    
    plt.loglog(Nproc_arr, T_forward.T, 'o-', label='A')
    # plot dashed line indicating linear scaling
    for i in range(len(N_arr)):
        # Find index of first non-nan value in T_forward[i,:]
        try:
            idx = np.nonzero(~np.isnan(T_forward[i,:]))[0][0]
            plt.plot([Nproc_arr[0], Nproc_arr[-1]], [T_forward[i,idx]*Nproc_arr[idx]/Nproc_arr[0], T_forward[i,idx]*Nproc_arr[idx]/Nproc_arr[-1]], 'k--', 
                    label='Linear scaling', 
                    linewidth=0.5)   
        except:
            pass
    
    plt.ylabel(r'Forward time - $[s]$')
    plt.xlabel('Number of processors')
    plt.title('Forward calculation')
    plt.grid()
    plt.legend(N_arr, loc='upper left')
    plt.ylim(1e-1, 1e+4)
    plt.xlim(Nproc_arr[0], Nproc_arr[-1])
    plt.tight_layout()
    plt.savefig('%s_forward_sec_CPU' % file_out)
    plt.show()
    plt.close()

    ## Forward time per sounding - Nproc
    plt.figure(figsize=(6,6))    
    plt.loglog(N_arr, T_forward, 'o-', label='A')
    # plot dashed line indicating linear scaling
    for i in range(len(N_arr)):
        # Find index of first non-nan value in T_forward[i,:]
        try:
            idx = np.nonzero(~np.isnan(T_forward[i,:]))[0][0]
            ref_time = T_forward[i,idx]
            ref_N = N_arr[i]
            plt.plot([N_arr[0], N_arr[-1]], [ref_time*N_arr[0]/ref_N, ref_time*N_arr[-1]/ref_N], 'k--', label='Linear scaling', linewidth=0.5)   
        except:
            pass
    plt.ylabel(r'Forward time - $[s]$')
    plt.xlabel('Number of models')
    plt.title('Forward calculation')
    plt.grid()
    plt.legend(Nproc_arr, loc='upper left')
    #plt.ylim(1e-1, 1e+4)
    #plt.xlim(Nproc_arr[0], Nproc_arr[-1])
    plt.tight_layout()
    plt.savefig('%s_forward_sec_N' % file_out)
    plt.show()
    plt.close()


    #
    plt.figure(figsize=(6,6))    
    plt.plot(Nproc_arr, T_forward_sounding_per_sec.T, 'o-')
    # plot line 
    plt.ylabel(r'Forward computations per second - $[s^{-1}]$')
    plt.xlabel('Number of processors')
    plt.title('Forward calculation')
    plt.grid()
    plt.legend(N_arr)
    plt.tight_layout()
    plt.savefig('%s_forward_sounding_per_sec' % file_out)
    plt.show()
    plt.close()

    #
    plt.figure(figsize=(6,6))    
    plt.plot(Nproc_arr, T_forward_sounding_per_sec_per_cpu.T, 'o-')
    plt.ylabel('Forward computations per second per cpu')
    plt.xlabel('Number of processors')
    plt.title('Forward calculation')
    plt.grid()
    # Make yaxis start at 0
    plt.ylim(0, 80)    
    plt.xlim(Nproc_arr[0], Nproc_arr[-1])
    plt.legend(N_arr)
    plt.savefig('%s_forward_sounding_per_sec_per_cpu' % file_out)
    plt.show()
    plt.close()
    #

    plt.figure(figsize=(6,6))    
    plt.plot(Nproc_arr, T_forward_sounding_speedup.T, 'o-')
    # plot a line from 0,0 tp Nproc_arr[-1], Nproc_arr[-1]
    plt.plot([0, Nproc_arr[-1]], [0, Nproc_arr[-1]], 'k--')
    # set xlim to 1, Nproc_arr[-1]
    plt.xlim(.8, Nproc_arr[-1])
    plt.ylim(.8, Nproc_arr[-1])
    plt.ylabel('gatdaem - speedup compared to 1 processor')
    plt.xlabel('Number of processors')
    plt.grid()
    plt.legend(N_arr)
    plt.savefig('%s_forward_speedup' % file_out)
    plt.show()
    plt.close()



    ### STATS FOR REJECTION SAMPLING
    # Average timer per sounding
    T_rejection_sounding = T_rejection/N_arr[:,np.newaxis]
    T_rejection_sounding_per_sec = N_arr[:,np.newaxis]/T_rejection
    T_rejection_sounding_per_sec_per_cpu = T_rejection_sounding_per_sec/Nproc_arr[np.newaxis,:]
    T_rejection_sounding_speedup = T_rejection_sounding_per_sec/T_rejection_sounding_per_sec[0,0]
    T_rejection_sounding_speedup = T_rejection_sounding_per_sec*0

    T_rejection_per_data = nobs/T_rejection

    for i in range(len(N_arr)):
        # find index of first value in T_rejection_sounding_per_sec[i,:] that is not nan
        try:
            idx = np.where(~np.isnan(T_rejection_sounding_per_sec[i,:]))[0][0]
            T_rejection_sounding_speedup[i,:] = T_rejection_sounding_per_sec[i,:]/(T_rejection_sounding_per_sec[i,idx]/Nproc_arr[idx]) 
        except:
            T_rejection_sounding_speedup[i,:] = T_rejection_sounding_per_sec[i,:]*0


    ## Rejection total sec - per CPU
    plt.figure(figsize=(6,6))
    plt.loglog(Nproc_arr, T_rejection.T, 'o-')
    for i in range(len(N_arr)):
        # Find index of first non-nan value in T_forward[i,:]
        try:
            idx = np.nonzero(~np.isnan(T_rejection[i,:]))[0][0]
            plt.plot([Nproc_arr[0], Nproc_arr[-1]], [T_rejection[i,idx]*Nproc_arr[idx]/Nproc_arr[0], T_rejection[i,idx]*Nproc_arr[idx]/Nproc_arr[-1]], 'k--',
                        label='Linear scaling', 
                        linewidth=0.5)
        except:
            pass
    plt.ylabel('Rejection sampling - time $[s]$')
    plt.xlabel('Number of processors')
    plt.grid()
    plt.legend(N_arr)
    plt.tight_layout()
    plt.ylim(1e-1, 2e+3)
    plt.savefig('%s_rejection_sec_CPU' % file_out)
    plt.show()
    plt.close()


    ## Rejection total sec - per process
    plt.figure(figsize=(6,6))
    plt.loglog(N_arr, T_rejection, 'o-')
    for i in range(len(Nproc_arr)):
        # Find index of first non-nan value in T_forward[i,:]
        try:
            idx = np.nonzero(~np.isnan(T_rejection[:,i]))[0][0]
            ref_time = np.abs(T_rejection[idx,i])
            plt.plot([N_arr[0], N_arr[-1]], [ref_time*N_arr[0]/N_arr[idx], ref_time*N_arr[-1]/N_arr[idx]], 'k--',
                        label='Linear scaling', 
                        linewidth=0.5)
        except:
            pass
    plt.ylabel('Rejection sampling - time $[s]$')
    plt.xlabel('Lookup table size')
    plt.grid()
    plt.legend(Nproc_arr)
    plt.tight_layout()
    plt.ylim(1e-1, 2e+3)
    plt.savefig('%s_rejection_sec_N' % file_out)
    plt.show()
    plt.close()


    ## Rejection speedup
    plt.figure(figsize=(6,6))
    plt.plot(Nproc_arr, T_rejection_sounding_speedup.T, 'o-')
    # plot a line from 0,0 tp Nproc_arr[-1], Nproc_arr[-1]
    plt.plot([0, Nproc_arr[-1]], [0, Nproc_arr[-1]], 'k--')
    # set xlim to 1, Nproc_arr[-1]
    plt.xlim(.8, Nproc_arr[-1])
    plt.ylim(.8, Nproc_arr[-1])
    plt.ylabel('Rejection sampling - speedup compared to 1 processor')
    plt.xlabel('Number of processors')
    plt.grid()
    plt.legend(N_arr)
    plt.savefig('%s_rejection_speedup' % file_out)
    plt.show()
    plt.close()


    ## Rejection sound per sec
    plt.figure(figsize=(6,6))
    plt.loglog(Nproc_arr, T_rejection_per_data.T, 'o-', label=N_arr)
    plt.plot([Nproc_arr[0], Nproc_arr[-1]], [1./t_lsq, 1./t_lsq], 'k--', label='LSQ')
    plt.plot([Nproc_arr[0], Nproc_arr[-1]], [1./t_mcmc, 1./t_mcmc], 'r--', label='MCMC')
    plt.ylabel('Rejection sampling - number of soundings per second - $s^{-1}$')
    plt.xlabel('Number of processors')
    plt.grid()
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.ylim(1e-3, 1e+5)
    plt.savefig('%s_rejection_sounding_per_sec' % file_out)
    plt.show()
    plt.close()

    ## Rejection sec per sounding
    plt.figure(figsize=(6,6))
    plt.semilogy(Nproc_arr, 1./T_rejection_per_data.T, 'o-', label=N_arr)
    #plt.plot(Nproc_arr, 1./T_rejection_per_data.T, 'o-', label=N_arr)
    plt.plot([Nproc_arr[0], Nproc_arr[-1]], [t_lsq, t_lsq], 'k--', label='LSQ')
    plt.plot([Nproc_arr[0], Nproc_arr[-1]], [t_mcmc, t_mcmc], 'r--', label='MCMC')
    plt.ylabel('Rejection sampling - seconds per sounding - $s$')
    plt.xlabel('Number of processors')
    plt.grid()
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.ylim(1e-5, 1e+3)
    plt.savefig('%s_rejection_sec_per_sound' % file_out)
    plt.show()
    plt.close()

    ## Rejection sound per sec - N
    plt.figure(figsize=(6,6))
    plt.loglog(N_arr, T_rejection_sounding_per_sec, 'o-')
    #plt.ylim(0, 8000)
    plt.ylabel('Rejection sampling - Soundings per second')
    plt.xlabel('Lookup table size')
    plt.grid()
    plt.legend(Nproc_arr)
    plt.savefig('%s_rejection_sounding_per_sec_N' % file_out)
    plt.show()
    plt.close()

    ## Rejection sound per sec - per CPU
    plt.figure(figsize=(6,6))
    plt.loglog(Nproc_arr, T_rejection_sounding_per_sec.T, 'o-')
    #plt.ylim(0, 8000)
    plt.ylabel('Rejection sampling - Soundings per second')
    plt.xlabel('Number of processors')
    plt.grid()
    plt.legend(N_arr)
    plt.savefig('%s_rejection_sounding_per_sec_CPU' % file_out)
    plt.show()
    plt.close()

    ##  Sound per sec per CPU - N  
    plt.figure(figsize=(6,6))
    plt.loglog(N_arr, T_rejection_sounding_per_sec_per_cpu, 'o-')
    plt.plot([0, Nproc_arr[-1]], [0, Nproc_arr[-1]], 'k--')
    plt.xlim(90, 5000000*1.1)
    #plt.ylim(0, 8000)
    plt.ylabel('Rejection sampling - Soundings per second per cpu')
    plt.xlabel('Lookup table size')
    plt.grid()
    plt.legend(Nproc_arr)
    plt.savefig('%s_rejection_sounding_per_sec_per_cpu_N' % file_out)
    plt.show()
    plt.close()


    ##  Sound per sec per CPU - CPU 
    plt.figure(figsize=(6,6))
    plt.semilogx(Nproc_arr, T_rejection_sounding_per_sec_per_cpu.T, 'o-')
    plt.ylim([0, np.nanmax(T_rejection_sounding_per_sec_per_cpu.T)*1.1])
    plt.ylabel('Rejection sampling - Soundings per second per cpu')
    plt.xlabel('Number of processors')
    plt.grid()
    plt.legend(N_arr)
    plt.savefig('%s_rejection_sounding_per_sec_per_cpu_CPU' % file_out)
    plt.show()
    plt.close()



    #### STATS FOR POSTERIOR STATISTICS
    # Average timer per sounding
    T_poststat_sounding = T_poststat/N_arr[:,np.newaxis]
    T_poststat_sounding_per_sec = N_arr[:,np.newaxis]/T_poststat
    T_poststat_sounding_per_sec_per_cpu = T_poststat_sounding_per_sec/Nproc_arr[np.newaxis,:]
    T_poststat_sounding_speedup = T_poststat_sounding_per_sec/T_poststat_sounding_per_sec[0,0]

    plt.figure(figsize=(6,6))
    plt.plot(Nproc_arr, T_poststat_sounding_per_sec.T, 'o-')
    plt.ylabel('Posterior statistics - Soundings per second - $[s^{-1}]$')
    plt.xlabel('Number of processors')
    plt.grid()
    plt.legend(N_arr)
    plt.tight_layout()
    plt.savefig('%s_poststat_sounding_per_sec' % file_out)
    plt.show()
    plt.close()

    # plt.figure(figsize=(6,6))
    # plt.plot(Nproc_arr, T_poststat_sounding_speedup.T, 'o-')
    # # plot a line from 0,0 tp Nproc_arr[-1], Nproc_arr[-1]
    # plt.plot([0, Nproc_arr[-1]], [0, Nproc_arr[-1]], 'k--')
    # # set xlim to 1, Nproc_arr[-1]
    # plt.xlim(.8, Nproc_arr[-1])
    # plt.ylim(.8, Nproc_arr[-1])
    # plt.ylabel('Posterior statistics - speedup compared to 1 processor')
    # plt.xlabel('Number of processors')
    # plt.grid()
    # plt.legend(N_arr)
    # plt.savefig('%s_poststat_speedup' % file_out)

    #####
    # ## Plot Cumulative Time useage for min and max number of used cores

    i_proc = len(Nproc_arr)-1
    #i_proc= 0

    for i_proc in [0,len(Nproc_arr)-1]:

        T=[T_prior[:,i_proc], T_forward[:,i_proc], T_rejection[:,i_proc], T_poststat[:,i_proc]]

        ### %% Plor cumT as an area plot
        plt.figure(figsize=(6,6))
        plt.stackplot(N_arr, T, labels=['Prior', 'Forward', 'Rejection', 'PostStat'])
        plt.plot(N_arr, T_total[:, i_proc], 'k--')
        plt.xscale('log')
        #plt.yscale('log')
        plt.xlabel('$N_{lookup}$')
        plt.ylabel('Time [$s$]')
        plt.title('Cumulative time, using %d processors' % Nproc_arr[i_proc])
        plt.legend(loc='upper left')
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.savefig('%s_Ncpu%d_cumT' % (file_out,Nproc_arr[i_proc]))
        plt.show()
        plt.close()

        # The same as thea area plot but normalized to the total time
        plt.figure(figsize=(6,6))
        plt.stackplot(N_arr, T/np.sum(T, axis=0), labels=['Prior', 'Forward', 'Rejection', 'PostStat'])
        plt.xscale('log')
        plt.xlabel('$N_{lookup}$')
        plt.ylabel('Normalized time')
        plt.legend(loc='upper left')
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.title('Normalized time, using %d processors' % Nproc_arr[i_proc])
        plt.savefig('%s_Ncpu%d_cumT_norm' % (file_out,Nproc_arr[i_proc]))
        plt.show()
        plt.close()



