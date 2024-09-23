import h5py
import numpy as np
import os.path
import subprocess
from sys import exit
import multiprocessing
from multiprocessing import Pool
from multiprocessing import Pool
from multiprocessing import shared_memory
from functools import partial
import time
    
def is_notebook():
    """
    Check if the code is running in a Jupyter notebook or IPython shell.

    Returns:
        bool: True if running in a Jupyter notebook or IPython shell, False otherwise.
    """
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
            print(name)
            if name.upper().startswith('M'):
                # Check if the attribute 'is_discrete' exists
                if 'x' in dataset.attrs:
                    pass
                else:
                    if 'z' in dataset.attrs:
                        dataset.attrs['x'] = dataset.attrs['z']
                    else:
                        if 'M1' in f.keys():
                            if 'x' in f['/M1'].attrs.keys():
                                f[name].attrs['x'] = f['/M1'].attrs['x']
                                print('Setting %s/x = /M1/x ' % name)
                            else:
                                print('No x attribute found in %s' % name)    
                
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


def integrate_posterior_stats(f_post_h5='DJURSLAND_P01_N0100000_NB-13_NR03_POST_Nu1000_aT1.h5', **kwargs):
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
    usePrior = kwargs.get('usePrior', False)

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
                    ir = np.int64(i_use[iid,:])
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

    return f_prior_h5


'''
Forward simulation
'''

def forward_gaaem(C=np.array(()), thickness=np.array(()), GEX={}, file_gex='', stmfiles=[], showtime=False, **kwargs):
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


def forward_gaaem_chunk(C_chunk, thickness, stmfiles, file_gex, Nhank, Nfreq, **kwargs):
    """
    Perform forward modeling using the GAAEM method on a chunk of data.

    :param C_chunk: The chunk of data to be processed.
    :type C_chunk: numpy.ndarray
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
    # pause for random time
    # time.sleep(np.random.rand()*10)
    return forward_gaaem(C=C_chunk, thickness=thickness, stmfiles=stmfiles, file_gex=file_gex, Nhank=Nhank, Nfreq=Nfreq, parallel=False, **kwargs)


# %% PRIOR DATA GENERATORS

def prior_data_gaaem(f_prior_h5, file_gex, N=0, doMakePriorCopy=True, im=1, id=1, Nhank=280, Nfreq=12, parallel=True, **kwargs):
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
    :param kwargs: Additional keyword arguments.
    :type kwargs: dict

    :return: Filename of the HDF5 file containing the updated prior data.
    :rtype: str
    """
    import integrate as ig

    type = 'TDEM'
    method = 'ga-aem'
    showInfo = kwargs.get('showInfo', 0)
    Nproc = kwargs.get('Nproc', 0)

    # Force open/close of hdf5 file
    with h5py.File(f_prior_h5, 'r') as f:
        # open and close
        pass

    with h5py.File(f_prior_h5, 'a') as f:
        N_in = f['M1'].shape[0]
    if N==0: 
        N = N_in     
    if N>N_in:
        N=N_in

    if not os.path.isfile(file_gex):
        print("ERRROR: file_gex=%s does not exist in the current folder." % file_gex)

    print('N=%d, N_in=%d' % (N,N_in))
    if doMakePriorCopy:
        if N < N_in:
            f_prior_data_h5 = '%s_%s_N%d_Nh%d_Nf%d.h5' % (os.path.splitext(f_prior_h5)[0], os.path.splitext(file_gex)[0], N, Nhank, Nfreq)
        else:
            f_prior_data_h5 = '%s_%s_Nh%d_Nf%d.h5' % (os.path.splitext(f_prior_h5)[0], os.path.splitext(file_gex)[0], Nhank, Nfreq)
        if (showInfo>-1):
            print("Creating a copy of %s as %s" % (f_prior_h5, f_prior_data_h5))
        # make a copy of the prior file
        #copy_hdf5_file(input_filename, output_filename, N=None)
        ig.copy_hdf5_file(f_prior_h5, f_prior_data_h5,N)
        #if N < N_in:
        #    # Truncate
        #    truncate_and_repack_hdf5_file(f_prior_data_h5, N)
            
        
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
        
        # Use spawn context for cross-platform compatibility
        #ctx = multiprocessing.get_context("spawn")
        #with ctx.Pool(processes=Nproc) as p:
        #    D_chunks = p.map(forward_gaaem_chunk_partial, C_chunks)
        
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
        if (showInfo>-1):
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
def prior_model_layered(lay_dist='uniform', dz = 1, z_max = 90, NLAY_min=3, NLAY_max=6, NLAY_deg=6, RHO_dist='log-uniform', RHO_min=0.1, RHO_max=100, RHO_MEAN=100, RHO_std=80, N=100000):
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

    :return: Filepath of the saved prior model.
    :rtype: str
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
        f_prior_h5 = 'PRIOR_UNIFORM_NL_%d-%d_%s_N%d.h5' % (NLAY_min, NLAY_max, RHO_dist, N)

    elif lay_dist == 'chi2':
        NLAY = np.random.chisquare(NLAY_deg, N)
        NLAY = np.ceil(NLAY).astype(int)    
        f_prior_h5 = 'PRIOR_CHI2_NF_%d_%s_N%d.h5' % (NLAY_deg, RHO_dist, N)

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

def prior_model_workbench(N=100000, RHO_dist='log-uniform', z1=0, z_max= 100, nlayers=30, p=2, RHO_min = 1, RHO_max= 300, RHO_mean=180, RHO_std=80, chi2_deg= 100):
    """
    Generate a prior model with increasingly thick layers
 
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

    :return: Filepath of the saved prior model.
    :rtype: str
    """

    f_prior_h5 = 'PRIOR_WB%d_N%d_%s' % (nlayers,N,RHO_dist)

    z2=z_max
    z= z1 + (z2 - z1) * np.linspace(0, 1, nlayers) ** p

    nz = len(z)
    
    if RHO_dist=='uniform':
        M_rho = np.random.uniform(low=RHO_min, high = RHO_max, size=(N, nz))
        f_prior_h5 = '%s_R%g_%g.h5' % (f_prior_h5, RHO_min, RHO_max)
    elif RHO_dist=='log-uniform':
        M_rho = np.exp(np.random.uniform(low=np.log(RHO_min), high = np.log(RHO_max), size=(N, nz)))
        f_prior_h5 = '%s_R%g_%g.h5' % (f_prior_h5, RHO_min, RHO_max)
    elif RHO_dist=='normal':
        M_rho = np.random.normal(loc=RHO_mean, scale = RHO_std, size=(N, nz))
        f_prior_h5 = '%s_R%g_%g.h5' % (f_prior_h5, RHO_mean, RHO_std)
    elif RHO_dist=='log-normal':
        M_rho = np.random.lognormal(mean=np.log(RHO_mean), sigma = RHO_std/RHO_mean, size=(N, nz))
        f_prior_h5 = '%s_R%g_%g.h5' % (f_prior_h5, RHO_mean, RHO_std)
    elif RHO_dist=='chi2':
        M_rho = np.random.chisquare(df = chi2_deg, size=(N, nz))
        f_prior_h5 = '%s_deg%d.h5' % (f_prior_h5,chi2_deg)

    #f_prior_h5 = f_prior_h5 + '.h5'

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


'''
THIS IS THE NEW MULTI DATA IMPLEMENTATION
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
    import integrate as ig

    # get optional arguments
    showInfo = kwargs.get('showInfo', 0)
    useRandomData = kwargs.get('useRandomData', True)
    #useRandomData = kwargs.get('useRandomData', False)
    
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
    T_all = np.zeros(nump)*np.nan
    EV_all = np.zeros(nump)*np.nan
    
    
    with h5py.File(f_prior_h5, 'r') as f_prior:
        N = f_prior['/D1'].shape[0]

    
    if N_use>N:
        N_use = N

    if N_use<N:  
    #    #np.random.seed(0)
        idx = np.sort(np.random.choice(N, N_use, replace=False))
    #    #idx = np.sort(np.arange(N_use))
    else:
        idx = np.arange(N)
    

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
    with h5py.File(f_prior_h5, 'r') as f_prior:
        for id in id_use:
            DS = '/D%d' % id
            N = f_prior[DS].shape[0]
            #print('Reading %s' % DS)
            if N_use<N:
                if useRandomData:
                    #print('Start Reading random subset of %s ' % DS)
                    # NEXT LINE IE EXTREMELY
                    #Dsub = f_prior[DS][idx]
                    # NEXT TWO LINES ARE MUCH FASTER!!!
                    Dsub = f_prior[DS][:]
                    Dsub = Dsub[idx]
                    #print('End Reading random subset of %s ' % DS)
                else:
                    Dsub = f_prior[DS][0:N_use]
                    # Read yje lasy N_use values from DS
                    #Dsub = f_prior[DS][N-N_use:]
                D.append(Dsub)
            else:        
                D.append(f_prior[DS][:])

            #print(D[-1].shape)

    # THIS IS THE ACTUAL INVERSION!!!!
    for j in tqdm(range(len(ip_range)), miniters=10):
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
                    d_std = f_data['%s/d_std' % DS][ip] 

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
        
        if useRandomData:
            # get the correct index of the subset used
            i_use = idx[i_use]
            

        # Unfortunately this code originally used matlab style codeing for i_use, 
        # this we need to add 1 to the index
        #i_use = i_use+1            

        t.append(time.time()-t0)        

    
        # find the number of unique indexes
        n_unique = len(np.unique(i_use))


        # Compute the evidence
        maxlogL = np.nanmax(L)
        exp_logL = np.exp(L - maxlogL)
        EV = maxlogL + np.log(np.nansum(exp_logL)/len(L))

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


        if showInfo>1:
            for i in range(len(t)):
                if i<len(D):
                    print(' Time id%d: %f - %s' % (i,t[i],noise_model[i]))
                else:
                    print(' Time id%d, sampling: %f' % (i,t[i]))
            print('Time total: %f' % np.sum(t))
        
    return i_use_all, T_all, EV_all, ip_range



def integrate_rejection(f_prior_h5='prior.h5', 
                              f_data_h5='DAUGAAD_AVG_inout.h5',
                              f_post_h5='',                              
                              N_use=100000000000, 
                              id_use=[1], 
                              ip_range=[], 
                              nr=400,
                              autoT=1,
                              T_base = 1,
                              Nchunks=0,
                              Ncpu=0,
                              parallel=True,                              
                              **kwargs):
    from datetime import datetime   
    from multiprocessing import Pool
    import multiprocessing
    import integrate as ig
    import numpy as np
    import h5py

    # get optional arguments
    showInfo = kwargs.get('showInfo', 1)
    # If set, Nproc will be used as the number of processors
    Nproc = kwargs.get('Nproc', -1)
    if Nproc>-1:
        Ncpu = Nproc

    updatePostStat = kwargs.get('updatePostStat', True)

    # Set default f_post_h5 filename if not set    
    if len(f_post_h5)==0:
        f_post_h5 = "POST_%s_%s_Nu%d_aT%d.h5" % (os.path.splitext(f_data_h5)[0],os.path.splitext(f_prior_h5)[0],N_use,autoT)

    # Check that f_post_h5 allready exists, and warn the user   
    if os.path.isfile(f_post_h5):
        print('File %s allready exists' % f_post_h5)
        print('Overwriting...')    
        
    
    # Get sample size N from f_prior_h5
    with h5py.File(f_prior_h5, 'r') as f_prior:
        N = f_prior['/D1'].shape[0]

    if N_use>N:
        N_use = N

    # Get number of data points from, f_data_h5
    with h5py.File(f_data_h5, 'r') as f_data:
        Ndp = f_data['/D1/d_obs'].shape[0]
    # if ip_range is empty then use all data points
    if len(ip_range)==0:
        ip_range = np.arange(Ndp)
    Ndp_invert = len(ip_range)
            
    print('Ncpu=%d' % Ncpu)        
    if Ncpu < 1 :
        Ncpu =  int(multiprocessing.cpu_count()/2)
        
    # Split the ip_range into Nchunks
    if Nchunks==0:
        if parallel:
            Nchunks = Ncpu
        else:   
            Nchunks = 1
    if Ncpu ==1:
        parallel = False

    ip_chunks = np.array_split(ip_range, Nchunks) 

    if showInfo>0:
        print('Number of data points: %d (available), %d (used). Nchunks=%s, Ncpu=%d' % (Ndp,Ndp_invert,Nchunks,Ncpu))    
    
    if showInfo>2:
        print('f_prior_h5=%s\nf_data_h5=%s\nf_post_h5=%s' % (f_prior_h5, f_data_h5, f_post_h5))
        print('N_use = %d' % (N_use))
        print('Ncpu = %d\nNchunks=%d' % (Ncpu, Nchunks))
    
        return 1
    
    
    # set i_use_all to be a 2d Matrie of size (nump,nr) of random integers in range(N)
    i_use_all = np.random.randint(0, N, (Ndp, nr))
    T_all = np.zeros(Ndp)*np.nan
    EV_all = np.zeros(Ndp)*np.nan

    date_start = str(datetime.now())
    t_start = datetime.now()
    
    # PERFORM INVERSION PERHAPS IN PARALLEL

    if parallel:
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

    if updatePostStat:
        ig.integrate_posterior_stats(f_post_h5, **kwargs)

    #return f_post_h5 T_all, EV_all, i_use_all
    return f_post_h5


def integrate_posterior_chunk(args):
    import integrate as ig
    
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
    import integrate as ig
    
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


# %%
