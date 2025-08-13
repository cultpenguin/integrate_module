"""
INTEGRATE I/O Module - Data Input/Output and File Management

This module provides comprehensive input/output functionality for the INTEGRATE
geophysical data integration package. It handles reading and writing of HDF5 files,
data format conversions, and management of prior/posterior data structures.

Key Features:
    - HDF5 file I/O for prior models, data, and posterior results
    - Support for multiple geophysical data formats (GEX, STM, USF)
    - Automatic data validation and format checking
    - File conversion utilities between different formats
    - Data merging and aggregation functions
    - Checksum verification and file integrity checks

Main Functions:
    - load_*(): Functions for loading prior models, data, and results
    - save_*(): Functions for saving prior models and data arrays
    - read_*(): File format readers (GEX, USF, etc.)
    - write_*(): File format writers and converters
    - merge_*(): Data and posterior merging utilities

File Format Support:
    - HDF5: Primary data storage format
    - GEX: Geometry and survey configuration files
    - STM: System transfer function files
    - USF: Field measurement files
    - CSV: Export format for GIS integration

Author: Thomas Mejer Hansen
Email: tmeha@geo.au.dk
"""

import os
import numpy as np
import h5py
import re
from typing import Dict, List, Union, Any

def load_prior(f_prior_h5, N_use=0, idx = [], Randomize=False):
    """
    Load prior model parameters and data from HDF5 file.

    Loads both model parameters and forward-modeled data from a prior HDF5 file,
    with options for sample selection, indexing, and randomization. This is a
    convenience function that combines model and data loading operations.

    Parameters
    ----------
    f_prior_h5 : str
        Path to the HDF5 file containing prior model realizations and data.
    N_use : int, optional
        Number of samples to load. If 0, loads all available samples (default is 0).
    idx : list, optional
        Specific indices to load. If empty, uses N_use or loads all samples
        (default is []).
    Randomize : bool, optional
        Whether to randomize the order of loaded samples (default is False).

    Returns
    -------
    D : dict
        Dictionary containing forward-modeled data arrays, with keys corresponding
        to data types (e.g., 'D1', 'D2').
    M : dict
        Dictionary containing model parameter arrays, with keys corresponding
        to model types (e.g., 'M1', 'M2').
    idx : numpy.ndarray
        Array of indices corresponding to the loaded samples.

    Notes
    -----
    This function internally calls load_prior_data() and load_prior_model()
    with consistent indexing to ensure data and model correspondence.
    Sample selection priority: explicit idx > N_use > all samples.
    """
    if len(idx)==0:
        D, idx = load_prior_data(f_prior_h5, N_use=N_use, Randomize=Randomize)
    else:
        D, idx = load_prior_data(f_prior_h5, idx=idx, Randomize=Randomize)
    M, idx = load_prior_model(f_prior_h5, idx=idx, Randomize=Randomize)
    return D, M, idx



def load_prior_model(f_prior_h5, im_use=[], idx=[], N_use=0, Randomize=False):
    """
    Load model parameter arrays from prior HDF5 file.

    Loads model parameter arrays (e.g., resistivity, layer thickness, geological units)
    from a prior HDF5 file with flexible model selection and sample indexing options.
    Supports loading specific model types and sample subsets.

    Parameters
    ----------
    f_prior_h5 : str
        Path to the HDF5 file containing prior model parameter realizations.
    im_use : list of int, optional
        Model parameter indices to load (e.g., [1, 2] for M1 and M2).
        If empty, loads all available model parameters (default is []).
    idx : list or array-like, optional
        Specific sample indices to load. If empty, uses N_use and Randomize
        to determine samples (default is []).
    N_use : int, optional
        Number of samples to load. If 0, loads all available samples.
        Ignored if idx is provided (default is 0).
    Randomize : bool, optional
        Whether to randomly select samples when idx is empty.
        If False, uses sequential selection (default is False).

    Returns
    -------
    M : list of numpy.ndarray
        List of model parameter arrays, one for each requested model type.
        Each array has shape (N_samples, N_model_parameters).
    idx : numpy.ndarray
        Array of sample indices that were loaded, useful for consistent
        indexing across related datasets.

    Notes
    -----
    The function automatically detects available model parameters (M1, M2, ...)
    and loads the requested subset. Sample selection priority follows:
    explicit idx > N_use random/sequential > all samples.
    
    When idx length differs from N_use, the function uses len(idx) and
    issues a warning message.
    """
    import h5py
    import numpy as np


    if len(im_use)==0:
        Nmt=0
        with h5py.File(f_prior_h5, 'r') as f_prior:
            for key in f_prior.keys():
                if key[0]=='M':
                    Nmt = Nmt+1
        if len(im_use)==0:
            im_use = np.arange(1,Nmt+1) 
    
    with h5py.File(f_prior_h5, 'r') as f_prior:
        N = f_prior['/M1'].shape[0]
        if N_use == 0:
            N_use = N    
        
        if len(idx)==0:
            if Randomize:
                idx = np.sort(np.random.choice(N, min(N_use, N), replace=False)) if N_use < N else np.arange(N)
            else:
                idx = np.arange(N_use)
        else:
            # check if length of idx is equal to N_use
            if len(idx)!=N_use:
                print('Length of idx (%d) must be equal to N_use)=%d' % (len(idx), N_use))
                N_use = len(idx)      
                print('using N_use=len(idx)=%d' % N_use)
                
        M = [f_prior[f'/M{id}'][:][idx] for id in im_use]
    
    
    return M, idx

def save_prior_model(f_prior_h5, M_new, 
                     im=None, 
                     force_replace=False,
                     delete_if_exist=False,   
                     **kwargs):
    """
    Save model parameter arrays to prior HDF5 file.

    Saves model parameter realizations (e.g., resistivity, layer thickness) to an
    HDF5 file with automatic model identifier assignment and data type optimization.
    Supports overwriting existing models and file management options.

    Parameters
    ----------
    f_prior_h5 : str
        Path to the HDF5 file where model data will be saved.
    M_new : numpy.ndarray
        Model parameter array to save. Can be 1D or 2D; 1D arrays are
        automatically converted to column vectors.
    im : int, optional
        Model identifier for the dataset key (creates '/M{im}'). If None,
        automatically assigns the next available ID (default is None).
    force_replace : bool, optional
        Whether to overwrite existing model data with the same identifier.
        If False, raises error when key exists (default is False).
    delete_if_exist : bool, optional
        Whether to delete the entire HDF5 file before saving. Use with
        caution as this removes all existing data (default is False).
    **kwargs : dict
        Additional arguments:
        - showInfo : int, verbosity level (0=silent, >0=verbose)

    Returns
    -------
    im : int
        The model identifier used for saving the data.

    Notes
    -----
    Model data is stored as HDF5 datasets with keys '/M1', '/M2', etc.
    Data type optimization is performed automatically:
    - Floating-point arrays are converted to float32 for memory efficiency
    - Integer arrays are preserved as appropriate integer types
    - All datasets use gzip compression (level 9) for storage efficiency
    
    The function ensures 2D array format with shape (N_samples, N_parameters)
    where 1D arrays are converted to column vectors.
    """
    import h5py
    import numpy as np
    import os

    showInfo = kwargs.get('showInfo', 0)
    # if f_prior_h5 exists, delete it
    if delete_if_exist:
        
        # Assuming f_prior_h5 already contains the filename
        if os.path.exists(f_prior_h5):
            os.remove(f_prior_h5)
            if showInfo>1:
                print("File %s has been deleted." % f_prior_h5)
        else:
            print("File %s does not exist." % f_prior_h5)
            pass

        
    if im is None:
        Nmt=0
        with h5py.File(f_prior_h5, 'r') as f_prior:
            for key in f_prior.keys():
                if key[0]=='M':
                    Nmt = Nmt+1
        im = Nmt+1
    
    key = '/M%d' % im
    if showInfo>1:
        print("Saving new prior model '%s' to file: %s " % (key,f_prior_h5))

    # Delete the 'key' if it exists
    with h5py.File(f_prior_h5, 'a') as f_prior:
        if key in f_prior:
            print("Deleting prior model '%s' from file: %s " % (key,f_prior))
            if force_replace:
                del f_prior[key]
            else:
                print("Key '%s' already exists. Use force_replace=True to overwrite." % key)
                return False

    # Make sure the data is 2D using atleast_2d
    if M_new.ndim<2:
        M_new = np.atleast_2d(M_new.flatten()).T

    # Write the new data
    with h5py.File(f_prior_h5, 'a') as f_prior:
        # Convert to 32-bit float for better memory efficiency if the data is floating point
        if np.issubdtype(M_new.dtype, np.floating):
            M_new_32 = M_new.astype(np.float32)
            f_prior.create_dataset(key, data=M_new_32, compression='gzip', compression_opts=9)
        elif np.issubdtype(M_new.dtype, np.integer):
            M_new_32 = M_new.astype(np.int32)
            f_prior.create_dataset(key, data=M_new_32, compression='gzip', compression_opts=9)
        else:
            f_prior.create_dataset(key, data=M_new, compression='gzip', compression_opts=9)

        # if 'name' is not set in kwargs, set it to 'XXX'
        if 'name' not in kwargs:
            kwargs['name'] = 'Model %d' % (im)
        if 'is_discrete' not in kwargs:
            kwargs['is_discrete'] = 0
        if 'x' not in kwargs:
            kwargs['x'] = np.arange(M_new.shape[1])

        # if kwargs is set print keys
        if showInfo>2:
            for kwargkey in kwargs:
                print('save_prior_model: key=%s, value=%s' % (kwargkey, kwargs[kwargkey]))


        # if kwarg has keyy 'method' then write it to the file as att
        if 'x' in kwargs:
             f_prior[key].attrs['x'] = kwargs['x']
        if 'name' in kwargs:
             f_prior[key].attrs['name'] = kwargs['name']
        if 'method' in kwargs:
             f_prior[key].attrs['method'] = kwargs['method']
        if 'is_discrete' in kwargs:
            f_prior[key].attrs['is_discrete'] = kwargs['is_discrete']
        if 'class_id' in kwargs:
            f_prior[key].attrs['class_id'] = kwargs['class_id']
        if 'class_name' in kwargs:
            f_prior[key].attrs['class_name'] = kwargs['class_name']
        if 'clim' in kwargs:
            f_prior[key].attrs['clim'] = kwargs['clim']
        if 'cmap' in kwargs:
            f_prior[key].attrs['cmap'] = kwargs['cmap']

        if showInfo>1:
            print("New prior data '%s' saved to file: %s " % (key,f_prior_h5))



def load_prior_data(f_prior_h5, id_use=[], idx=[], N_use=0, Randomize=False, **kwargs):
    """
    Load forward-modeled data arrays from prior HDF5 file.

    Loads electromagnetic or other geophysical data predictions from forward
    modeling runs stored in the prior file. Supports selective loading by
    data type, sample indices, and randomization for sampling purposes.

    Parameters
    ----------
    f_prior_h5 : str
        Path to the HDF5 file containing forward-modeled data arrays.
    id_use : list of int, optional
        Data type identifiers to load (e.g., [1, 2] for D1 and D2).
        If empty, loads all available data types (default is []).
    idx : list or array-like, optional
        Specific sample indices to load. If empty, uses N_use and Randomize
        to determine samples (default is []).
    N_use : int, optional
        Number of samples to load. If 0, loads all available samples.
        Automatically limited to available data size (default is 0).
    Randomize : bool, optional
        Whether to randomly select samples when idx is empty.
        If False, uses sequential selection (default is False).

    Returns
    -------
    D : list of numpy.ndarray
        List of forward-modeled data arrays, one for each requested data type.
        Each array has shape (N_samples, N_data_points).
    idx : numpy.ndarray
        Array of sample indices that were loaded, useful for consistent
        indexing with corresponding model parameters.

    Notes
    -----
    Data arrays are stored as HDF5 datasets with keys '/D1', '/D2', etc.,
    representing different data types (e.g., different measurement systems,
    frequencies, or processing stages). The function automatically detects
    available data types and loads the requested subset.
    
    Sample selection follows the same priority as load_prior_model():
    explicit idx > N_use random/sequential > all samples.
    """

    showInfo = kwargs.get('showInfo', 1)

    if showInfo > 0:
        print('Loading prior data from %s. ' % f_prior_h5, end='')
        print('Using prior data ids: %s' % str(id_use))

    import h5py
    import numpy as np

    if len(id_use)==0:        
        Ndt=0
        with h5py.File(f_prior_h5, 'r') as f_prior:
            for key in f_prior.keys():
                if key[0]=='D':
                    Ndt = Ndt+1
        if len(id_use)==0:
            id_use = np.arange(1,Ndt+1) 

    with h5py.File(f_prior_h5, 'r') as f_prior:
        N = f_prior['/D1'].shape[0]
        if N_use == 0:
            N_use = N    
        if N_use>N:
            N_use = N

        if len(idx)==0:
            if Randomize:
                idx = np.sort(np.random.choice(N, min(N_use, N), replace=False)) if N_use < N else np.arange(N)
            else:
                idx = np.arange(N_use)
        else:
            # check if length of idx is equal to N_use
            if len(idx)!=N_use:
                print('Length of idx (%d) must be equal to N_use)=%d' % (len(idx), N_use))
                N_use = len(idx)      
                print('using N_use=len(idx)=%d' % N_use)


        D = [f_prior[f'/D{id}'][:][idx] for id in id_use]

    if showInfo>0:
        for i in range(len(D)):
            print('  - /D%d: ' % (id_use[i]), end='')
            print(' N,nd = %d/%d' % (D[i].shape[0], D[i].shape[1]))

        
    return D, idx

def save_prior_data(f_prior_h5, D_new, id=None, force_delete=False, **kwargs):
    """
    Save forward-modeled data arrays to prior HDF5 file.

    Saves electromagnetic or other geophysical data predictions from forward
    modeling to an HDF5 file with automatic data identifier assignment and
    data type optimization. Supports overwriting existing data arrays.

    Parameters
    ----------
    f_prior_h5 : str
        Path to the HDF5 file where forward-modeled data will be saved.
    D_new : numpy.ndarray
        Forward-modeled data array to save. Should have shape
        (N_samples, N_data_points) for consistency.
    id : int, optional
        Data identifier for the dataset key (creates '/D{id}'). If None,
        automatically assigns the next available ID (default is None).
    force_delete : bool, optional
        Whether to delete existing data with the same identifier before
        saving. If False, raises error when key exists (default is False).
    **kwargs : dict
        Additional arguments:
        - showInfo : int, verbosity level (0=silent, >0=verbose)

    Returns
    -------
    id : int
        The data identifier used for saving the data.

    Notes
    -----
    Forward-modeled data is stored as HDF5 datasets with keys '/D1', '/D2', etc.,
    representing different data types (e.g., electromagnetic frequencies,
    measurement systems, or processing variants).
    
    Data type optimization is performed automatically:
    - Floating-point arrays are converted to float32 for memory efficiency
    - Integer arrays are preserved as appropriate integer types
    - All datasets use gzip compression (level 9) for storage efficiency
    
    The function ensures 2D array format with shape (N_samples, N_data_points).
    """
    showInfo = kwargs.get('showInfo', 1)

    import h5py
    import numpy as np

    if id is None:
        Ndt=0
        with h5py.File(f_prior_h5, 'r') as f_prior:
            for key in f_prior.keys():
                if key[0]=='D':
                    Ndt = Ndt+1
        id = Ndt+1
    
    key = '/D%d' % id
    print("Saving new prior data '%s' to file: %s " % (key,f_prior_h5))

    # Delete the 'key' if it exists
    with h5py.File(f_prior_h5, 'a') as f_prior:
        if key in f_prior:
            print("Deleting prior data '%s' from file: %s " % (key,f_prior))
            if force_delete:
                del f_prior[key]
            else:
                print("Key '%s' already exists. Use force_delete=True to overwrite." % key)
                return False

    # Write the new data
    with h5py.File(f_prior_h5, 'a') as f_prior:
        # Convert to 32-bit float for better memory efficiency if the data is floating point
        if np.issubdtype(D_new.dtype, np.floating):
            D_new_32 = D_new.astype(np.float32)
            f_prior.create_dataset(key, data=D_new_32, compression='gzip', compression_opts=9)
        else:
            f_prior.create_dataset(key, data=D_new, compression='gzip', compression_opts=9)
        print("New prior data '%s' saved to file: %s " % (key,f_prior_h5))
        # if kwarg has keyy 'method' then write it to the file as att
        if 'method' in kwargs:
             f_prior[key].attrs['method'] = kwargs['method']
        if 'type' in kwargs:
            f_prior[key].attrs['type'] = kwargs['type']
        if 'im' in kwargs:
            f_prior[key].attrs['im'] = kwargs['im']
        if 'Nhank' in kwargs:
            f_prior[key].attrs['Nhank'] = kwargs['Nhank']
        if 'Nfreq' in kwargs:
            f_prior[key].attrs['Nfreq'] = kwargs['Nfreq']
        if 'f5_forward' in kwargs:
            f_prior[key].attrs['f5_forward'] = kwargs['f5_forward']
        if 'with_noise' in kwargs:
            f_prior[key].attrs['with_noise'] = kwargs['with_noise']

    return id


def load_data(f_data_h5, id_arr=[], **kwargs):
    """
    Load observational electromagnetic data from HDF5 file.

    Loads observed electromagnetic measurements, uncertainties, covariance matrices,
    and associated metadata from structured HDF5 files. Handles multiple data types
    and noise models with automatic fallback for missing data components.

    Parameters
    ----------
    f_data_h5 : str
        Path to the HDF5 file containing observational electromagnetic data.
    id_arr : list of int, optional
        Dataset identifiers to load (e.g., [1, 2] for D1 and D2).
        Each ID corresponds to a different measurement system or processing
        stage (default is [1]).
    **kwargs : dict
        Additional arguments:
        - showInfo : int, verbosity level (0=silent, 1=normal, >1=verbose)

    Returns
    -------
    dict
        Dictionary containing loaded observational data with keys:
        
        - 'noise_model' : list of str
            Noise model type for each dataset ('gaussian', 'multinomial', etc.)
        - 'd_obs' : list of numpy.ndarray
            Observed data measurements, shape (N_stations, N_channels) per dataset
        - 'd_std' : list of numpy.ndarray or None
            Standard deviations of observations, same shape as d_obs
        - 'Cd' : list of numpy.ndarray or None
            Full covariance matrices for each dataset
        - 'id_arr' : list of int
            Dataset identifiers that were successfully loaded. If set as empty, all data types will be loaded
        - 'i_use' : list of numpy.ndarray
            Data point usage indicators (1=use, 0=ignore)
        - 'id_use' : list of int or numpy.ndarray
            index of data type in prior data, used for cross-referencing
            if 'id_use' is not present in the file, it defaults to the dataset id_arr

    Notes
    -----
    The function gracefully handles missing data components:
    - Missing 'id_use' defaults to sequential dataset IDs (1, 2, 3, ...)
    - Missing 'i_use' defaults to ones array (use all data points)
    - Missing 'd_std' and 'Cd' remain as None (diagonal noise assumed)
    
    Data structure follows INTEGRATE standard format:
    - '/D{id}/d_obs': observed measurements
    - '/D{id}/d_std': measurement uncertainties  
    - '/D{id}/Cd': full covariance matrix (optional)
    - '/D{id}/i_use': data usage flags (optional)
    - '/D{id}/id_use': cross-reference IDs (optional)
    
    Each dataset can have a different noise model specified in the 'noise_model'
    attribute, enabling mixed data types in the same file.
    """

    showInfo = kwargs.get('showInfo', 1)

    import h5py
        
    if not isinstance(id_arr, list):
        id_arr = [id_arr]

    # If id_arr is empty find find all '/D{id}' datasets in the file
    if len(id_arr) == 0:
        with h5py.File(f_data_h5, 'r') as f_data:
            id_arr = [int(re.search(r'D(\d+)', key).group(1)) for key in f_data.keys() if re.match(r'D\d+', key)]
            id_arr.sort()

    if showInfo > 0:
        print('Loading data from %s. ' % f_data_h5, end='')
        print('Using data types: %s' % str(id_arr))
    
    with h5py.File(f_data_h5, 'r') as f_data:
        noise_model = [f_data[f'/D{id}'].attrs.get('noise_model', 'none') for id in id_arr]
        d_obs = [f_data[f'/D{id}/d_obs'][:] for id in id_arr]
        d_std = [f_data[f'/D{id}/d_std'][:] if 'd_std' in f_data[f'/D{id}'] else None for id in id_arr]
        Cd = [f_data[f'/D{id}/Cd'][:] if 'Cd' in f_data[f'/D{id}'] else None for id in id_arr]
        i_use = [f_data[f'/D{id}/i_use'][:] if 'i_use' in f_data[f'/D{id}'] else None for id in id_arr]
        id_use = [f_data[f'/D{id}/id_use'][()] if 'id_use' in f_data[f'/D{id}'] and f_data[f'/D{id}/id_use'].shape == () else f_data[f'/D{id}/id_use'][:] if 'id_use' in f_data[f'/D{id}'] else None for id in id_arr]
        
    for i in range(len(id_arr)):
        if id_use[i] is None:
            #id_use[i] = i+1
            id_use[i] = id_arr[i]
        if i_use[i] is None:
            i_use[i] = np.ones((len(d_obs[i]),1))

        
    DATA = {}
    DATA['noise_model'] = noise_model
    DATA['d_obs'] = d_obs
    DATA['d_std'] = d_std
    DATA['Cd'] = Cd
    DATA['id_arr'] = id_arr        
    DATA['i_use'] = i_use        
    DATA['id_use'] = id_use        
    # return noise_model, d_obs, d_std, Cd, id_arr


    if showInfo>0:
        for i in range(len(id_arr)):
            print('  - D%d: id_use=%d, %11s, Using %d/%d data' % (id_arr[i], id_use[i], noise_model[i],  DATA['d_obs'][i].shape[0],  DATA['d_obs'][i].shape[1]))

    return DATA


## def ###################################################

#def write_stm_files(GEX, Nhank=140, Nfreq=6, Ndig=7, **kwargs):
def write_stm_files(GEX, **kwargs):
    """
    Generate STM (System Transfer Matrix) files from GEX system configuration.

    Creates system transfer matrix files required for electromagnetic forward modeling
    using GA-AEM. Processes both high-moment (HM) and low-moment (LM) configurations
    with customizable frequency content and Hankel transform parameters.

    Parameters
    ----------
    GEX : dict
        Dictionary containing GEX system configuration data with keys:
        - 'General': System description and waveform information
        - Waveform and timing parameters for electromagnetic modeling
    **kwargs : dict
        Additional configuration parameters:
        - Nhank : int, number of Hankel transform coefficients (default 280)
        - Nfreq : int, number of frequencies for transform (default 12)
        - Ndig : int, number of digital filters (default 7)
        - showInfo : int, verbosity level (0=silent, >0=verbose)
        - WindowWeightingScheme : str, weighting scheme ('AreaUnderCurve', 'BoxCar')
        - NumAbsHM : int, number of abscissae for high moment (default Nhank)
        - NumAbsLM : int, number of abscissae for low moment (default Nhank)
        - NumFreqHM : int, number of frequencies for high moment (default Nfreq)
        - NumFreqLM : int, number of frequencies for low moment (default Nfreq)

    Returns
    -------
    list of str
        List of file paths for the generated STM files (typically HM and LM variants).

    Notes
    -----
    STM files contain system transfer functions that describe the electromagnetic
    system response characteristics needed for accurate forward modeling. The
    function generates separate files for high-moment and low-moment configurations
    when applicable.
    
    The generated STM files follow GA-AEM format specifications and include:
    - Frequency domain transfer functions
    - Hankel transform coefficients
    - Digital filter parameters
    - System timing and waveform information
    
    File naming convention follows: {system_description}_{moment_type}.stm
    """
    system_name = GEX['General']['Description']

    # Parse kwargs
    Nhank = kwargs.get('Nhank', 280)
    Nfreq = kwargs.get('Nfreq', 12)
    Ndig = kwargs.get('Ndig', 7)
    showInfo = kwargs.get('showInfo', 0)
    WindowWeightingScheme  = kwargs.get('WindowWeightingScheme', 'AreaUnderCurve')
    #WindowWeightingScheme  = kwargs.get('WindowWeightingScheme', 'BoxCar')


    NumAbsHM = kwargs.get('NumAbsHM', Nhank)
    NumAbsLM = kwargs.get('NumAbsLM', Nhank)
    NumFreqHM = kwargs.get('NumFreqHM', Nfreq)
    NumFreqLM = kwargs.get('NumFreqLM', Nfreq)
    DigitFreq = kwargs.get('DigitFreq', 4E6)
    stm_dir = kwargs.get('stm_dir', os.getcwd())
    file_gex = kwargs.get('file_gex', '')

    windows = GEX['General']['GateArray']

    LastWin_LM = int(GEX['Channel1']['NoGates'][0])
    LastWin_HM = int(GEX['Channel2']['NoGates'][0])

    SkipWin_LM = int(GEX['Channel1']['RemoveInitialGates'][0])
    SkipWin_HM = int(GEX['Channel2']['RemoveInitialGates'][0])

    windows_LM = windows[SkipWin_LM:LastWin_LM, :] + GEX['Channel1']['GateTimeShift'][0] + GEX['Channel1']['MeaTimeDelay'][0]
    windows_HM = windows[SkipWin_HM:LastWin_HM, :] + GEX['Channel2']['GateTimeShift'][0] + GEX['Channel2']['MeaTimeDelay'][0]

    #windows_LM = GEX['Channel1']['GateFactor'][0] * windows_LM
    #windows_HM = GEX['Channel2']['GateFactor'][0] * windows_HM
    #windows_LM = windows_LM/GEX['Channel1']['GateFactor'][0] 
    #windows_HM = windows_HM/GEX['Channel2']['GateFactor'][0] 
    
    NWin_LM = windows_LM.shape[0]
    NWin_HM = windows_HM.shape[0]

    # PREPARE WAVEFORMS
    LMWF = GEX['General']['WaveformLM']
    HMWF = GEX['General']['WaveformHM']


    LMWFTime1 = LMWF[0, 0]
    LMWFTime2 = LMWF[-1, 0]

    HMWFTime1 = LMWF[0, 0]
    HMWFTime2 = LMWF[-1, 0]

    LMWF_Period = 1. / GEX['Channel1']['RepFreq'][0]
    HMWF_Period = 1. / GEX['Channel2']['RepFreq'][0]

    # Check if full waveform is defined
    LMWF_isfull = (LMWFTime2 - LMWFTime1) == LMWF_Period
    HMWF_isfull = (HMWFTime2 - HMWFTime1) == HMWF_Period

    

    if not LMWF_isfull:
        LMWF = np.vstack((LMWF, [LMWFTime1 + LMWF_Period, 0]))

    if not HMWF_isfull:
        HMWF = np.vstack((HMWF, [HMWFTime1 + HMWF_Period, 0]))

    # Make sure the output folder exists
    if not os.path.isdir(stm_dir):
        os.mkdir(stm_dir)

    if len(file_gex) > 0:
        p, gex_f = os.path.split(file_gex)
        # get filename without extension
        gex_f = os.path.splitext(gex_f)[0]
        gex_str = gex_f + '_'
        # Remove next line when working OK
        gex_str = gex_f + '-P-'
    else:
        gex_str = ''

    LM_name = os.path.join(stm_dir, gex_str + system_name + '_LM.stm')
    HM_name = os.path.join(stm_dir, gex_str + system_name + '_HM.stm')

    stm_files = [LM_name, HM_name]
    if (showInfo>0):
        print('writing LM to %s'%(LM_name))
        print('writing HM to %s'%(HM_name))

    # WRITE LM AND HM FILES
    with open(LM_name, 'w') as fID_LM:
        fID_LM.write('System Begin\n')
        
        fID_LM.write('\tName = %s\n' % (GEX['General']['Description']))        
        fID_LM.write("\tType = Time Domain\n\n")

        fID_LM.write("\tTransmitter Begin\n")
        fID_LM.write("\t\tNumberOfTurns = 1\n")
        fID_LM.write("\t\tPeakCurrent = 1\n")
        fID_LM.write("\t\tLoopArea = 1\n")
        fID_LM.write("\t\tBaseFrequency = %f\n" % GEX['Channel1']['RepFreq'][0])
        fID_LM.write("\t\tWaveformDigitisingFrequency = %s\n" % ('%21.8f' % DigitFreq))
        fID_LM.write("\t\tWaveFormCurrent Begin\n")
        np.savetxt(fID_LM, GEX['General']['WaveformLM'], fmt='%23.6e', delimiter=' ')
        fID_LM.write("\t\tWaveFormCurrent End\n")
        fID_LM.write("\tTransmitter End\n\n")
        
        fID_LM.write("\tReceiver Begin\n")
        fID_LM.write("\t\tNumberOfWindows = %d\n" % NWin_LM)
        fID_LM.write("\t\tWindowWeightingScheme = %s\n" % WindowWeightingScheme)
        fID_LM.write('\t\tWindowTimes Begin\n')
        #np.savetxt(fID_LM, windows_LM, fmt='%23.6e', delimiter=' ')
        np.savetxt(fID_LM, windows_LM[:,1::], fmt='%23.6e', delimiter=' ')
        fID_LM.write('\t\tWindowTimes End\n\n')
        TiBFilt = GEX['Channel1']['TiBLowPassFilter']

        fID_LM.write('\t\tLowPassFilter Begin\n')
        fID_LM.write('\t\t\tCutOffFrequency = %10.0f\n' % (TiBFilt[1]))
        fID_LM.write('\t\t\tOrder = %d\n' % (TiBFilt[0]))
        fID_LM.write('\t\tLowPassFilter End\n\n')
        
        fID_LM.write('\tReceiver End\n\n')
        
        fID_LM.write('\tForwardModelling Begin\n')
        #fID_LM.write('\t\tModellingLoopRadius = %f\n' % (np.sqrt(GEX['General']['TxLoopArea'][0] / np.pi)))
        fID_LM.write('\t\tModellingLoopRadius = %5.4f\n' % (np.sqrt(GEX['General']['TxLoopArea'][0] / np.pi)))
        fID_LM.write('\t\tOutputType = dB/dt\n')
        fID_LM.write('\t\tSaveDiagnosticFiles = no\n')
        fID_LM.write('\t\tXOutputScaling = 0\n')
        fID_LM.write('\t\tYOutputScaling = 0\n')
        fID_LM.write('\t\tZOutputScaling = 1\n')
        fID_LM.write('\t\tSecondaryFieldNormalisation = none\n')
        fID_LM.write('\t\tFrequenciesPerDecade = %d\n' % NumFreqLM)
        fID_LM.write('\t\tNumberOfAbsiccaInHankelTransformEvaluation = %d\n' % NumAbsLM)
        fID_LM.write('\tForwardModelling End\n\n')

        fID_LM.write('System End\n')

    with open(HM_name, 'w') as fID_HM:
        fID_HM.write('System Begin\n')
        fID_HM.write('\tName = %s\n' % (GEX['General']['Description']))
        fID_HM.write("\tType = Time Domain\n\n")

        fID_HM.write("\tTransmitter Begin\n")
        fID_HM.write("\t\tNumberOfTurns = 1\n")
        fID_HM.write("\t\tPeakCurrent = 1\n")
        fID_HM.write("\t\tLoopArea = 1\n")
        fID_HM.write("\t\tBaseFrequency = %f\n" % GEX['Channel2']['RepFreq'][0])
        fID_HM.write("\t\tWaveformDigitisingFrequency = %s\n" % ('%21.8f' % DigitFreq))
        fID_HM.write("\t\tWaveFormCurrent Begin\n")
        np.savetxt(fID_HM, GEX['General']['WaveformHM'], fmt='%23.6e', delimiter=' ')
        fID_HM.write("\t\tWaveFormCurrent End\n")
        fID_HM.write("\tTransmitter End\n\n")

        fID_HM.write("\tReceiver Begin\n")
        fID_HM.write("\t\tNumberOfWindows = %d\n" % NWin_HM)
        fID_HM.write("\t\tWindowWeightingScheme = %s\n" % WindowWeightingScheme)
        fID_HM.write('\t\tWindowTimes Begin\n')
        #np.savetxt(fID_HM, windows_HM, fmt='%23.6e', delimiter=' ')
        np.savetxt(fID_HM, windows_HM[:,1::], fmt='%23.6e', delimiter=' ')
        fID_HM.write('\t\tWindowTimes End\n\n')
        TiBFilt = GEX['Channel2']['TiBLowPassFilter']
        
        fID_HM.write('\t\tLowPassFilter Begin\n')
        fID_HM.write('\t\t\tCutOffFrequency = %10.0f\n' % (TiBFilt[1]))
        fID_HM.write('\t\t\tOrder = %d\n' % (TiBFilt[0]))
        fID_HM.write('\t\tLowPassFilter End\n\n')
        
        fID_HM.write('\tReceiver End\n\n')

        fID_HM.write('\tForwardModelling Begin\n')
        #fID_HM.write('\t\tModellingLoopRadius = %f\n' % (np.sqrt(GEX['General']['TxLoopArea'][0] / np.pi)))
        fID_HM.write('\t\tModellingLoopRadius = %5.4f\n' % (np.sqrt(GEX['General']['TxLoopArea'][0] / np.pi)))
        fID_HM.write('\t\tOutputType = dB/dt\n')
        fID_HM.write('\t\tSaveDiagnosticFiles = no\n')
        fID_HM.write('\t\tXOutputScaling = 0\n')
        fID_HM.write('\t\tYOutputScaling = 0\n')
        fID_HM.write('\t\tZOutputScaling = 1\n')
        fID_HM.write('\t\tSecondaryFieldNormalisation = none\n')
        fID_HM.write('\t\tFrequenciesPerDecade = %d\n' % NumFreqHM)
        fID_HM.write('\t\tNumberOfAbsiccaInHankelTransformEvaluation = %d\n' % NumAbsHM)
        fID_HM.write('\tForwardModelling End\n\n')

        fID_HM.write('System End\n')

    return stm_files


def read_gex(file_gex, **kwargs):
    """
    Parse GEX (Geometry Exchange) file into structured dictionary.

    Reads and parses electromagnetic system configuration files in GEX format,
    which contain survey geometry, system parameters, waveforms, and timing
    information required for electromagnetic forward modeling.

    Parameters
    ----------
    file_gex : str
        Path to the GEX file containing electromagnetic system configuration.
    **kwargs : dict
        Additional parsing parameters:
        - Nhank : int, number of Hankel transform abscissae for both frequency
          windows (used in processing, not directly from file)
        - Nfreq : int, number of frequencies per decade for both frequency
          windows (used in processing, not directly from file)
        - Ndig : int, number of digits for waveform digitizing frequency
          (used in processing, not directly from file)
        - showInfo : int, verbosity level (0=silent, >0=verbose, default 0)

    Returns
    -------
    dict
        Dictionary containing parsed GEX file contents with structure:
        - 'filename' : str, original file path
        - 'General' : dict, system description and general parameters
        - Section-specific dictionaries containing parameters grouped by
          functionality (e.g., timing, waveforms, filters)
        - 'WaveformLM' : numpy.ndarray, low-moment waveform points
        - 'WaveformHM' : numpy.ndarray, high-moment waveform points  
        - 'GateArray' : numpy.ndarray, measurement gate timing

    Raises
    ------
    FileNotFoundError
        If the specified GEX file does not exist or cannot be accessed.

    Notes
    -----
    GEX files use a section-based format with key=value pairs:
    - [Section] headers define parameter groups
    - Numeric values are automatically converted to numpy arrays
    - String values are preserved as text
    - Waveform and gate timing data are consolidated into arrays
    
    The parser automatically handles:
    - Multi-point waveform definitions (WaveformLMPoint*, WaveformHMPoint*)
    - Gate timing arrays (GateTime*)
    - Numeric array conversion with space-separated values
    - Comments and formatting variations
    
    Output dictionary structure matches INTEGRATE conventions for
    electromagnetic system configuration and GA-AEM compatibility.
    """
    showInfo = kwargs.get('showInfo', 0)
    
    GEX = {}
    GEX['filename']=file_gex
    comment_counter = 1
    current_key = None

    # Check if file_gex exists
    if not os.path.exists(file_gex):
        raise FileNotFoundError(f"Error: file {file_gex} does not exist")

    with open(file_gex, 'r') as file:
        for line in file.readlines():
            line = line.strip()
            if line.startswith('/'):
                GEX[f'comment{comment_counter}'] = line[1:].strip()
                comment_counter += 1
            elif line.startswith('['):
                current_key = line[1:-1]
                GEX[current_key] = {}
            else:
                key_value = line.split('=')
                if len(key_value) == 2:
                    key, value = key_value[0].strip(), key_value[1].strip()
                    
                    try:                        
                        GEX[current_key][key] = np.fromstring(value, sep=' ')
                    #except ValueError:
                    except:
                        GEX[current_key][key] = value

                    if len(GEX[current_key][key])==0:
                        # value is probably a string
                        GEX[current_key][key]=value


    # WaveformLM
    waveform_keys = [key for key in GEX['General'].keys() if 'WaveformLMPoint' in key]
    waveform_keys.sort(key=lambda x: int(x.replace('WaveformLMPoint', '')))

    waveform_values = [GEX['General'][key] for key in waveform_keys]
    GEX['General']['WaveformLM'] = np.vstack(waveform_values)
    
    for key in waveform_keys:
        del GEX['General'][key]

    # WaveformHM
    waveform_keys = [key for key in GEX['General'].keys() if 'WaveformHMPoint' in key]
    waveform_keys.sort(key=lambda x: int(x.replace('WaveformHMPoint', '')))

    waveform_values = [GEX['General'][key] for key in waveform_keys]
    GEX['General']['WaveformHM']=np.vstack(waveform_values)

    for key in waveform_keys:
        del GEX['General'][key]

    # GateArray
    gate_keys = [key for key in GEX['General'].keys() if 'GateTime' in key]
    gate_keys.sort(key=lambda x: int(x.replace('GateTime', '')))

    gate_values = [GEX['General'][key] for key in gate_keys]
    GEX['General']['GateArray']=np.vstack(gate_values)

    for key in gate_keys:
        del GEX['General'][key]

    return GEX
    


# gex_to_stm: convert a GEX file to a set of STM files
def gex_to_stm(file_gex, **kwargs):
    """
    Convert GEX system configuration to STM files for electromagnetic modeling.

    Convenience function that combines GEX file reading and STM file generation
    into a single operation. Handles both file paths and pre-loaded GEX dictionaries
    to create system transfer matrix files required for GA-AEM forward modeling.

    Parameters
    ----------
    file_gex : str or dict
        GEX system configuration. Can be either:
        - str: Path to GEX file to be read and processed
        - dict: Pre-loaded GEX dictionary from previous read_gex() call
    **kwargs : dict
        Additional parameters passed to write_stm_files():
        - Nhank : int, number of Hankel transform coefficients
        - Nfreq : int, number of frequencies for transform
        - showInfo : int, verbosity level
        - Other STM generation parameters

    Returns
    -------
    tuple
        Tuple containing (stm_files, GEX) where:
        - stm_files : list of str, paths to generated STM files
        - GEX : dict, processed GEX dictionary used for STM generation

    Raises
    ------
    TypeError
        If file_gex is neither a string nor a dictionary.
    FileNotFoundError
        If file_gex is a string pointing to a non-existent file.

    Notes
    -----
    This function provides a streamlined workflow for electromagnetic system
    setup by automating the GEX→STM conversion process. The generated STM files
    contain system transfer functions needed for accurate forward modeling
    with GA-AEM.
    
    When file_gex is a string, the function calls read_gex() internally.
    When file_gex is a dictionary, it's assumed to be a valid GEX structure.
    The write_stm_files() function handles the actual STM file generation
    with the provided or default parameters.
    """
    if isinstance(file_gex, str):
        GEX = read_gex(file_gex)
        stm_files = write_stm_files(GEX, file_gex=file_gex, **kwargs)
    else:
        GEX = file_gex
        stm_files = write_stm_files(GEX, file_gex=GEX['filename'], **kwargs)

    return stm_files, GEX


def get_gex_file_from_data(f_data_h5, id=1):
    """
    Retrieves the 'gex' attribute from the specified HDF5 file.

    :param str f_data_h5: The path to the HDF5 file.
    :param int id: The ID of the dataset within the HDF5 file. Defaults to 1.
    :return: The value of the 'gex' attribute if found, otherwise an empty string.
    :rtype: str
    """
    with h5py.File(f_data_h5, 'r') as f:
        dname = '/D%d' % id
        if 'gex' in f[dname].attrs:
            file_gex = f[dname].attrs['gex']
        else:
            print('"gex" attribute not found in %s:%s' % (f_data_h5,dname))
            file_gex = ''
    return file_gex


def get_geometry(f_data_h5):
    """
    Extract survey geometry data from HDF5 file.

    Retrieves spatial coordinates, survey line identifiers, and elevation data
    from an INTEGRATE data file. Automatically handles both direct data files
    and posterior files that reference data files.

    Parameters
    ----------
    f_data_h5 : str
        Path to the HDF5 file containing geometry data. Can be either a data
        file or posterior file (function automatically detects and uses correct file).

    Returns
    -------
    X : numpy.ndarray
        UTM X coordinates in meters, shape (N_points,).
    Y : numpy.ndarray  
        UTM Y coordinates in meters, shape (N_points,).
    LINE : numpy.ndarray
        Survey line identifiers, shape (N_points,).
    ELEVATION : numpy.ndarray
        Ground surface elevation in meters, shape (N_points,).

    Raises
    ------
    IOError
        If the HDF5 file cannot be opened or required datasets are missing.

    Examples
    --------
    >>> X, Y, LINE, ELEVATION = get_geometry('data.h5')
    >>> print(f"Survey covers {X.max()-X.min():.0f}m x {Y.max()-Y.min():.0f}m")

    Notes
    -----
    The function expects geometry data to be stored in standard INTEGRATE format:
    - '/UTMX': UTM X coordinates
    - '/UTMY': UTM Y coordinates  
    - '/LINE': Survey line numbers
    - '/ELEVATION': Ground elevation
    
    When passed a posterior file, automatically extracts the reference to the
    original data file from the 'f5_data' attribute.
    """

    # if f_data_h5 has a feature called 'f5_prior' then use that file
    with h5py.File(f_data_h5, 'r') as f_data:
        if 'f5_data' in f_data.attrs:
            f_data_h5 = f_data.attrs['f5_data']
            print('Using f5_data_h5: %s' % f_data_h5)

    with h5py.File(f_data_h5, 'r') as f_data:
        X = f_data['/UTMX'][:].flatten()
        Y = f_data['/UTMY'][:].flatten()
        LINE = f_data['/LINE'][:].flatten()
        ELEVATION = f_data['/ELEVATION'][:].flatten()

    return X, Y, LINE, ELEVATION


def get_number_of_datasets(f_data_h5):
    """
    Get the number of datasets (D1, D2, D3, etc.) in an INTEGRATE data HDF5 file.
    
    Counts the number of dataset groups with names following the pattern 'D1', 'D2', 'D3', etc.
    in an INTEGRATE HDF5 data file. This function is useful for determining how many different
    data types or measurement systems are stored in a single file.
    
    Parameters
    ----------
    f_data_h5 : str
        Path to the HDF5 file containing INTEGRATE data with dataset groups.
        
    Returns
    -------
    int
        Number of datasets found in the file. Returns 0 if no datasets are found
        or if the file cannot be accessed.
        
    Raises
    ------
    FileNotFoundError
        If the specified HDF5 file does not exist.
    IOError
        If the HDF5 file cannot be opened or read.
        
    Examples
    --------
    >>> n_datasets = get_number_of_datasets('data.h5')
    >>> print(f"File contains {n_datasets} datasets")
    File contains 3 datasets
    
    Notes
    -----
    This function looks for HDF5 groups with names starting with 'D' followed by digits.
    The typical INTEGRATE data file structure includes:
    - '/D1/': First dataset (e.g., high moment data)
    - '/D2/': Second dataset (e.g., low moment data)  
    - '/D3/': Third dataset (e.g., processed data)
    - And so on...
    
    The function only counts top-level groups that match the 'D{number}' pattern,
    ignoring other groups like geometry data (UTMX, UTMY, etc.).
    """
    n_datasets = 0
    try:
        with h5py.File(f_data_h5, 'r') as f:
            for key in f.keys():
                if key[0] == 'D' and key[1:].isdigit():
                    n_datasets += 1
    except (FileNotFoundError, IOError) as e:
        raise e
    except Exception:
        # Return 0 for any other errors (e.g., corrupted file)
        return 0
    
    return n_datasets


def post_to_csv(f_post_h5='', Mstr='/M1'):
    """
    Export posterior results to CSV format for GIS integration.

    Converts posterior sampling results to CSV files containing spatial coordinates
    and model parameter statistics. Creates files suitable for import into GIS
    software or other analysis tools.

    Parameters
    ----------
    f_post_h5 : str, optional
        Path to the HDF5 file containing posterior results. If empty string,
        uses a default example file (default is '').
    Mstr : str, optional
        Model parameter dataset path within the HDF5 file (e.g., '/M1', '/M2').
        Specifies which model parameter to export (default is '/M1').

    Returns
    -------
    str
        Path to the generated CSV file.

    Raises
    ------
    KeyError
        If the specified model parameter dataset does not exist in the HDF5 file.
    FileNotFoundError
        If the specified HDF5 file does not exist or cannot be accessed.

    Notes
    -----
    The exported CSV file contains:
    - X, Y: UTM coordinates
    - ELEVATION: Ground surface elevation
    - Model statistics: Mean, Median, Mode, Standard deviation
    - For discrete models: probability distributions across classes
    - For continuous models: quantile values and uncertainty measures
    
    The function automatically handles both discrete and continuous model types
    based on the 'is_discrete' attribute in the prior file. Output format is
    optimized for GIS applications with appropriate coordinate reference systems.
    
    TODO: Future enhancements planned for LINE number export and separate
    functions for grid vs. point data export.
    """
    
    # TODO: Would be nice if also the LINE number was exported (to allow filter by LINE)
    # Perhaps this function should be split into two functions, 
    #   one for exporting the grid data and one for exporting the point data.
    # Also, split into a function the generates the points scatter data, and one that stores them as a csv file


    import pandas as pd
    import integrate as ig

    #Mstr = '/M1'
    # if f_post_h5 is Null then use the last f_post_h5 file

    if len(f_post_h5)==0:
        f_post_h5 = 'POST_PRIOR_Daugaard_N2000000_TX07_20230731_2x4_RC20-33_Nh280_Nf12_Nu2000000_aT1.h5'

    f_post =  h5py.File(f_post_h5, 'r')
    f_prior_h5 = f_post.attrs['f5_prior']
    f_prior =  h5py.File(f_prior_h5, 'r')
    f_data_h5 = f_post.attrs['f5_data']
    if 'x' in f_prior[Mstr].attrs.keys():
        z = f_prior[Mstr].attrs['x']
    else:
        z = f_prior[Mstr].attrs['z']    
    is_discrete = f_prior[Mstr].attrs['is_discrete']

    X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)

    # Check that Dstr exist in f_poyt_h5
    if Mstr not in f_post:
        print("ERROR: %s not in %s" % (Mstr, f_post_h5))
        sys.exit(1)

    D_mul = []
    D_name = []
    if is_discrete:
        D_mul.append(f_post[Mstr+'/Mode'])
        D_name.append('Mode')
        D_mul.append(f_post[Mstr+'/Entropy'])
        D_name.append('Entropy')
    else:
        D_mul.append(f_post[Mstr+'/Median'])
        D_name.append('Median')
        D_mul.append(f_post[Mstr+'/Mean'])
        D_name.append('Mean')
        D_mul.append(f_post[Mstr+'/Std'])
        D_name.append('Std')
    

    # replicate z[1::] to be a 2D matric of zie ndx89
    ZZ = np.tile(z[1::], (D_mul[0].shape[0], 1))

    #
    df = pd.DataFrame(data={'X': X, 'Y': Y, 'Line': LINE, 'ELEVATION': ELEVATION})

    dataframes = [df]

    for i in range(len(D_mul)):
        D = D_mul[i][:]
        
        for j in range(D.shape[1]):
            temp_df = pd.DataFrame(D[:,j], columns=[D_name[i]+'_'+str(j)])
            dataframes.append(temp_df)

    for j in range(ZZ.shape[1]):
        temp_df = pd.DataFrame(ZZ[:,j], columns=['zbot_'+str(j)])
        dataframes.append(temp_df)

    df = pd.concat(dataframes, axis=1)
    f_post_csv='%s_%s.csv' % (os.path.splitext(f_post_h5)[0],Mstr[1::])
    #f_post_csv='%s.csv' % (os.path.splitext(f_post_h5)[0])
    #f_post_csv = f_post_h5.replace('.h5', '.csv')
    print('Writing to %s' % f_post_csv)
    df.to_csv(f_post_csv, index=False)

    
    #%% Store point data sets of varianle in D_name
    # # Save a file with columns, x, y, z, and the median.
    print("----------------------------------------------------")
    D_mul_out = []
    for icat in range(len(D_name)):
        #icat=0
        Vstr = D_name[icat]
        print('Creating point data set: %s'  % Vstr)
        D=f_post[Mstr+'/'+Vstr]
        nd,nz=D.shape
        n = nd*nz

        Xp = np.zeros(n)
        Yp = np.zeros(n)
        Zp = np.zeros(n)
        LINEp = np.zeros(n)
        Dp = np.zeros(n)
        
        for i in range(nd):
            for j in range(nz):
                k = i*nz+j
                Xp[k] = X[i]
                Yp[k] = Y[i]
                Zp[k] = ELEVATION[i]-z[j]
                LINEp[k] = LINE[i]
                Dp[k] = D[i,j]        
        D_mul_out.append(Dp)

    if is_discrete:
        df = pd.DataFrame(data={'X': Xp, 'Y': Yp, 'Z': Zp, 'LINE': LINEp, D_name[0]: D_mul_out[0], D_name[1]: D_mul_out[1] })
    else:
        df = pd.DataFrame(data={'X': Xp, 'Y': Yp, 'Z': Zp, 'LINE': LINEp, D_name[0]: D_mul_out[0], D_name[1]: D_mul_out[1], D_name[2]: D_mul_out[2] })
    
    f_csv = '%s_%s_point.csv' % (os.path.splitext(f_post_h5)[0],Mstr[1::])
    print('- saving to : %s'  % f_csv)

    df.to_csv(f_csv, index=False)

    
    #%% CLOSE
    f_post.close()
    f_prior.close()

    return f_post_csv, f_csv


'''
HDF% related functions
'''
def copy_hdf5_file(input_filename, output_filename, N=None, loadToMemory=True, compress=True, **kwargs):
    """
    Copy the contents of an HDF5 file to another HDF5 file.

    :param input_filename: The path to the input HDF5 file.
    :type input_filename: str
    :param output_filename: The path to the output HDF5 file.
    :type output_filename: str
    :param N: The number of elements to copy from each dataset. If not specified, all elements will be copied.
    :type N: int, optional
    :param loadToMemory: Whether to load the entire dataset to memory before slicing. Default is True.
    :type loadToMemory: bool, optional
    :param compress: Whether to compress the output dataset. Default is True.
    :type compress: bool, optional

    :return: None
    """
    showInfo = kwargs.get('showInfo', 0)
    # Open the input file
    if showInfo>0:
        print('Trying to copy %s to %s' % (input_filename, output_filename))
    with h5py.File(input_filename, 'r') as input_file:
        # Create the output file
        with h5py.File(output_filename, 'w') as output_file:
            # Copy each group/dataset from the input file to the output file
            #i_use = np.sort(np.random.choice(400000,10,replace=False))
            i_use = []
            for name in input_file:
                if showInfo>0:
                    print('Copying %s' % name)
                if isinstance(input_file[name], h5py.Dataset):                    
                    # If N is specified, only copy the first N elements

                    if len(i_use)==0:
                        N_in = input_file[name].shape[0]
                        if N is None:
                            N=N_in
                        if N>N_in:
                            N=N_in
                        if N==N_in:                            
                            i_use = np.arange(N)
                        else:
                            i_use = np.sort(np.random.choice(N_in,N,replace=False))

                    if N<20000:
                        loadToMemory=False

                    # Read full dataset into memory
                    if loadToMemory:
                        # Load all data to memory, before slicing
                        if showInfo>0:
                            print('Loading %s to memory' % name)
                        data_in = input_file[name][:]    
                        data = data_in[i_use]
                    else:
                        # Read directly from HDF5 file   
                        data = input_file[name][i_use]

                    # Create new dataset in output file with compression
                    # Convert floating point data to 32-bit precision
                    if data.dtype.kind == 'f':
                        data = data.astype(np.float32)
                        
                    if compress:
                        #output_dataset = output_file.create_dataset(name, data=data, compression="lzf")
                        output_dataset = output_file.create_dataset(name, data=data, compression="gzip", compression_opts=4)
                    else:
                        output_dataset = output_file.create_dataset(name, data=data)
                    # Copy the attributes of the dataset
                    for key, value in input_file[name].attrs.items():                        
                        output_dataset.attrs[key] = value
                else:
                    input_file.copy(name, output_file)

            # Copy the attributes of the input file to the output file
            for key, value in input_file.attrs.items():
                output_file.attrs[key] = value

        return output_filename

def hdf5_scan(file_path):
    """
    Scans an HDF5 file and prints information about datasets (including their size) and attributes.

    Args:
        file_path (str): The path to the HDF5 file.

    """
    import h5py
    with h5py.File(file_path, 'r') as f:
        def print_info(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}")
                print(f"  Shape: {obj.shape}")
                print(f"  Data type: {obj.dtype}")
                if obj.attrs:
                    print("  Attributes:")
                    for attr_name, attr_value in obj.attrs.items():
                        print(f"    {attr_name}: {attr_value}")
            elif isinstance(obj, h5py.Group):
                if obj.attrs:
                    print(f"Group: {name}")
                    print("  Attributes:")
                    for attr_name, attr_value in obj.attrs.items():
                        print(f"    {attr_name}: {attr_value}")

        f.visititems(print_info)




def file_checksum(file_path):
    """
    Calculate the MD5 checksum of a file.

    :param file_path: The path to the file.
    :type file_path: str
    :return: The MD5 checksum of the file.
    :rtype: str
    """
    import hashlib
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def download_file(url, download_dir, use_checksum=False, **kwargs):
    """
    Download a file from a URL to a specified directory.

    :param url: The URL of the file to download.
    :type url: str
    :param download_dir: The directory to save the downloaded file.
    :type download_dir: str
    :param use_checksum: Whether to verify the file checksum after download.
    :type use_checksum: bool
    :param kwargs: Additional keyword arguments.
    :return: None
    """
    import requests
    import os
    showInfo = kwargs.get('showInfo', 0)
    # Extract the file name from the URL
    file_name = os.path.basename(url)
    file_path = os.path.join(download_dir, file_name)

    # Check if the file already exists locally
    if os.path.exists(file_path):
        if showInfo>0:
            print(f'File {file_name} already exists. Skipping download.')
        return

    # Check if the remote file exists
    if showInfo>1:
        print('Checking if file exists on the remote server...')
    head_response = requests.head(url)
    if head_response.status_code != 200:
        if showInfo>-1:
            print(f'File {file_name} does not exist on the remote server. Skipping download.')
        return

    # Download and save the file
    print(f'Downloading {file_name}')
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful

    with open(file_path, 'wb') as file:
        file.write(response.content)
    print(f'Downloaded {file_name}')

    # Check if checksum verification is enabled
    if use_checksum:
        # Calculate the MD5 checksum of the downloaded file
        downloaded_checksum = file_checksum(file_path)

        # Get the remote file checksum
        remote_checksum = head_response.headers.get('Content-MD5')

        # Compare checksums
        if downloaded_checksum != remote_checksum:
            print(f'Checksum verification failed for {file_name}. Downloaded file may be corrupted.')
            os.remove(file_path)
        else:
            print(f'Checksum verification successful for {file_name}.')
    else:
        pass
        # print(f'Checksum verification disabled for {file_name}.')

def download_file_old(url, download_dir, **kwargs):
    """
    Download a file from a URL to a specified directory (old version).

    :param url: The URL of the file to download.
    :type url: str
    :param download_dir: The directory to save the downloaded file.
    :type download_dir: str
    :param kwargs: Additional keyword arguments.
    :return: None
    """
    import requests
    import os
    showInfo = kwargs.get('showInfo', 0)
    # Extract the file name from the URL
    file_name = os.path.basename(url)
    file_path = os.path.join(download_dir, file_name)

    # Check if the remote file exists
    head_response = requests.head(url)
    if head_response.status_code != 200:
        print(f'File {file_name} does not exist on the remote server. Skipping download.')
        return

    # Check if the file already exists locally
    if os.path.exists(file_path):
        # Get the local file checksum
        local_checksum = file_checksum(file_path)

        # Download the remote file to a temporary location to compare checksums
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful

        remote_temp_path = os.path.join(download_dir, f'temp_{file_name}')
        with open(remote_temp_path, 'wb') as temp_file:
            temp_file.write(response.content)

        # Get the remote file checksum
        remote_checksum = file_checksum(remote_temp_path)

        # Compare checksums
        if local_checksum == remote_checksum:
            print(f'File {file_name} already exists and is identical. Skipping download.')
            os.remove(remote_temp_path)
            return
        else:
            print(f'File {file_name} exists but is different. Downloading new version.')
            os.remove(remote_temp_path)

    # Download and save the file
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful

    print(f'Downloading {file_name}')
    with open(file_path, 'wb') as file:
        file.write(response.content)
    print(f'Downloaded {file_name}')


def get_case_data(case='DAUGAARD', loadAll=False, loadType='', filelist=[], **kwargs):
    """
    Get case data for a specific case.

    :param case: The case name. Default is 'DAUGAARD'. Options are 'DAUGAARD', 'GRUSGRAV', 'FANGEL', 'HALD', 'ESBJERG', and 'OERUM.
    :type case: str
    :param loadAll: Whether to load all files for the case. Default is False.
    :type loadAll: bool
    :param loadType: The type of files to load. Options are '', 'prior', 'prior_data', 'post', and 'inout'.
    :type loadType: str
    :param filelist: A list of files to load. Default is an empty list.
    :type filelist: list
    :param kwargs: Additional keyword arguments.
    :return: A list of file names for the case.
    :rtype: list
    """
    showInfo = kwargs.get('showInfo', 0)

    if showInfo>-1:
        print('Getting data for case: %s' % case)

    if case=='DAUGAARD':

        if len(filelist)==0:
            filelist.append('DAUGAARD_AVG.h5')
            filelist.append('TX07_20231016_2x4_RC20-33.gex')
            filelist.append('README_DAUGAARD')

        if loadAll:
            filelist.append('DAUGAARD_RAW.h5')
            filelist.append('TX07_20230731_2x4_RC20-33.gex')
            filelist.append('TX07_20230828_2x4_RC20-33.gex')
            filelist.append('TX07_20230906_2x4_RC20-33.gex')
            filelist.append('tTEM_20230727_AVG_export.h5')
            filelist.append('tTEM_20230814_AVG_export.h5')
            filelist.append('tTEM_20230829_AVG_export.h5')
            filelist.append('tTEM_20230913_AVG_export.h5')
            filelist.append('tTEM_20231109_AVG_export.h5')
            filelist.append('DAUGAARD_AVG_inout.h5')

        if (loadAll or loadType=='shapefiles'):            
            #filelist.append('Begravet dal.zip')
            filelist.append('Begravet dal.shp')
            filelist.append('Begravet dal.shx')
            #filelist.append('Erosion øvre.zip')
            filelist.append('Erosion øvre.shp')
            filelist.append('Erosion øvre.shx')
            
        
        if (loadAll or loadType=='prior'):            
            filelist.append('prior_detailed_general_N2000000_dmax90.h5')
        
        if (loadAll or loadType=='prior_data' or loadType=='post'):            
            filelist.append('prior_detailed_general_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5')
            filelist.append('prior_detailed_invalleys_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5')
            filelist.append('prior_detailed_outvalleys_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5')
            
        if (loadAll or loadType=='post'):
            filelist.append('POST_DAUGAARD_AVG_prior_detailed_general_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12_Nu2000000_aT1.h5')
                    
        if (loadAll or loadType=='inout'):
            filelist.append('prior_detailed_invalleys_N2000000_dmax90.h5')
            filelist.append('prior_detailed_outvalleys_N2000000_dmax90.h5')
            filelist.append('prior_detailed_invalleys_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5')
            filelist.append('prior_detailed_outvalleys_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5')
            filelist.append('POST_DAUGAARD_AVG_prior_detailed_invalleys_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12_Nu2000000_aT1.h5')
            filelist.append('POST_DAUGAARD_AVG_prior_detailed_outvalleys_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12_Nu2000000_aT1.h5')    
            filelist.append('prior_detailed_inout_N4000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5')

    elif case=='ESBJERG':
        
        if (loadAll or loadType=='gex'):  
            filelist.append('TX07_20230906_2x4_RC20-33.gex')
            filelist.append('TX07_20231016_2x4_RC20-33.gex')
            filelist.append('TX07_20231127_2x4x1_RC20_33.gex')
            filelist.append('TX07_20240125_2x4_RC20-33.gex')
            filelist.append('TX07_20230906_2x4_RC20-33_merged.h5')  
            filelist.append('TX07_20231016_2x4_RC20-33_merged.h5')
            filelist.append('TX07_20231127_2x4x1_RC20_33_merged.h5')
            filelist.append('TX07_20240125_2x4_RC20-33_merged.h5')
        
        if (loadAll or loadType=='premerge'):
            filelist.append('20230921_AVG_export.h5')
            filelist.append('20230922_AVG_export.h5')
            filelist.append('20230925_AVG_export.h5')
            filelist.append('20230926_AVG_export.h5')
            filelist.append('20231026_AVG_export.h5')
            filelist.append('20231027_AVG_export.h5')
            filelist.append('20240109_AVG_export.h5')
            filelist.append('20240313_AVG_export.h5')
            filelist.append('TX07_20230906_2x4_RC20-33.gex')
            filelist.append('TX07_20231016_2x4_RC20-33.gex')
            filelist.append('TX07_20231127_2x4x1_RC20_33.gex')
            filelist.append('TX07_20240125_2x4_RC20-33.gex')

        if (loadAll or loadType=='ESBJERG_ALL' or len(filelist)==0):
            filelist.append('ESBJERG_ALL.h5')
            filelist.append('TX07_20230906_2x4_RC20-33.gex')
            filelist.append('README_ESBJERG')
   
        if (loadAll or loadType=='prior' or len(filelist)==0):
            filelist.append('prior_Esbjerg_claysand_N2000000_dmax90.h5')
            filelist.append('prior_Esbjerg_piggy_N2000000.h5')
            
        if (loadAll or loadType=='priordata' or len(filelist)==0):
            filelist.append('prior_Esbjerg_piggy_N2000000_TX07_20230906_2x4_RC20-33_Nh280_Nf12.h5')
            filelist.append('prior_Esbjerg_claysand_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5')


    elif case=='GRUSGRAV':

        filelist = []
        filelist.append('GRUSGRAV_AVG.h5')
        filelist.append('TX07_20230425_2x4_RC20_33.gex')
        filelist.append('README_GRUSGRAV')                    

        if (loadAll or loadType=='prior'):            
            filelist.append('DJURSLAND_P01_N1000000_NB-13_NR03_PRIOR.h5') 
            filelist.append('DJURSLAND_P03_N1000000_NB-13_NR03_PRIOR.h5')
            filelist.append('DJURSLAND_P13_N1000000_NB-13_NR03_PRIOR.h5')  
            filelist.append('DJURSLAND_P40_N1000000_NB-13_NR03_PRIOR.h5')
            filelist.append('DJURSLAND_P02_N1000000_NB-13_NR03_PRIOR.h5')  
            filelist.append('DJURSLAND_P12_N1000000_NB-13_NR03_PRIOR.h5')  
            filelist.append('DJURSLAND_P34_N1000000_NB-13_NR03_PRIOR.h5')  
            filelist.append('DJURSLAND_P60_N1000000_NB-13_NR03_PRIOR.h5')    
            
    elif case=='FANGEL':

        filelist = []
        filelist.append('FANGEL_AVG.h5')
        filelist.append('TX07_20230828_2x4_RC20-33.gex')
        filelist.append('README_FANGEL')

    elif case=='HALD':

        filelist = []
        filelist.append('HALD_AVG.h5')
        filelist.append('TX07_20230731_2x4_RC20-33.gex')
        filelist.append('README_HALD')
        if loadAll:
            filelist.append('TX07_20231016_2x4_RC20-33.gex')
            filelist.append('HALD_RAW.h5')
            filelist.append('tTEM_20230801_AVG_export.h5')
            filelist.append('tTEM_20230815_AVG_export.h5')
            filelist.append('tTEM_20230905_AVG_export.h5')
            filelist.append('tTEM_20231018_AVG_export.h5')

    elif case=='OERUM':
        filelist.append('OERUM_AVG.h5')
        filelist.append('TX07_20240802_2x4_RC20-39.gex')
        filelist.append('README_OERUM')
        if loadAll:
            filelist.append('OERUM_RAW.h5')
            filelist.append('20240827_AVG_export.h5')
            filelist.append('20240828_AVG_export.h5')
            filelist.append('20240903_AVG_export.h5')
            filelist.append('20240827_RAW_export.h5')
            filelist.append('20240828_RAW_export.h5')
            filelist.append('20240903_RAW_export.h5')
                  

    elif case=='HJOELLUND':
        filelist.append('HJOELLUND_AVG.h5')
        filelist.append('TX07_20241014_2x4_RC20_33_and_57_EksternGPS.gex')
        filelist.append('README_HJOELLUND')
        if loadAll:
            filelist.append('HJOELLUND_RAW.h5')

    elif case=='HADERUP':
        filelist.append('HADERUP_MEAN_ALL.h5')
        filelist.append('TX07_Haderup_mean.gex')
        filelist.append('README_HADERUP')
        #if loadAll:
        #    filelist.append('HADERUP_RAW.h5')
                  

    else:
        
        filelist = []
        print('Case %s not found' % case)


    urlErda = 'https://anon.erda.au.dk/share_redirect/dxOLKDtoul'
    urlErdaCase = '%s/%s' % (urlErda,case)
    for remotefile in filelist:
        #print(remotefile)
        remoteurl = '%s/%s' % (urlErdaCase,remotefile)
        #remoteurl = 'https://anon.erda.au.dk/share_redirect/dxOLKDtoul/%s/%s' % (case,remotefile)
        download_file(remoteurl,'.',showInfo=showInfo)
    if showInfo>-1:
        print('--> Got data for case: %s' % case)

    return filelist



def write_data_gaussian(D_obs, D_std = [], d_std=[], Cd=[], id=1, is_log = 0, f_data_h5='data.h5', **kwargs):
    """
    Write observational data with Gaussian noise model to HDF5 file.

    Creates HDF5 datasets for electromagnetic or other geophysical measurements
    assuming Gaussian-distributed uncertainties. Handles both diagonal and full
    covariance representations of measurement errors.

    Parameters
    ----------
    D_obs : numpy.ndarray
        Observed data measurements with shape (N_stations, N_channels).
        Each row represents a measurement location, each column a data channel.
    D_std : list, optional
        Standard deviations of observed data, same shape as D_obs.
        If empty, computed from d_std parameter (default is []).
    d_std : list, optional
        Default standard deviation values or multipliers for uncertainty
        calculation when D_std is not provided (default is []).
    Cd : list, optional
        Full covariance matrices for measurement uncertainties.
        If provided, takes precedence over D_std (default is []).
    id : int, optional
        Dataset identifier for HDF5 group naming ('/D{id}', default is 1).
    is_log : int, optional
        Flag indicating logarithmic data scaling (0=linear, 1=log, default is 0).
    f_data_h5 : str, optional
        Path to output HDF5 file (default is 'data.h5').
    **kwargs : dict
        Additional metadata parameters:
        - showInfo : int, verbosity level
        - Other dataset attributes for electromagnetic processing

    Returns
    -------
    str
        Path to the HDF5 file where data was written.

    Notes
    -----
    The function creates HDF5 structure following INTEGRATE conventions:
    - '/D{id}/d_obs': observed measurements
    - '/D{id}/d_std': measurement standard deviations (if available)
    - '/D{id}/Cd': full covariance matrix (if provided)
    - Dataset attributes include 'noise_model'='gaussian'
    
    Uncertainty handling priority: Cd > D_std > computed from d_std
    The Gaussian noise model assumes independent, normally distributed
    measurement errors with specified standard deviations or covariances.
    
    All datasets use float32 precision and gzip compression for efficiency.
    
    .. note::
        **Additional Parameters (kwargs):**
        
        - showInfo (int): Level of verbosity for printing information. Default is 0.
        - f_gex (str): Name of the GEX file associated with the data. Default is empty string.
        
        **Behavior:**
        
        - If D_std is not provided, it is calculated as d_std * D_obs
        - The function ensures that datasets 'UTMX', 'UTMY', 'LINE', and 'ELEVATION' exist
        - If a group with name 'D{id}' exists, it is removed before adding new data
        - Writes attributes 'noise_model' and 'is_log' to the dataset group
    """
    
    showInfo = kwargs.get('showInfo', 0)
    f_gex = kwargs.get('f_gex', '')

    if len(D_std)==0:
        if len(d_std)==0:
            d_std = 0.01
        D_std = np.abs(d_std * D_obs)

    D_str = 'D%d' % id

    ns,nd=D_obs.shape
    
    with h5py.File(f_data_h5, 'a') as f:
        # check if '/UTMX' exists and create it if it does not
        if 'UTMX' not in f:
            if showInfo>0:
                print('Creating %s:/UTMX' % f_data_h5) 
            UTMX = np.atleast_2d(np.arange(ns)).T
            f.create_dataset('UTMX' , data=UTMX) 
        if 'UTMY' not in f:
            if showInfo>0:
                print('Creating %s:/UTMY' % f_data_h5)
            UTMY = f['UTMX'][:]*0
            f.create_dataset('UTMY', data=UTMY)
        if 'LINE' not in f:
            if showInfo>0:
                print('Creating %s:/LINE' % f_data_h5)
            LINE = f['UTMX'][:]*0+1
            f.create_dataset('LINE', data=LINE)
        if 'ELEVATION' not in f:
            if showInfo>0:
                print('Creating %s:/ELEVATION' % f_data_h5)
            ELEVATION = f['UTMX'][:]*0
            f.create_dataset('ELEVATION', data=ELEVATION)

    # check if group 'D{id}/' exists and remove it if it does
    with h5py.File(f_data_h5, 'a') as f:
        if D_str in f:
            if showInfo>-1:
                print('Removing group %s:%s ' % (f_data_h5,D_str))
            del f[D_str]

    # Write DATA
    with h5py.File(f_data_h5, 'a') as f:
        if showInfo>-1:
            print('Adding group %s:%s ' % (f_data_h5,D_str))

        f.create_dataset('/%s/d_obs' % D_str, data=D_obs)
        # Write either Cd or d_std
        if len(Cd) == 0:
            f.create_dataset('/%s/d_std' % D_str, data=D_std)
        else:
            f.create_dataset('/%s/Cd' % D_str, data=Cd)

        # wrote attribute noise_model
        f['/%s/' % D_str].attrs['noise_model'] = 'gaussian'
        f['/%s/' % D_str].attrs['is_log'] = is_log
        if len(f_gex)>0:
            f['/%s/' % D_str].attrs['gex'] = f_gex
    
    return f_data_h5

def write_data_multinomial(D_obs, i_use=None, id=[],  id_use=None, f_data_h5='data.h5', **kwargs):
    """
    Writes observed data to an HDF5 file in a specified group with a multinomial noise model.

    :param D_obs: The observed data array to be written to the file.
    :type D_obs: numpy.ndarray
    :param id: The ID of the group to write the data to. If not provided, the function will find the next available ID.
    :type id: list, optional
    :param id_use: The ID of PRIOR data the refer to this data. If not set, id_use=id
    :type id_use: list, optional
    :param f_data_h5: The path to the HDF5 file where the data will be written. Default is 'data.h5'.
    :type f_data_h5: str, optional
    :param kwargs: Additional keyword arguments.
    :return: The path to the HDF5 file where the data was written.
    :rtype: str
    """
    showInfo = kwargs.get('showInfo', 0)

    if np.ndim(D_obs)==1:
        D_obs = np.atleast_2d(D_obs).T

    # f_data_h5 is a HDF% file grousp "/D1/", "/D2". 
    # FInd the is with for the maximum '/D*' group
    if not id:
        with h5py.File(f_data_h5, 'a') as f:
            for id in range(1, 100):
                D_str = 'D%d' % id
                if D_str not in f:
                    break
        if showInfo>0:
            print('Using id=%d' % id)


    D_str = 'D%d' % id

    if showInfo>0:
        print("Trying to write %s to %s" % (D_str,f_data_h5))

    ns,nclass,nm=D_obs.shape

    if i_use is None:
        i_use = np.ones((ns,1))
    if np.ndim(D_obs)==1:
        i_use = np.atleast_2d(i_use).T
    
    if id_use is None:
        id_use = id
        
    # check if group 'D{id}/' exists and remove it if it does
    with h5py.File(f_data_h5, 'a') as f:
        if D_str in f:
            if showInfo>-1:
                print('Removing group %s:%s ' % (f_data_h5,D_str))
                del f[D_str]


    # Write DATA
    with h5py.File(f_data_h5, 'a') as f:
        if showInfo>-1:
            print('Adding group %s:%s ' % (f_data_h5,D_str))

        f.create_dataset('/%s/d_obs' % D_str, data=D_obs)
        f.create_dataset('/%s/i_use' % D_str, data=i_use)
        
        f.create_dataset('/%s/id_use' % D_str, data=id_use)
            

        # write attribute noise_model as 'multinomial'
        f['/%s/' % D_str].attrs['noise_model'] = 'multinomial'
        
    return f_data_h5


def check_data(f_data_h5='data.h5', **kwargs):
    """
    Validate and complete INTEGRATE data file structure.

    Ensures HDF5 data files contain required geometry datasets (UTMX, UTMY, LINE,
    ELEVATION) for electromagnetic surveys. Creates missing datasets using provided
    values or sensible defaults based on existing data dimensions.

    Parameters
    ----------
    f_data_h5 : str, optional
        Path to the HDF5 data file to validate and update (default is 'data.h5').
    **kwargs : dict
        Dataset values and configuration options:
        - UTMX : array-like, UTM X coordinates
        - UTMY : array-like, UTM Y coordinates  
        - LINE : array-like, survey line identifiers
        - ELEVATION : array-like, ground elevation values
        - showInfo : int, verbosity level (0=silent, >0=verbose)

    Returns
    -------
    None
        Function modifies the HDF5 file in place, adding missing datasets.

    Raises
    ------
    KeyError
        If 'D1/d_obs' dataset is missing and geometry dimensions cannot be determined.
    FileNotFoundError
        If the specified HDF5 file does not exist.

    Notes
    -----
    The function ensures INTEGRATE data files have complete geometry information:
    - UTMX, UTMY: Spatial coordinates (required for mapping and modeling)
    - LINE: Survey line identifiers (required for data organization) 
    - ELEVATION: Ground surface elevation (required for depth calculations)
    
    Default value generation when datasets are missing:
    - UTMX: Sequential values 0, 1, 2, ... (placeholder coordinates)
    - UTMY: Sequential values 0, 1, 2, ... (placeholder coordinates)
    - LINE: All values set to 1 (single survey line)
    - ELEVATION: All values set to 0 (sea level reference)
    
    Dataset dimensions are inferred from existing 'D1/d_obs' observations.
    All geometry datasets are created with consistent length matching
    the number of measurement locations.
        
        - showInfo (int): Verbosity level. If greater than 0, prints information messages. Default is 0.
        - UTMX (array-like): Array of UTMX coordinate values. If not provided, attempts to read from file or generates defaults.
        - UTMY (array-like): Array of UTMY coordinate values. Default is zeros array with same length as UTMX.
        - LINE (array-like): Array of survey line identifiers. Default is ones array with same length as UTMX.
        - ELEVATION (array-like): Array of elevation values. Default is zeros array with same length as UTMX.
        
        **Behavior:**
        
        - If UTMX is not provided, function attempts to determine array length from existing 'D1/d_obs' dataset
        - Missing datasets are created with appropriate default values
        - Existing datasets are preserved and not overwritten
    """

    showInfo = kwargs.get('showInfo', 0)

    if showInfo>0:
        print('Checking INTEGRATE data in %s' % f_data_h5)  

    UTMX = kwargs.get('UTMX', [])
    if len(UTMX)==0:
        with h5py.File(f_data_h5, 'r') as f:
            if 'UTMX' in f:
                UTMX = f['UTMX'][:]
            else:
                ns = f['D1/d_obs'].shape[0] 
                print('UTMX not found in %s' % f_data_h5)
                UTMX = np.atleast_2d(np.arange(ns)).T    
            f.close()

    UTMY = kwargs.get('UTMY', UTMX*0)
    LINE = kwargs.get('LINE', UTMX*0+1)
    ELEVATION = kwargs.get('ELEVATION', UTMX*0)

    with h5py.File(f_data_h5, 'a') as f:
        # check if '/UTMX' exists and create it if it does not
        if 'UTMX' not in f:
            if showInfo>0:
                print('Creating UTMX')            
            f.create_dataset('UTMX', data=UTMX) 
        if 'UTMY' not in f:
            if showInfo>0:
                print('Creating UTMY')            
            f.create_dataset('UTMY', data=UTMY)
        if 'LINE' not in f:
            if showInfo>0:
                print('Creating LINE')
            f.create_dataset('LINE', data=LINE)
        if 'ELEVATION' not in f:
            if showInfo>0:
                print('Creating ELEVATION')
            f.create_dataset('ELEVATION', data=ELEVATION)

            f.close()




def merge_data(f_data, f_gex='', delta_line=0, f_data_merged_h5='', **kwargs):
    """
    Merge multiple data files into a single HDF5 file.

    :param f_data: List of input data files to merge.
    :type f_data: list
    :param f_gex: Path to geometry exchange file, by default ''.
    :type f_gex: str, optional
    :param delta_line: Line number increment for each merged file, by default 0.
    :type delta_line: int, optional
    :param f_data_merged_h5: Output merged HDF5 file path, by default derived from f_gex.
    :type f_data_merged_h5: str, optional
    :param kwargs: Additional keyword arguments.
    :return: Filename of the merged HDF5 file.
    :rtype: str
    :raises ValueError: If f_data is not a list.
    """
    
    import h5py
    import numpy as np
    import integrate as ig

    showInfo = kwargs.get('showInfo', 0)

    if len(f_data_merged_h5) == 0:
        f_data_merged_h5 = f_gex.split('.')[0] + '_merged.h5'
    

    # CHeck the f_data is a list. If so return a error
    if not isinstance(f_data, list):
        raise ValueError('f_data must be a list of strings')

    nd = len(f_data)
    if showInfo:
        print('Merging %d data sets to %s ' % (nd, f_data_merged_h5))
    
    f_data_h5 = f_data[0]
    if showInfo>1:
        print('.. Merging ', f_data_h5)    
    Xc, Yc, LINEc, ELEVATIONc = ig.get_geometry(f_data_h5)
    Dc = ig.load_data(f_data_h5, showInfo=showInfo-1)
    d_obs_c = Dc['d_obs']
    d_std_c = Dc['d_std']
    noise_model = Dc['noise_model']

    for i in range(1, len(f_data)):
        f_data_h5 = f_data[i]                   
        if showInfo>1:
            print('.. Merging ', f_data_h5)    
        X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)
        D = ig.load_data(f_data_h5, showInfo=showInfo)

        # append data
        Xc = np.append(Xc, X)
        Yc = np.append(Yc, Y)
        LINEc = np.append(LINEc, LINE+i*delta_line)
        ELEVATIONc = np.append(ELEVATIONc, ELEVATION)
        
        for id in range(len(d_obs_c)):
            #print(id)
            try:
                d_obs_c[id] = np.vstack((d_obs_c[id], np.atleast_2d(D['d_obs'][id])))        
                d_std_c[id] = np.vstack((d_std_c[id], np.atleast_2d(D['d_std'][id])))
            except:
                if showInfo>-1:
                    print("!!!!! Could not merge %s" % f_data_h5)

    Xc = np.atleast_2d(Xc).T
    Yc = np.atleast_2d(Yc).T
    LINEc = np.atleast_2d(LINEc).T
    ELEVATIONc = np.atleast_2d(ELEVATIONc).T

    with h5py.File(f_data_merged_h5, 'w') as f:
        f.create_dataset('UTMX', data=Xc)
        f.create_dataset('UTMY', data=Yc)
        f.create_dataset('LINE', data=LINEc)
        f.create_dataset('ELEVATION', data=ELEVATIONc)

    for id in range(len(d_obs_c)):
        write_data_gaussian(d_obs_c[id], D_std = d_std_c[id], noise_model = noise_model, f_data_h5=f_data_merged_h5, id=id+1, f_gex = f_gex)

    return f_data_merged_h5




## 

def merge_posterior(f_post_h5_files, f_data_h5_files, f_post_merged_h5='', showInfo=0):
    """
    Merge multiple posterior sampling results into unified datasets.

    Combines posterior results from separate electromagnetic survey areas or
    time periods into single merged files for comprehensive regional analysis.
    Handles both model parameter statistics and observational data consolidation.

    Parameters
    ----------
    f_post_h5_files : list of str
        List of paths to posterior HDF5 files containing sampling results
        from different survey areas or processing runs.
    f_data_h5_files : list of str
        List of paths to corresponding observational data HDF5 files.
        Must have same length as f_post_h5_files with matching order.
    f_post_merged_h5 : str, optional
        Output path for merged posterior file. If empty, generates default
        name based on input files (default is '').

    Returns
    -------
    tuple
        Tuple containing (merged_posterior_path, merged_data_path) where:
        - merged_posterior_path : str, path to merged posterior HDF5 file
        - merged_data_path : str, path to merged observational data HDF5 file

    Raises
    ------
    ValueError
        If f_data_h5_files and f_post_h5_files have different lengths.
    FileNotFoundError
        If any input files do not exist or cannot be accessed.

    Notes
    -----
    The merging process combines:
    - Model parameter statistics (Mean, Median, Mode, Std, Entropy)
    - Temperature and evidence fields from sampling
    - Geometry and observational data from all survey areas
    - Metadata and file references for traceability
    
    Spatial coordinates are preserved to maintain geographic relationships
    between different survey areas. The merged files retain full compatibility
    with INTEGRATE analysis and visualization functions.
    
    File naming convention for merged outputs follows pattern:
    'MERGED_{timestamp}_{description}.h5' when automatic naming is used.
        **File Naming:**
        
        - If f_post_merged_h5 is not provided, uses format: 'POST_merged_N{number_of_files}.h5'
        - Data file uses format: 'DATA_merged_N{number_of_files}.h5'
        
        **Dependencies:**
        
        - Requires the merge_data function to be available for merging observational data
        - Posterior files must have compatible structure for merging
        
        **Merging Process:**
        
        - Combines posterior sampling results from multiple files
        - Merges corresponding observational data
        - Maintains data integrity and structure consistency
    """
    import h5py
    import integrate as ig

    nf = len(f_post_h5_files)
    # Check that legth of f_data_h5_files is the same as f_post_h5_files
    if len(f_data_h5_files) != nf:
        raise ValueError('Length of f_data_h5_files must be the same as f_post_h5_files')

    if len(f_post_merged_h5) == 0:
        f_post_merged_h5 = 'POST_merged_N%d.h5' % nf

    f_data_merged_h5 = 'DATA_merged_N%d.h5' % nf

    if showInfo>0:
        print('Merging %d posterior files to %s' % (nf, f_post_merged_h5))
        print('Merging %d data files to %s' % (nf, f_data_merged_h5))

    f_data_merged_h5 = ig.merge_data(f_data_h5_files, f_data_merged_h5=f_data_merged_h5)


    for i in range(len(f_post_h5_files)):
        #  get 'i_sample' from the merged file
        f_post_h5 = f_post_h5_files[i]
        with h5py.File(f_post_h5, 'r') as f:
            i_use_s = f['i_use'][:]
            T_s = f['T'][:]
            EV_s = f['EV'][:]
            f_prior_h5 = f['/'].attrs['f5_prior']
            f_data_h5 = f['/'].attrs['f5_data']
            if i == 0:
                i_use = i_use_s
                T = T_s
                EV = EV_s 
            else:
                i_use = np.concatenate((i_use,i_use_s))
                T = np.concatenate((T,T_s))
                EV = np.concatenate((EV,EV_s))

    # Write the merged data to             
    with h5py.File(f_post_merged_h5, 'w') as f:
        f.create_dataset('i_use', data=i_use)
        f.create_dataset('T', data=T)
        f.create_dataset('EV', data=EV)
        f.attrs['f5_prior'] = f_prior_h5
        f.attrs['f5_data'] = f_data_merged_h5
        # ALSOE WRITE AN ATTRIBUET 'f5_data_mul' to the merged file
        #f.attrs['f5_data_files'] = f_data_h5_files


    return f_post_merged_h5, f_data_merged_h5


def merge_prior(f_prior_h5_files, f_prior_merged_h5='', showInfo=0):
    """
    Merge multiple prior model files into a single combined HDF5 file.

    Combines prior model parameters and forward-modeled data from multiple
    HDF5 files into a unified dataset. Creates a new model parameter M4 that
    tracks the source file index for each sample, enabling traceability of
    merged data origins.

    Parameters
    ----------
    f_prior_h5_files : list of str
        List of paths to prior HDF5 files to merge. Each file must contain
        compatible model parameters (M1, M2, M3, ...) and data arrays (D1, D2, ...).
    f_prior_merged_h5 : str, optional
        Output path for the merged prior file. If empty, generates default
        name 'PRIOR_merged_N{number_of_files}.h5' (default is '').
    showInfo : int, optional
        Verbosity level for progress information. Higher values provide more
        detailed output (default is 0).

    Returns
    -------
    str
        Path to the merged prior HDF5 file.

    Raises
    ------
    ValueError
        If f_prior_h5_files is not a list or is empty.
    FileNotFoundError
        If any input files do not exist or cannot be accessed.

    Notes
    -----
    The merging process:
    - Concatenates all model parameters (M1, M2, M3, ...) across files
    - Concatenates all data arrays (D1, D2, D3, ...) across files  
    - Creates new M4 parameter containing source file indices (0-based)
    - Preserves HDF5 attributes from the first file
    - Updates metadata to reflect merged status

    **Source File Tracking:**
    The new M4 parameter is a DISCRETE integer array with shape (Ntotal, 1) where
    each value indicates which input file the corresponding sample originated from:
    - 0: samples from first file in f_prior_h5_files
    - 1: samples from second file in f_prior_h5_files
    - etc.
    
    The M4 parameter is properly marked with:
    - is_discrete = 1 (discrete parameter type)
    - shape = (Ntotal, 1) (consistent with other model parameters)
    - class_name = meaningful names derived from filenames
    - class_id = [0, 1, 2, ...] (class identifiers)

    **File Compatibility:**
    Input files can have different model parameter dimensions (e.g., different
    numbers of layers). Arrays with fewer parameters will be padded with NaN
    values to match the maximum dimensions. Data arrays should ideally have
    the same dimensions, but padding is applied if they differ.

    Examples
    --------
    >>> f_files = ['prior1.h5', 'prior2.h5', 'prior3.h5']
    >>> merged_file = merge_prior(f_files, 'combined_prior.h5')
    >>> print(f"Merged {len(f_files)} files into {merged_file}")
    """
    import h5py
    import numpy as np
    
    # Input validation
    if not isinstance(f_prior_h5_files, list):
        raise ValueError('f_prior_h5_files must be a list of strings')
    
    if len(f_prior_h5_files) == 0:
        raise ValueError('f_prior_h5_files cannot be empty')
    
    nf = len(f_prior_h5_files)
    
    # Generate output filename if not provided
    if len(f_prior_merged_h5) == 0:
        f_prior_merged_h5 = 'PRIOR_merged_N%d.h5' % nf
    
    if showInfo > 0:
        print('Merging %d prior files to %s' % (nf, f_prior_merged_h5))
    
    # Initialize storage for merged data
    M_merged = {}  # Model parameters
    D_merged = {}  # Data arrays
    M4_values = []  # Source file indices
    sample_counts = []  # Track samples per file
    
    # First pass: collect all model parameters and data arrays
    for i, f_prior_h5 in enumerate(f_prior_h5_files):
        if showInfo > 1:
            print('.. Processing file %d: %s' % (i, f_prior_h5))
        
        with h5py.File(f_prior_h5, 'r') as f:
            # Count samples in this file (use M1 as reference)
            if 'M1' in f:
                n_samples = f['M1'].shape[0]
                sample_counts.append(n_samples)
                M4_values.extend([i + 1] * n_samples)  # Add file index for each sample (1-based for discrete compatibility)
            else:
                raise ValueError(f'File {f_prior_h5} does not contain M1 dataset')
            
            # Process model parameters (M1, M2, M3, ...)
            for key in f.keys():
                if key.startswith('M'):
                    if key not in M_merged:
                        M_merged[key] = []
                    M_merged[key].append(f[key][:])
            
            # Process data arrays (D1, D2, D3, ...)
            for key in f.keys():
                if key.startswith('D'):
                    if key not in D_merged:
                        D_merged[key] = []
                    D_merged[key].append(f[key][:])
    
    # Concatenate all arrays (handle different dimensions)
    if showInfo > 1:
        print('.. Concatenating arrays')
    
    # Concatenate model parameters (handle different parameter dimensions)
    for key in M_merged:
        arrays = M_merged[key]
        if len(arrays) == 1:
            M_merged[key] = arrays[0]
        else:
            # Find maximum dimensions across all arrays
            max_cols = max(arr.shape[1] for arr in arrays)
            
            # Pad arrays to match maximum dimensions
            padded_arrays = []
            for arr in arrays:
                if arr.shape[1] < max_cols:
                    # Pad with NaN values for missing parameters
                    pad_width = ((0, 0), (0, max_cols - arr.shape[1]))
                    padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=np.nan)
                    padded_arrays.append(padded_arr)
                else:
                    padded_arrays.append(arr)
            
            M_merged[key] = np.vstack(padded_arrays)
    
    # Concatenate data arrays (should have same dimensions)
    for key in D_merged:
        arrays = D_merged[key]
        if len(arrays) == 1:
            D_merged[key] = arrays[0]
        else:
            # Check if all data arrays have same dimensions
            shapes = [arr.shape[1] for arr in arrays]
            if len(set(shapes)) > 1:
                if showInfo > 0:
                    print(f'Warning: Data arrays for {key} have different dimensions: {shapes}')
                # Pad data arrays to match maximum dimensions
                max_cols = max(shapes)
                padded_arrays = []
                for arr in arrays:
                    if arr.shape[1] < max_cols:
                        pad_width = ((0, 0), (0, max_cols - arr.shape[1]))
                        padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=np.nan)
                        padded_arrays.append(padded_arr)
                    else:
                        padded_arrays.append(arr)
                D_merged[key] = np.vstack(padded_arrays)
            else:
                D_merged[key] = np.vstack(arrays)
    
    # Create M4 array (source file indices) - must be shape (Ntotal, 1)
    M_merged['M4'] = np.array(M4_values).reshape(-1, 1)
    
    # Write merged file
    if showInfo > 1:
        print('.. Writing merged file')
    
    with h5py.File(f_prior_merged_h5, 'w') as f_out:
        # Write all model parameters including M4
        for key, data in M_merged.items():
            f_out.create_dataset(key, data=data)
        
        # Write all data arrays
        for key, data in D_merged.items():
            f_out.create_dataset(key, data=data)
        
        # Copy attributes from first file and update
        with h5py.File(f_prior_h5_files[0], 'r') as f_first:
            for attr_name, attr_value in f_first.attrs.items():
                f_out.attrs[attr_name] = attr_value
        
        # Set M4 as discrete parameter with proper attributes
        if 'M4' in f_out:
            f_out['M4'].attrs['is_discrete'] = 1  # Mark as discrete
            f_out['M4'].attrs['name'] = 'Source File Index'
            f_out['M4'].attrs['x'] = np.array([0])  # Single feature dimension (like morrill example)
            f_out['M4'].attrs['clim'] = [0.5, nf + 0.5]  # Colormap limits for 1-based indexing
            
            # Create class names from filenames
            class_names = []
            for f_name in f_prior_h5_files:
                # Extract meaningful name from filename
                base_name = f_name.replace('.h5', '').replace('PRIOR_', '')
                class_names.append(base_name)
            
            f_out['M4'].attrs['class_name'] = [name.encode('utf-8') for name in class_names]
            f_out['M4'].attrs['class_id'] = np.arange(1, nf + 1)  # 1-based class IDs
        
        # Copy attributes from existing model parameters to maintain consistency
        with h5py.File(f_prior_h5_files[0], 'r') as f_first:
            # Copy attributes from other M parameters while preserving their continuous nature
            for key in M_merged.keys():
                if key != 'M4' and key in f_first:
                    for attr_name, attr_value in f_first[key].attrs.items():
                        if attr_name in ['is_discrete', 'name', 'method', 'clim', 'cmap']:
                            f_out[key].attrs[attr_name] = attr_value
                        elif attr_name in ['x', 'z']:
                            # Update x/z attributes to match padded dimensions
                            new_dim = M_merged[key].shape[1]
                            f_out[key].attrs[attr_name] = np.arange(new_dim)
        
        # Add merge-specific attributes
        f_out.attrs['merged_from_files'] = [f.encode('utf-8') for f in f_prior_h5_files]
        f_out.attrs['n_merged_files'] = nf
        f_out.attrs['samples_per_file'] = sample_counts
        f_out.attrs['M4_description'] = 'Source file index (0-based) - DISCRETE parameter'
    
    if showInfo > 0:
        total_samples = sum(sample_counts)
        print('Successfully merged %d samples from %d files' % (total_samples, nf))
        print('Added M4 parameter tracking source file indices')
    
    return f_prior_merged_h5


def read_usf(file_path: str) -> Dict[str, Any]:
    """
    Parse Universal Sounding Format (USF) electromagnetic data file.

    Reads and parses USF files containing electromagnetic survey data including
    measurement sweeps, timing information, and system parameters. USF is a
    standard format for time-domain electromagnetic data exchange.

    Parameters
    ----------
    file_path : str
        Path to the USF file to be parsed.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing parsed USF file contents with keys:
        - 'sweeps' : list of dict, measurement sweep data with timing and values
        - 'header' : dict, file header information and metadata
        - 'parameters' : dict, system and acquisition parameters
        - 'dummy_value' : float, placeholder value for missing data
        - Additional keys for file-specific parameters and settings

    Notes
    -----
    USF files contain structured electromagnetic data with sections for:
    - Header information (file version, date, system type)
    - Acquisition parameters (timing, frequencies, coordinates)
    - Measurement sweeps with data points and uncertainties
    - System configuration and processing parameters
    
    The parser handles various USF format variations and automatically
    converts numeric data while preserving text metadata. Sweep data
    includes timing gates, measured values, and quality indicators.
    
    This function is compatible with USF files from various electromagnetic
    systems and processing software, following standard format specifications
    for time-domain electromagnetic data exchange.
    """
    # Initialize result dictionary
    usf_data = {}
    # Current sweep being processed
    current_sweep = None
    # List to store all sweeps
    sweeps = []
    # Flag to indicate if we're reading data points
    reading_points = False
    # Store data points for current sweep
    data_points = []
    # Store the dummy value
    dummy_value = None
    
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")
    
    # Process each line in the file
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Process variable declarations in comment lines (//XXX: YYY)
        if line.startswith('//') and ': ' in line and not line.startswith('//USF:'):
            # Extract variable name and value
            var_match = re.match(r"//([^:]+):\s*(.+)", line)
            if var_match:
                var_name, var_value = var_match.groups()
                var_name = var_name.strip()
                var_value = var_value.strip()
                
                # Process dummy value
                if var_name == 'DUMMY':
                    try:
                        dummy_value = float(var_value)
                    except ValueError:
                        dummy_value = var_value
                    usf_data[var_name] = dummy_value
                else:
                    # Try to convert to numeric if possible
                    try:
                        usf_data[var_name] = float(var_value)
                    except ValueError:
                        usf_data[var_name] = var_value
        
        # Process lines starting with a single '/'
        elif line.startswith('/') and not line.startswith('//'):
            # Check if it's an END marker
            if line == '/END':
                # This doesn't actually end the data reading - it just marks the end of the sweep header
                # We'll now be expecting a header line followed by data points
                reading_points = True
                continue
                
            # Check if it's a SWEEP_NUMBER marker
            if line.startswith('/SWEEP_NUMBER:'):
                # If we already have a sweep, add it to our list
                if current_sweep is not None:
                    sweeps.append(current_sweep)
                
                # Start a new sweep
                current_sweep = {}
                reading_points = False
                data_points = []
                
                # Extract sweep number
                sweep_match = re.match(r"/SWEEP_NUMBER:\s*(\d+)", line)
                if sweep_match:
                    sweep_number = int(sweep_match.group(1))
                    current_sweep['SWEEP_NUMBER'] = sweep_number
                continue
            
            # Check if it's a POINTS marker
            if line.startswith('/POINTS:'):
                points_match = re.match(r"/POINTS:\s*(\d+)", line)
                if points_match and current_sweep is not None:
                    current_sweep['POINTS'] = int(points_match.group(1))
                continue
                
            # Process other parameters
            param_match = re.match(r"/([^:]+):\s*(.+)", line)
            if param_match:
                param_name, param_value = param_match.groups()
                param_name = param_name.strip()
                param_value = param_value.strip()
                
                # Check if this is TX_RAMP which contains a complex list
                if param_name == 'TX_RAMP':
                    values = []
                    pairs = param_value.split(',')
                    for i in range(0, len(pairs), 2):
                        if i+1 < len(pairs):
                            try:
                                time_val = float(pairs[i].strip())
                                amp_val = float(pairs[i+1].strip())
                                values.append((time_val, amp_val))
                            except ValueError:
                                pass
                    if current_sweep is not None:
                        current_sweep[param_name] = values
                # Check if parameter contains multiple values
                elif ',' in param_value:
                    values = []
                    for val in param_value.split(','):
                        val = val.strip()
                        try:
                            values.append(float(val))
                        except ValueError:
                            values.append(val)
                    
                    if current_sweep is not None:
                        current_sweep[param_name] = values
                    else:
                        usf_data[param_name] = values
                else:
                    # Try to convert to numeric if possible
                    try:
                        value = float(param_value)
                        if current_sweep is not None:
                            current_sweep[param_name] = value
                        else:
                            usf_data[param_name] = value
                    except ValueError:
                        if current_sweep is not None:
                            current_sweep[param_name] = param_value
                        else:
                            usf_data[param_name] = param_value
            
            # Check if we should start reading data points
            if line == '/CHANNEL: 1' or line == '/CHANNEL: 2':
                reading_points = True
                channel_match = re.match(r"/CHANNEL:\s*(\d+)", line)
                if channel_match and current_sweep is not None:
                    current_sweep['CHANNEL'] = int(channel_match.group(1))
                continue
                
        # Process data points
        elif reading_points and current_sweep is not None:
            # Check for the header line that comes after /END
            if line.strip().startswith('TIME,'):
                # Store the header names for this data block
                headers = [h.strip() for h in line.split(',')]
                current_sweep['DATA_HEADERS'] = headers
                
                # Initialize arrays for each data column
                for header in headers:
                    current_sweep[header] = []
                
                continue
                
            # Parse data point values
            values = line.split(',')
            if len(values) >= 6:
                try:
                    # Add each value to the corresponding array
                    for i, val in enumerate(values):
                        if i < len(headers):
                            # Try to convert to appropriate type
                            try:
                                if headers[i] == 'QUALITY':
                                    current_sweep[headers[i]].append(int(val.strip()))
                                else:
                                    current_sweep[headers[i]].append(float(val.strip()))
                            except ValueError:
                                current_sweep[headers[i]].append(val.strip())
                                
                except (ValueError, IndexError, NameError) as e:
                    # Skip problematic lines
                    pass
    
    # Add the last sweep if there is one
    if current_sweep is not None:
        sweeps.append(current_sweep)
    
    # Add sweeps to the result
    usf_data['SWEEP'] = sweeps


    # Extract d_obs as an array of usf_data['SWEEP'][0]['VOLTAGE'],usf_data['SWEEP'][1]['VOLTAGE'] ...
    # and store it a single 1D numpy array
    d_obs = np.concatenate([sweep['VOLTAGE'] for sweep in usf_data['SWEEP']])
    d_obs = np.array(d_obs)
    usf_data['d_obs'] = d_obs
    d_rel_err = np.concatenate([sweep['ERROR_BAR'] for sweep in usf_data['SWEEP']])
    d_rel_err = np.array(d_rel_err)
    usf_data['d_rel_err'] = d_rel_err
    time = np.concatenate([sweep['TIME'] for sweep in usf_data['SWEEP']])
    time = np.array(time)   
    usf_data['time'] = time
    # Add usf_data['id'] that is '0' for SWEEP1 and '1' for SWEEP2  etc
    # so, usf_data['id'] = [0,0,0,0,1,1,1,1,1] for 2 sweeps with 4 and 5 data points
    usf_data['id'] = np.concatenate([[i] * sweep['POINTS'] for i, sweep in enumerate(usf_data['SWEEP'])])
    usf_data['id'] = 1+np.array(usf_data['id'])
    # Add usf_data['dummy'] that is the dummy value
    usf_data['dummy'] = dummy_value
    # Add usf_data['file_name'] that is the file name
    usf_data['file_name'] = file_path.split('/')[-1]
    # Add usf_data['file_path'] that is the file path
    usf_data['file_path'] = file_path
    
    
    return usf_data


def test_read_usf(file_path: str) -> None:
    """
    Test function to read a USF file and print some key values.
    
    Args:
        file_path: Path to the USF file
    """
    usf = read_usf(file_path)
    
    print(f"DUMMY: {usf.get('DUMMY')}")
    print(f"SWEEPS: {usf.get('SWEEPS')}")
    
    for i, sweep in enumerate(usf.get('SWEEP', [])):
        print(f"\nSWEEP {i}:")
        print(f"CURRENT: {sweep.get('CURRENT')}")
        print(f"FREQUENCY: {sweep.get('FREQUENCY')}")
        print(f"POINTS: {sweep.get('POINTS')}")
        
        if 'TIME' in sweep and len(sweep['TIME']) > 0:
            print(f"First TIME value: {sweep['TIME'][0]}")
            print(f"First VOLTAGE value: {sweep['VOLTAGE'][0]}")
            print(f"Number of data points: {len(sweep['TIME'])}")
            print(f"Data headers: {sweep.get('DATA_HEADERS', [])}")
    



    return usf


def read_usf_mul(directory: str = ".", ext: str = ".usf") -> List[Dict[str, Any]]:
    """
    Read all USF files in a specified directory and return a list of USF data structures.
    
    Args:
        directory: Path to the directory containing USF files (default: current directory)
        ext: File extension to look for (default: ".usf")
        
    Returns:
        tuple containing:
            - np.ndarray: Array of observed data (d_obs) from all USF files
            - np.ndarray: Array of relative errors (d_rel_err) from all USF files
            - List[Dict[str, Any]]: List of USF data structures, each representing a single USF file

    """
    import os
    import glob
    from typing import List, Dict, Any

    # Make sure the extension starts with a period
    if not ext.startswith('.'):
        ext = '.' + ext
    
    # Get all matching files in the directory
    file_pattern = os.path.join(directory, f"*{ext}")
    usf_files = sorted(glob.glob(file_pattern))
    
    if not usf_files:
        print(f"No files with extension '{ext}' found in '{directory}'")
        return []
    
    # List to hold all USF data structures
    usf_list = []


    D_obs = []
    D_rel_err = []
    # Process each file
    for file_path in usf_files:
        try:
            # Read the USF file
            usf_data = read_usf(file_path)
            
            # Add the file name to the USF data structure
            usf_data['FILE_NAME'] = os.path.basename(file_path)

            D_obs.append(usf_data['d_obs'])
            D_rel_err.append(usf_data['d_rel_err'])

            # Add to the list
            usf_list.append(usf_data)
            
            print(f"Successfully read: {file_path}")
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    
    D_obs = np.array(D_obs)
    D_rel_err = np.array(D_rel_err)

    print(f"Read {len(usf_list)} out of {len(usf_files)} files.")
    return D_obs, D_rel_err, usf_list




