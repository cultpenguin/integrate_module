
def compute_P_obs_discrete(depth_top=None, depth_bottom=None, lithology_obs=None, z=None, class_id=None, lithology_prob=0.8, P_prior=None, W=None):
    """
    Compute discrete observation probability matrix from depth intervals and lithology observations.

    This function creates a probability matrix where each depth point is assigned
    probabilities based on observed lithology classes within specified depth intervals.

    Parameters
    ----------
    depth_top : array-like, optional
        Array of top depths for each observation interval. Required if W is not provided.
    depth_bottom : array-like, optional
        Array of bottom depths for each observation interval. Required if W is not provided.
    lithology_obs : array-like, optional
        Array of observed lithology class IDs for each interval. Required if W is not provided.
    z : array-like, optional
        Array of depth/position values where probabilities are computed. Required if W is not provided.
    class_id : array-like, optional
        Array of unique class identifiers (e.g., [0, 1, 2] for 3 lithology types). Required if W is not provided.
    lithology_prob : float or array-like, optional
        Probability assigned to the observed class. Can be:
        - float: Same probability for all intervals (default is 0.8)
        - array: Array of probabilities, one per interval (must match length of lithology_obs)
    P_prior : ndarray, optional
        Prior probability matrix of shape (nclass, nm). If None, uses uniform distribution
        for depths not covered by observations. Default is None.
    W : dict, optional
        Well/borehole dictionary containing observation data. If provided, overrides
        the individual parameters. Expected keys:
        - 'depth_top': Array of top depths
        - 'depth_bottom': Array of bottom depths
        - 'lithology_obs': Array of observed lithology class IDs
        - 'lithology_prob': Probability or array of probabilities (optional, defaults to 0.8)
        - 'X': X coordinate of well location (optional, not used in this function)
        - 'Y': Y coordinate of well location (optional, not used in this function)
        Default is None.

    Returns
    -------
    P_obs : ndarray
        Probability matrix of shape (nclass, nm) where nclass is the number of classes
        and nm is the number of depth points. For each depth point covered by observations,
        the observed class gets probability lithology_prob and other classes share (1-lithology_prob).
        Depths not covered by any observation contain NaN or prior probabilities if provided.

    Examples
    --------
    >>> # Traditional usage with individual parameters
    >>> depth_top = [0, 10, 20]
    >>> depth_bottom = [10, 20, 30]
    >>> lithology_obs = [1, 2, 1]  # clay, sand, clay
    >>> z = np.arange(30)
    >>> class_id = [0, 1, 2]  # gravel, clay, sand
    >>> P_obs = compute_P_obs_discrete(depth_top, depth_bottom, lithology_obs, z, class_id)
    >>> print(P_obs.shape)  # (3, 30)

    >>> # With different probabilities per interval
    >>> lithology_prob = [0.9, 0.7, 0.85]  # Higher confidence in first interval
    >>> P_obs = compute_P_obs_discrete(depth_top, depth_bottom, lithology_obs, z, class_id, lithology_prob=lithology_prob)

    >>> # Using well dictionary (cleaner interface)
    >>> W = {'depth_top': [0, 10, 20], 'depth_bottom': [10, 20, 30],
    ...      'lithology_obs': [1, 2, 1], 'lithology_prob': [0.9, 0.7, 0.85],
    ...      'X': 543000.0, 'Y': 6175800.0}
    >>> P_obs = compute_P_obs_discrete(z=z, class_id=class_id, W=W)
    """
    import numpy as np

    # Override parameters with W dictionary if provided
    if W is not None:
        if 'depth_top' in W:
            depth_top = W['depth_top']
        if 'depth_bottom' in W:
            depth_bottom = W['depth_bottom']
        if 'lithology_obs' in W:
            lithology_obs = W['lithology_obs']
        if 'lithology_prob' in W:
            lithology_prob = W['lithology_prob']
        # Note: X and Y coordinates stored for reference but not used in this function
        # X_well = W.get('X', None)
        # Y_well = W.get('Y', None)

    # Validate required parameters
    if depth_top is None or depth_bottom is None or lithology_obs is None:
        raise ValueError("depth_top, depth_bottom, and lithology_obs must be provided either as arguments or in W dictionary")
    if z is None or class_id is None:
        raise ValueError("z and class_id are required parameters")

    nm = len(z)
    nclass = len(class_id)

    # Convert lithology_prob to array if it's a scalar
    lithology_prob_array = np.atleast_1d(lithology_prob)
    if len(lithology_prob_array) == 1:
        # Scalar case: broadcast to all intervals
        lithology_prob_array = np.full(len(lithology_obs), lithology_prob_array[0])
    elif len(lithology_prob_array) != len(lithology_obs):
        raise ValueError(f"lithology_prob array length ({len(lithology_prob_array)}) must match lithology_obs length ({len(lithology_obs)})")

    # Initialize with NaN or prior
    if P_prior is not None:
        P_obs = P_prior.copy()
    else:
        P_obs = np.zeros((nclass, nm)) * np.nan

    # Loop through each depth point
    for im in range(nm):
        # Loop through each observation interval
        for i in range(len(depth_top)):
            # Check if current depth is within this interval
            if z[im] >= depth_top[i] and z[im] < depth_bottom[i]:
                # Get the probability for this specific interval
                lithology_prob_i = lithology_prob_array[i]
                # Compute probability for non-hit classes
                P_nohit = (1 - lithology_prob_i) / (nclass - 1)

                # Assign probabilities for all classes
                for ic in range(nclass):
                    if class_id[ic] == lithology_obs[i]:
                        P_obs[ic, im] = lithology_prob_i
                    else:
                        P_obs[ic, im] = P_nohit

    return P_obs

def _compute_mode_sequential(M_lithology, z, depth_top, depth_bottom, nl, showInfo=1):
    """
    Sequential computation of mode lithology for each realization and depth interval.

    This is the original implementation extracted for maintainability.
    """
    import numpy as np
    from tqdm import tqdm
    import time

    nreal = len(M_lithology)
    lithology_mode = np.zeros((nreal, nl), dtype=int)

    # Show info based on showInfo level
    if showInfo == 1:
        print(f'compute_P_obs_sparse: Processing {nreal} realizations, {nl} intervals')

    # Start timing
    t_start = time.time()

    # Extract mode lithology for each realization and depth interval
    iterator = np.arange(nreal)
    if showInfo > 1:
        iterator = tqdm(iterator, desc='compute_P_obs_sparse')

    for im in iterator:
        M_test = M_lithology[im]
        for i in range(len(depth_top)):
            z_top = depth_top[i]
            z_bottom = depth_bottom[i]
            id_top = np.argmin(np.abs(z - z_top))
            id_bottom = np.argmin(np.abs(z - z_bottom))

            if id_top == id_bottom:
                lithology_layer = M_test[id_top]
                lithology_mode_layer = lithology_layer
            else:
                lithology_layer = M_test[id_top:id_bottom]
                # Find the most frequent lithology in this layer
                values, counts = np.unique(lithology_layer, return_counts=True)
                lithology_mode_layer = values[np.argmax(counts)]

            lithology_mode[im, i] = lithology_mode_layer

    # Show timing information if showInfo > 1
    if showInfo > 1:
        t_elapsed = time.time() - t_start
        time_per_model = t_elapsed / nreal
        print(f'compute_P_obs_sparse: Total runtime: {t_elapsed:.2f} seconds')
        print(f'compute_P_obs_sparse: Time per model: {time_per_model*1000:.2f} ms')

    return lithology_mode

def _compute_mode_worker(args):
    """
    Worker function for parallel processing of lithology mode computation.

    Processes a chunk of realizations assigned to this worker.
    """
    import numpy as np
    import integrate as ig

    # Unpack arguments
    chunk_indices, shared_memory_refs, z, depth_top, depth_bottom, nl = args

    # Reconstruct shared array
    [M_lithology], worker_shm_objects = ig.reconstruct_shared_arrays(shared_memory_refs)

    try:
        # Initialize output for this chunk
        lithology_mode_chunk = np.zeros((len(chunk_indices), nl), dtype=int)

        # Process assigned realizations
        for local_idx, im in enumerate(chunk_indices):
            M_test = M_lithology[im]
            for i in range(nl):
                z_top = depth_top[i]
                z_bottom = depth_bottom[i]
                id_top = np.argmin(np.abs(z - z_top))
                id_bottom = np.argmin(np.abs(z - z_bottom))

                if id_top == id_bottom:
                    lithology_mode_chunk[local_idx, i] = M_test[id_top]
                else:
                    lithology_layer = M_test[id_top:id_bottom]
                    values, counts = np.unique(lithology_layer, return_counts=True)
                    lithology_mode_chunk[local_idx, i] = values[np.argmax(counts)]

        return lithology_mode_chunk

    finally:
        # Close shared memory in worker
        for shm in worker_shm_objects:
            shm.close()

def _compute_mode_parallel(M_lithology, z, depth_top, depth_bottom, nl, Ncpu, showInfo=1):
    """
    Parallel computation of mode lithology using multiprocessing.

    Splits realizations across worker processes and uses shared memory for efficiency.
    """
    import numpy as np
    import multiprocessing
    from multiprocessing import Pool
    import integrate as ig
    import time
    from tqdm import tqdm

    # Setup
    if Ncpu < 1:
        Ncpu = multiprocessing.cpu_count()

    nreal = len(M_lithology)

    # Show info based on showInfo level
    if showInfo == 1:
        print(f'compute_P_obs_sparse: Processing {nreal} realizations, {nl} intervals (parallel, {Ncpu} CPUs)')
    elif showInfo > 1:
        print(f'compute_P_obs_sparse: Processing {nreal} realizations, {nl} intervals')
        print(f'compute_P_obs_sparse: Using {Ncpu} CPU cores in parallel mode')

    # Start timing
    t_start = time.time()

    # Create shared memory for M_lithology
    shared_memory_refs, shm_objects = ig.create_shared_memory([M_lithology])

    try:
        # Split realizations into many small chunks for better progress tracking
        # More chunks = more frequent progress updates and better load balancing
        min_chunk_size = 100   # Minimum realizations per chunk
        max_chunks = 500       # Cap to avoid excessive overhead
        n_chunks = min(max(nreal // min_chunk_size, Ncpu), max_chunks)

        if showInfo > 1:
            print(f'compute_P_obs_sparse: Splitting into {n_chunks} chunks for {Ncpu} workers')

        realization_chunks = np.array_split(np.arange(nreal), n_chunks)

        # Create worker arguments (include small arrays directly)
        worker_args = [
            (chunk_indices, shared_memory_refs, z, depth_top, depth_bottom, nl)
            for chunk_indices in realization_chunks
        ]

        # Execute in parallel
        with Pool(processes=Ncpu) as p:
            if showInfo > 1:
                # Use imap to get results as they complete and show progress
                results = list(tqdm(
                    p.imap(_compute_mode_worker, worker_args),
                    total=len(worker_args),
                    desc='compute_P_obs_sparse (parallel chunks)',
                    unit='chunk'
                ))
            else:
                # Use regular map without progress tracking
                results = p.map(_compute_mode_worker, worker_args)

        # Concatenate results
        lithology_mode = np.concatenate(results, axis=0)

        # Show timing information if showInfo > 1
        if showInfo > 1:
            t_elapsed = time.time() - t_start
            time_per_model = t_elapsed / nreal
            print(f'compute_P_obs_sparse: Total runtime: {t_elapsed:.2f} seconds')
            print(f'compute_P_obs_sparse: Time per model: {time_per_model*1000:.2f} ms')
            print(f'compute_P_obs_sparse: Speedup with {Ncpu} cores vs sequential: ~{Ncpu*0.7:.1f}x (estimated)')

        return lithology_mode

    finally:
        # Cleanup shared memory
        ig.cleanup_shared_memory(shm_objects)

def compute_P_obs_sparse(M_lithology, depth_top=None, depth_bottom=None, lithology_obs=None, z=None, class_id=None, lithology_prob=0.8, W=None, parallel=False, Ncpu=-1, showInfo=1):
    """
    Compute sparse observation probability matrix by extracting mode lithology from prior models within depth intervals.

    This function processes lithology models from a prior ensemble to create sparse observations.
    For each depth interval, it finds the most frequent (mode) lithology class within that interval
    from each prior model, then creates a probability matrix based on how well these modes match
    the observed lithology.

    Parameters
    ----------
    M_lithology : ndarray
        Array of lithology models from prior ensemble, shape (nreal, nz) where nreal is the
        number of realizations and nz is the number of depth points.
    depth_top : array-like, optional
        Array of top depths for each observation interval. Required if W is not provided.
    depth_bottom : array-like, optional
        Array of bottom depths for each observation interval. Required if W is not provided.
    lithology_obs : array-like, optional
        Array of observed lithology class IDs for each interval. Required if W is not provided.
    z : array-like, optional
        Array of depth/position values corresponding to M_lithology depth discretization.
        Required if W is not provided.
    class_id : array-like, optional
        Array of unique class identifiers (e.g., [0, 1, 2] for 3 lithology types).
        Required if W is not provided.
    lithology_prob : float or array-like, optional
        Probability assigned to the observed class. Can be:
        - float: Same probability for all intervals (default is 0.8)
        - array: Array of probabilities, one per interval (must match length of lithology_obs)
    W : dict, optional
        Well/borehole dictionary containing observation data. If provided, overrides
        the individual parameters. Expected keys:
        - 'depth_top': Array of top depths
        - 'depth_bottom': Array of bottom depths
        - 'lithology_obs': Array of observed lithology class IDs
        - 'lithology_prob': Probability or array of probabilities (optional, defaults to 0.8)
        - 'X': X coordinate of well location (optional, not used in this function)
        - 'Y': Y coordinate of well location (optional, not used in this function)
        Default is None.
    parallel : bool, optional
        Enable parallel processing for large ensembles. Default is False.
        When True and parallel processing is available, distributes realization
        processing across multiple CPU cores for significant speedup.
        Recommended for N > 10,000 realizations.
    Ncpu : int, optional
        Number of CPU cores to use for parallel processing. Default is -1 (auto-detect).
        Only used when parallel=True. Set to specific value to control parallelism.
    showInfo : int, optional
        Control information output level. Default is 1.
        - 0: No information printed
        - 1: Single line info (number of realizations, intervals)
        - >1: Progress bar with tqdm, runtime statistics, and time per model

    Returns
    -------
    P_obs : ndarray
        Probability matrix of shape (nclass, n_obs) where nclass is the number of classes
        and n_obs is the number of observation intervals. Each column represents the
        probability distribution for one depth interval.
    lithology_mode : ndarray
        Array of mode lithology values extracted from prior models, shape (nreal, n_obs).
        For each realization and observation interval, contains the most frequent lithology
        class ID within that depth range.

    Examples
    --------
    >>> # Load prior lithology models
    >>> M_lithology = f_prior['M2'][:]  # Shape: (100000, 50)
    >>> z = np.linspace(0, 100, 50)
    >>> class_id = [0, 1, 2]  # sand, clay, gravel
    >>>
    >>> # Define observations
    >>> depth_top = [0, 20, 40]
    >>> depth_bottom = [20, 40, 60]
    >>> lithology_obs = [1, 0, 1]  # clay, sand, clay
    >>>
    >>> # Compute sparse observations
    >>> P_obs, lithology_mode = compute_P_obs_sparse(M_lithology, depth_top, depth_bottom,
    ...                                               lithology_obs, z, class_id)
    >>> print(P_obs.shape)  # (3, 3) - 3 classes, 3 observations
    >>> print(lithology_mode.shape)  # (100000, 3) - mode for each realization and interval

    >>> # Using well dictionary
    >>> W = {'depth_top': [0, 20, 40], 'depth_bottom': [20, 40, 60],
    ...      'lithology_obs': [1, 0, 1], 'lithology_prob': [0.9, 0.8, 0.85],
    ...      'X': 543000.0, 'Y': 6175800.0}
    >>> P_obs, lithology_mode = compute_P_obs_sparse(M_lithology, z=z, class_id=class_id, W=W)
    >>>
    >>> # Using parallel processing for large ensembles
    >>> P_obs, lithology_mode = compute_P_obs_sparse(M_lithology, depth_top, depth_bottom,
    ...                                               lithology_obs, z, class_id,
    ...                                               parallel=True, Ncpu=8)
    >>>
    >>> # Control output verbosity
    >>> P_obs, lithology_mode = compute_P_obs_sparse(..., showInfo=0)  # Silent
    >>> P_obs, lithology_mode = compute_P_obs_sparse(..., showInfo=1)  # Single line info
    >>> P_obs, lithology_mode = compute_P_obs_sparse(..., showInfo=2)  # Progress bar + timing

    Notes
    -----
    The function extracts lithology mode for each depth interval by:
    1. Finding depth indices corresponding to interval boundaries
    2. Extracting lithology values within the interval
    3. Computing the most frequent (mode) lithology class
    4. Assigning probabilities based on match with observed lithology

    This approach creates "sparse" observations because instead of using the full depth profile,
    only the mode lithology from each interval is used to construct the probability matrix.

    Parallel processing uses shared memory for M_lithology array to minimize memory overhead.
    Expected speedup: 4-8x on 8-core machines for large ensembles (N > 100,000 realizations).
    The parallel implementation distributes realizations across worker processes while
    maintaining identical results to the sequential version.

    Both sequential and parallel modes support timing information via showInfo parameter:
    - showInfo=0: Silent mode
    - showInfo=1: Single line summary (default)
    - showInfo>1: Detailed timing (progress bar for sequential, runtime stats for both)
    """
    import numpy as np

    # Override parameters with W dictionary if provided
    if W is not None:
        if 'depth_top' in W:
            depth_top = W['depth_top']
        if 'depth_bottom' in W:
            depth_bottom = W['depth_bottom']
        if 'lithology_obs' in W:
            lithology_obs = W['lithology_obs']
        if 'lithology_prob' in W:
            lithology_prob = W['lithology_prob']
        # Note: X and Y coordinates stored for reference but not used in this function
        # X_well = W.get('X', None)
        # Y_well = W.get('Y', None)

    # Validate required parameters
    if depth_top is None or depth_bottom is None or lithology_obs is None:
        raise ValueError("depth_top, depth_bottom, and lithology_obs must be provided either as arguments or in W dictionary")
    if z is None or class_id is None:
        raise ValueError("z and class_id are required parameters")

    # Get dimensions
    nreal = len(M_lithology)
    nclass = len(class_id)
    nl = len(lithology_obs)
    n_obs = nl

    # Convert lithology_prob to array if it's a scalar
    lithology_prob_array = np.atleast_1d(lithology_prob)
    if len(lithology_prob_array) == 1:
        # Scalar case: broadcast to all intervals
        lithology_prob_array = np.full(n_obs, lithology_prob_array[0])
    elif len(lithology_prob_array) != n_obs:
        raise ValueError(f"lithology_prob array length ({len(lithology_prob_array)}) must match lithology_obs length ({n_obs})")

    # Compute mode lithology using sequential or parallel method
    import integrate as ig
    if parallel and ig.use_parallel():
        # Parallel execution path
        lithology_mode = _compute_mode_parallel(M_lithology, z, depth_top, depth_bottom, nl, Ncpu, showInfo)
    else:
        # Sequential execution path (original implementation)
        lithology_mode = _compute_mode_sequential(M_lithology, z, depth_top, depth_bottom, nl, showInfo)

    # Convert observed lithologies to P_obs probabilities
    P_obs = np.zeros((nclass, n_obs)) * np.nan
    for i in range(n_obs):
        # Get the probability for this specific interval
        lithology_prob_i = lithology_prob_array[i]

        for j in range(nclass):
            if class_id[j] == lithology_obs[i]:
                P_obs[j, i] = lithology_prob_i
            else:
                P_obs[j, i] = (1 - lithology_prob_i) / (nclass - 1)

    return P_obs, lithology_mode

def rescale_P_obs_temperature(P_obs, T=1.0):
    """
    Rescale discrete observation probabilities by temperature and renormalize.

    This function applies temperature annealing to probability distributions by raising
    each probability to the power (1/T), then renormalizing each column (depth point)
    so that probabilities sum to 1. Higher temperatures (T > 1) flatten the distribution,
    while lower temperatures (T < 1) sharpen it.

    Parameters
    ----------
    P_obs : ndarray
        Probability matrix of shape (nclass, nm) where nclass is the number of classes
        and nm is the number of model parameters (e.g., depth points).
        Each column should represent a probability distribution over classes.
    T : float, optional
        Temperature parameter for annealing. Default is 1.0 (no scaling).
        - T = 1.0: No change (original probabilities)
        - T > 1.0: Flattens distribution (less certain)
        - T < 1.0: Sharpens distribution (more certain)
        - T → ∞: Approaches uniform distribution
        - T → 0: Approaches one-hot distribution

    Returns
    -------
    P_obs_scaled : ndarray
        Temperature-scaled and renormalized probability matrix of shape (nclass, nm).
        Each column sums to 1.0. NaN values in input are preserved in output.

    Examples
    --------
    >>> P_obs = np.array([[0.8, 0.6, 0.5],
    ...                   [0.1, 0.2, 0.3],
    ...                   [0.1, 0.2, 0.2]])
    >>> P_scaled = rescale_P_obs_temperature(P_obs, T=2.0)
    >>> print(P_scaled)  # More uniform distribution
    >>> P_scaled = rescale_P_obs_temperature(P_obs, T=0.5)
    >>> print(P_scaled)  # Sharper distribution

    Notes
    -----
    The temperature scaling follows the Boltzmann distribution:
        P_new(c) ∝ P_old(c)^(1/T)

    After scaling, each column (depth point) is renormalized:
        P_new(c) = P_new(c) / sum_c(P_new(c))

    This is commonly used in simulated annealing and rejection sampling to control
    the strength of discrete observations during Bayesian inference.
    """
    import numpy as np

    # Copy to avoid modifying the original
    P_obs_scaled = P_obs.copy()

    # Get shape
    nclass, nm = P_obs.shape

    # Apply temperature scaling: p^(1/T)
    # Handle special case where T=1 (no scaling needed)
    if T != 1.0:
        P_obs_scaled = np.power(P_obs_scaled, 1.0 / T)

    # Renormalize each column (each depth point) to sum to 1
    for im in range(nm):
        col_sum = np.nansum(P_obs_scaled[:, im])

        # Only renormalize if the sum is non-zero and not NaN
        if col_sum > 0 and not np.isnan(col_sum):
            P_obs_scaled[:, im] = P_obs_scaled[:, im] / col_sum

    return P_obs_scaled

def Pobs_to_datagrid(P_obs, X, Y, f_data_h5, r_data=10, r_dis=100, doPlot=False):
    """
    Convert point-based discrete probability observations to gridded data with distance-based weighting.

    This function distributes discrete probability observations (e.g., from a borehole) across
    a spatial grid using distance-based weighting. Observations at location (X, Y) are applied
    to nearby grid points with decreasing influence based on distance. Temperature annealing
    is used to reduce the strength of observations far from the source point.

    Parameters
    ----------
    P_obs : ndarray
        Probability matrix of shape (nclass, nm) where nclass is the number of classes
        and nm is the number of model parameters (e.g., depth points).
        Each column represents a probability distribution over discrete classes.
    X : float
        X coordinate (e.g., UTM Easting) of the observation point.
    Y : float
        Y coordinate (e.g., UTM Northing) of the observation point.
    f_data_h5 : str
        Path to HDF5 data file containing survey geometry (X, Y coordinates).
    r_data : float, optional
        Inner radius in meters within which observations have full strength.
        Default is 10 meters.
    r_dis : float, optional
        Outer radius in meters for distance-based weighting. Beyond this distance,
        observations are fully attenuated (temperature → ∞). Default is 100 meters.
    doPlot : bool, optional
        If True, creates diagnostic plots showing weight distributions.
        Default is False.

    Returns
    -------
    d_obs : ndarray
        Gridded observation data of shape (nd, nclass, nm) where nd is the number
        of spatial locations in the survey. Each location gets temperature-scaled
        probabilities based on distance from (X, Y).
    i_use : ndarray
        Binary mask of shape (nd, 1) indicating which grid points should be used
        (1) or ignored (0) in the inversion. Points with temperature < 100 are used.

    Notes
    -----
    The function uses distance-based temperature annealing:
    1. Computes distance-based weights using `get_weight_from_position()`
    2. Converts distance weight to temperature: T = 1 / w_dis
    3. Caps maximum temperature at 100 (very weak influence)
    4. For each grid point:
       - If T < 100: include point (i_use=1) and apply temperature scaling
       - If T ≥ 100: exclude point (i_use=0) and set observations to NaN

    Temperature scaling reduces probability certainty with distance:
    - T = 1 (close to observation): Original probabilities preserved
    - T > 1 (far from observation): Probabilities become more uniform
    - T ≥ 100 (very far): Observations effectively ignored

    Examples
    --------
    >>> # Borehole observation at specific location
    >>> P_obs = compute_P_obs_discrete(depth_top, depth_bottom, lithology, z, class_id)
    >>> X_well, Y_well = 543000.0, 6175800.0
    >>> d_obs, i_use = Pobs_to_datagrid(P_obs, X_well, Y_well, 'survey_data.h5',
    ...                                  r_data=10, r_dis=100)
    >>> # Write to data file
    >>> ig.save_data_multinomial(d_obs, i_use=i_use, id=2, f_data_h5='survey_data.h5')

    See Also
    --------
    rescale_P_obs_temperature : Temperature scaling function
    compute_P_obs_discrete : Create P_obs from depth intervals
    get_weight_from_position : Distance-based weighting function
    """
    import numpy as np
    import integrate as ig

    # Get grid dimensions from data file
    X_grid, Y_grid, _, _ = ig.get_geometry(f_data_h5)
    nd = len(X_grid)
    nclass, nm = P_obs.shape

    # Initialize output arrays
    i_use = np.zeros((nd, 1))
    d_obs = np.zeros((nd, nclass, nm)) * np.nan

    # Compute distance-based weights for all grid points
    w_combined, w_dis, w_data, i_use_from_func = ig.get_weight_from_position(
        f_data_h5, X, Y, r_data=r_data, r_dis=r_dis, doPlot=doPlot
    )

    # Convert distance weight to temperature
    # w_dis is 1 at observation point, decreases with distance
    # T = 1/w_dis means T increases with distance (weaker influence)
    T_all = 1 / w_dis

    # Cap maximum temperature at 100 (beyond this, observation has negligible effect)
    T_all[T_all > 100] = 100

    # Apply temperature scaling to each grid point
    for ip in np.arange(nd):
        T = T_all[ip]

        # Only use points where temperature is reasonable (< 100)
        if T < 100:
            i_use[ip] = 1
            # Scale probabilities based on distance (higher T = more uniform distribution)
            P_obs_local = rescale_P_obs_temperature(P_obs, T=T)
            d_obs[ip, :, :] = P_obs_local
        # else: i_use[ip] = 0 and d_obs[ip] stays NaN

    return d_obs, i_use