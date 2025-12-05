
def compute_P_obs_discrete(depth_top, depth_bottom, lithology_obs, z, class_id, P_single=0.8, P_prior=None):
    """
    Compute discrete observation probability matrix from depth intervals and lithology observations.
    
    This function creates a probability matrix where each depth point is assigned 
    probabilities based on observed lithology classes within specified depth intervals.
    
    Parameters
    ----------
    depth_top : array-like
        Array of top depths for each observation interval.
    depth_bottom : array-like
        Array of bottom depths for each observation interval.
    lithology_obs : array-like
        Array of observed lithology class IDs for each interval.
    z : array-like
        Array of depth/position values where probabilities are computed.
    class_id : array-like
        Array of unique class identifiers (e.g., [0, 1, 2] for 3 lithology types).
    P_single : float, optional
        Probability assigned to the observed class. Default is 0.8.
    P_prior : ndarray, optional
        Prior probability matrix of shape (nclass, nm). If None, uses uniform distribution
        for depths not covered by observations. Default is None.
    
    Returns
    -------
    P_obs : ndarray
        Probability matrix of shape (nclass, nm) where nclass is the number of classes
        and nm is the number of depth points. For each depth point covered by observations,
        the observed class gets probability P_single and other classes share (1-P_single).
        Depths not covered by any observation contain NaN or prior probabilities if provided.
    
    Examples
    --------
    >>> depth_top = [0, 10, 20]
    >>> depth_bottom = [10, 20, 30]
    >>> lithology_obs = [1, 2, 1]  # clay, sand, clay
    >>> z = np.arange(30)
    >>> class_id = [0, 1, 2]  # gravel, clay, sand
    >>> P_obs = compute_P_obs_discrete(depth_top, depth_bottom, lithology_obs, z, class_id)
    >>> print(P_obs.shape)  # (3, 30)
    """
    import numpy as np
    
    nm = len(z)
    nclass = len(class_id)
    
    # Compute probability for non-hit classes
    P_nohit = (1 - P_single) / (nclass - 1)
    
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
                # Assign probabilities for all classes
                for ic in range(nclass):
                    if class_id[ic] == lithology_obs[i]:
                        P_obs[ic, im] = P_single
                    else: 
                        P_obs[ic, im] = P_nohit
    
    return P_obs

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