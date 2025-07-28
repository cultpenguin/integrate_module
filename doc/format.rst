=============
Data format
=============

The HDF5_ file format is used as a container for all data in INTEGRATE. 
Each HDF5 file can contain multiple data sets (typically arranged in matrix format) with associated attributes that describe the data. `HDF View`_ is usefull to inspect the content of HDF5 files.

.. _HDF View: https://www.hdfgroup.org/downloads/hdfview/
.. _HDf5: https://www.hdfgroup.org/solutions/hdf5/


The following HDF files are used for any INTEGRATE project 

**DATA.h5** : Stores observed data, and its associated geometry.

**PRIOR.h5**: Stores realizations of the prior model, and corresponding forward response

**FORWARD.h5**: Stores information need to solved the forward, problem, and/or needed to describe the observed data in DATA.h5

**POST.h5**: stores index of posterior realizations, as well as posterior statistics 


DATA.h5
=======
DATA.h5 contains observed data, and its associated geometry. 
The observed data can be of many types, such as TEM data and well-log data


The ID of the observed data, must, for now, be the same as the ID of the prior data. 
Thus, the observed data in f_data:``D1/`` must refer to the prior data in f_prior:``D1``. 

  
  ``Np``: Number of data locations (typically one set data per unique X-Y location)
  
  ``Ndi``: Number of data points ``Nd`` per data type ``i`` per location.
  
  ``Nclass``: Number of classes

The datasets ``UTMX``, ``UTMY``, ``ELEVATION``, and ``LINE`` are mandatory for most plotting routines in INTEGRATE, 
but are not used in the inversion itself.

The attribute ``D1/noise_model`` is mandatory for all data types, and describes the noise model used for the data.

.. list-table:: Data and attributes for DATA.h5
   :widths: 10 10 5 5 70 
   :header-rows: 1

   * - Dataset
     - Format
     - Attribute
     - Mandatory
     - Description
   * - /UTMX
     - [Np,1]
     - 
     - (*)
     - X - location of data points

   * - /UTMY
     - [Np,1]
     - 
     - (*)
     - Y - location of data points    
   * - /ELEVATION
     - [Np,1]
     - 
     - (*)
     - Elevation at data points    
   * - /LINE
     - [Np,1]
     - 
     - (*)
     - Linenumber at data points    
   * - /D1/noise_model
     - [string]
     - *
     - *
     - A string describing the noise model used for the data. 
   * - /D1/id_use
     - [integer]
     - 
     - 
     - The prior data if A string describing the noise model used for the data. If not set it will the same id as the data id 
   * - /D1/i_use
     - [NP,1] int [0/1]
     - 
     - 
     - Determines wether a data point should be used or not. All data are used by default

The format of the observed data, and the associate uncertainty, depends on the type of data, and the choice of noise model.

See the function :func:`integrate.load_data()` for an example on how read DATA.h5 files.

"""""""""""""""""""""""""""""""""
Gaussian noise  - continuous data
"""""""""""""""""""""""""""""""""

For continuous data and the multivariate Gaussian noise model can be chosen by setting the attribute ``D1/noise_model=gaussian`` 

.. list-table:: Data and attributes om DATA.h5 for continuous data and multivariate Gaussian noise model
   :widths: 10 10 5 5 70 
   :header-rows: 1

   * - Dataset
     - Format
     - Attribute
     - Mandatory
     - Description
   * - /D1/noise_model
     - [string]='gaussian'
     - *
     - *
     - A string describing the noise model used for the data. Here ``'gaussian'`` to represent a multivariate Gaussian noise model.
   * - /D1/d_obs
     - [Np,Nd1]
     - 
     - *
     - Observed data (#1)
   * - /D1/d_std
     - [Np,Nd1]
     - 
     - *
     - Standard deviation of observed data (db/dT). Is the size is [1,Nd], the same ``d_std`` is used for all data.
   * - /D1/Ct
     - [Nd1,Nd1]
     - 
     - 
     - Correlated noise matrix. ``Ct`` is the same for all data
   * - /D1/Ct
     - [Np,Nd1,Nd1]
     - 
     - 
     - Correlated noise matrix; each data observation has its own correlated noise matrix 
   * - /gatetimes
     - [Ndata,1]
     - 
     - 
     - Gate times (in seconds) for each data point
   * - /i_lm
     - [Nlm,1]
     - 
     - 
     - Index (rel to /gatetimes) of Nlm gates for the low moment. 
   * - /i_hm
     - [Nhm,1]
     - 
     - 
     - Index (rel to /gatetimes) of Nhm gates for the high moment. 
   * - --
     - 
     - 
     - 
     - 

"""""""""""""""""""""""""""""""""
Multinomial noise - discrete data
"""""""""""""""""""""""""""""""""

For discrete data the multinomial distribution can use as likelihhood by setting the attribute ``D1/noise_model=multinomial`` 

.. list-table:: Data and attributes om DATA.h5 for  data and multinomial noise model
   :widths: 10 10 5 5 70 
   :header-rows: 1

   * - Dataset
     - Format
     - Attribute
     - Mandatory
     - Description
   * - /D2/noise_model
     - [string]='multinomial'
     - *
     - *
     - The multinomial distribution is used as likelihood model for the data.
   * - /D2/d_obs
     - [Np,Nclass,Nm]
     - 
     - *
     - Observed data (class probabilities)
   * - /D2/i_group
     - [Np,Nm]
     - 
     - 
     - Indicates whether the Nd2 data should considered as groups or individually.
   * - /D2/i_use
     - [Np,1]
     - 
     - 
     - Binary indicator of whether a data point should be used or not



PRIOR 
=====

PRIOR.h5 contains ``N`` realizations of a prior model (represented as potentially multiple types of model parameters, such as resistivity, lithology, grainsize,....), and corresponding data (consisting of potentially multiple types of data, such as tTEM, SkyTEM, WellData,..)

``N``: Number of realizations of the prior model

``Nm1``: Number of model parameters of type 1

``Nm2``: Number of model parameters of type 2

``NmX``: Number of model parameters of type X


.. list-table:: PRIOR model realizations in PRIOR.h5
   :widths: 10 10 5 5 70 
   :header-rows: 1

   * - Dataset
     - Format
     - attribute
     - Mandatory
     - Description
   * - /M1
     - [N,Nm1]
     - 
     - *
     - N realizations of model parameter 1, 
       each consisting of Nm1 model param1eters
   * - /M1/x
     - [nm]
     - *
     - *
     - Array of values describing each value in M1 (e.g. depth to layer)
   * - /M1/name
     - [string]
     - *
     - 
     - Name of model parameter /M1
   * - /M1/is_discrete
     - [nm]
     - *
     - *
     - [0/1] described whether /M1 is a discrete or continuous parameter
   * - /M1/class_id
     - [1,n_class]
     - *
     - 
     - A list of  ``n_class`` class id; only when case /M1/is_discrete=1.
   * - /M1/class_name
     - [1,n_class]
     - *
     - 
     - A list of ``n_class`` strings describing each class; only when case /M1/is_discrete=1.
   * - /M1/clim
     - [1,2]
     - *
     - 
     - Min and maximum value for colorbar
   * - /M1/cmap
     - [3,nlev]
     - *
     - 
     - Colormap with ``nlev`` levels.
   * - /M2
     -  [N,Nm2]
     - 
     - 
     - N realizations of model parameter 2, 
       each consisting of Nm2 model parameters
   * - /Mx
     -  [N,NmX]
     - 
     - 
     - N realizations of model parameter X, 
       each consisting of NmX model parameters



.. list-table:: prior data realizations in PRIOR.h5
   :widths: 10 10 5 5 70 
   :header-rows: 1

   * - Dataset
     - Format
     - attribute
     - Mandatory
     - Description
   * - /D1
     - [N,Nd1]
     - 
     - *
     - N realizations of data number 1, 
       each consisting of ``Nd`` model parameters
   * - /D1/f5_forward
     - [string]
     - *
     - 
     - HDF file describing the forward model used to compute prior data.
   * - /D1/with_noise
     - [1]
     - *
     - 
     - Indicates whether noise was added to the data[1] or not[0].
   * - /D2
     -  [N,Nd2]
     - 
     - 
     - N realizations of data number 2, 
       each consisting of ``Nd2`` model parameters
     

``/D1`` is only mandatory when PRIOR.h5 is used for inversion

All the mandatory attributes specified for ``/M1`` are also mandatory for other attributes, i.e.  ``/M1``,  ``/M2``, ... . 


f_forward_h5 [string]: Defines the name of the HDF5 file that contains information need to solved the forward problem...



FORWARD.h5
==========
The FORWARD.h5 needs to hold' as much information as needed to define the use fo a specific forward model.

The attribute ``/method`` refer to a specific choice of forward method.


.. list-table:: posterior data realizations in PRIOR.h5
   :widths: 10 10 5 5 70 
   :header-rows: 1

   * - Dataset
     - Format
     - attribute
     - Mandatory
     - Description
   * - /method
     - [string]
     - *
     - 
     - Defines the type of forward model def:'TDEM'.
   * - /type
     - [string]
     - *
     - 
     - Define the algorithm used to solve the forward model. def:'GA-AEM'.
     

``/method`` can, for example, be ``TDEM`` for Time Domain EM (The default in INTEGRATE),
ot can be ``identity`` for an identity mapping (useful to represent log data).

TDEM: Time domain EM, method='tdem'.
------------------------------------

``/method='TDEM'`` make use of time-domain EM forward modeling. 
The following three types of forward models will (eventually) be available:


``/type='GA-AEM'`` [DEFAULT].
[GA-AEM]_. Available for both Linux and Windows, Matlab and Python.


``/type='AarhusInv'``.
[AarhusInv]_. Windows only.
Not yet implemented


``/type='SimPEG'``.
[SimPEG]_. Python only.

LOG: Well log conditioning, method='log'
----------------------------------------

``/method='identity'`` maps attributes of a specific model (realizations of the prior) directly into data. 
  

POST - :samp:`f_post_h5`
========================

At the very minimum POST.h5 needs to contain the index (in PRIOR.h5) of realizations from the posterior

.. list-table:: Data and attributes in POST.h5
   :widths: 10 10 5 5 70 
   :header-rows: 1

   * - Dataset     
     - Format
     - attribute
     - Mandatory
     - Description     
   * - /i_use
     - [N,Nr]
     - 
     - *
     - Index of posterior realizations for each data 
   * - /T
     -  [N,1]
     - 
     - *
     - The annealing temperature used for inversion
   * - /EV
     -  [N,1]
     - 
     - *
     - Evidence
   * - /f5_data
     - F [string]
     - *
     - *
     - Filename of HDF5 data file.
   * - /f5_prior
     - F [string]
     - *
     - *
     - Filename of HDF5 PRIOR file.






Continious parameters
---------------------

For continuous model parameters the following generic posterior statistics are computed

.. list-table:: Data and attributes for continuous parameters in POST.h5
   :widths: 10 10 5 5 70 
   :header-rows: 1

   * - Dataset     
     - Format
     - attribute
     - Mandatory
     - Description     
   * - /M1/Mean
     - [N,Nm]
     - 
     - 
     - Point-wise mean of the posterior
   * - /M1/Median
     - [N,Nm]
     - 
     - 
     - Point-wise median of the posterior
   * - /M1/Std
     - [N,Nm]
     - 
     - 
     - Point-wise standard deviation of the posterior





Discrete parameters
-------------------


For discrete model parameters the following generic posterior statistics are computed


.. list-table:: Data and attributes for discrete parameters in POST.h5
   :widths: 10 10 5 5 70 
   :header-rows: 1

   * - Dataset     
     - Format
     - attribute
     - Mandatory
     - Description     
   * - /M1/Mode
     - [N,Nm]
     - 
     - 
     - Point-wise mode of the posterior
   * - /M1/Entropy
     - [N,Nm]
     - 
     - 
     - Point-wise entropy of the posterior
   * - /M1/P
     - [N,Nclass,Nm]
     - 
     - 
     - Pointwise posterior probability of each class.
   * - /M1/M_N
     - [N,Nclass]
     - 
     - 
     - Median number of layers with specific class_id
     



A typical workflow
==================
1. Setup DATA.h5
   
   * Store the observed data and its associated uncertainty in DATA.h5

2. Setup FORWARD.h5

   * Define the forward problem for data type A in FORWARD_A.h5.
   * Define the forward problem for data type B in FORWARD_B.h5.

3. Setup PRIOR.h5

   * Generate prior model realizations of model parameter 1 in in /M1
   * Generate prior model realizations of model parameter 2 in in /M2
   * Use FORWARD_A.h5 to compute prior data of the prior realizations for data type A
   * Use FORWARD_A.h5 to compute prior data of the prior realizations for data type B
  
4. Sample the posterior and output POST.h5

5. Update POST.h5 with some statistics computed from the posterior.
