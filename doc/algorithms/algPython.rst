=================
Python Algorithms 
=================


integrate_rejection.py 
----------------------
This function implements the rejection sampler 

..  code-block:: python
    
    integrate_rejection.py [f_prior_h5] [f_data_h5] [-h] [--autoT AUTOT] [--N_use N_USE] [--ns NS] [--parallel PARALLEL] [--updatePostStat   1] 

By default `autoT=1`, `N_use=1000000`, `ns=1000`, `parallel=1`, and 'updatePostStat=1'.	

For example, to use the prior models and data in `DJURSLAND_P01_N0010000_NB-13_NR03_PRIOR.h5` and the data in `tTEM-Djursland.h5`, run

..  code-block:: python

    python integrate_rejection.py DJURSLAND_P01_N0010000_NB-13_NR03_PRIOR.h5 tTEM-Djursland.h5


.. #.. literalinclude:: ../../src/python/integrate_rejection.py
.. #    :language: python
.. #    :lines: 1-50

.. integrate module
.. ---------------- 



integrate python functions
--------------------------

.. .. automodule:: integrate
..    :members:

.. .. autosummary::
..    :toctree: generated

..    integrate_io
..    integrate_rejection
..    integrate_update_prior_attributes
..    integrate_posterior_stats
..    lu_post_sample_logl
..    read_gex

   