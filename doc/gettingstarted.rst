===============
Getting started
===============


:: 

    import integrate as ig
    import numpy as np
    import matplotlib.pyplot as plt


0. Define the data
==================

::

    f_data_h5 = 'DATA.h5'


1. Setup a prior model
======================

1a. sample a prior distribution 

::

    RHO_min = 1
    RHO_max = 1500
    z_max = 50 
    N=1000000
    f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='uniform', z_max = z_max, NLAY_min=1,NLAY_max=8, rho_dist='log-uniform', RHO_min=RHO_min, RHO_max=RHO_max)
    
    # Plot simple statistic from the prior
    ig.plot_prior_stats(f_prior_h5)


1b. Compute prior data (the forward response of the prior realizations)

:: 

    file_gex ='ttem.gex'
    f_prior_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex)

2. Sample the posterior distribution
====================================

::

    f_post_h5 = ig.integrate_rejection(f_prior_h5, f_data_h5)


3. Plot statistic from the posterior distribution
=================================================

::

    # Plot the Temperature used for inversion
    ig.plot_T_EV(f_post_h5, pl='T')
    ig.plot_T_EV(f_post_h5, pl='EV')
    ig.plot_T_EV(f_post_h5, pl='ND')


::

    # plot prior and posterior data
    ig.plot_data_prior_post(f_post_h5, i_plot = 0)
    ig.plot_data_prior_post(f_post_h5, i_plot = 1199)    


::

    # Plot profiles
    ig.plot_profile(f_post_h5, i1=400, i2=800)


:: 

    # Plot posterior probability of cumulative thickness of discrete parameter
    ig.plot_posterior_cumulative_thickness(f_post_h5,im=2, icat=[2]], property='median')



