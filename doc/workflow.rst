=========
Workflows
=========

DIfferent examples of workflows using INTEGRATE.


INTEGRATE workflows - a simple (resistivity only) prior, tTEM data
====================================================================



## Create realizations from the prior and store in PRIOR.h5

The Duagaard type prior results in 3 types of model parameters stored as 

    PRIOR.h5:/M1
    PRIOR.h5:/M2
    PRIOR.h5:/M2

## Compute prior data and store in PRIOR.h5

### Compute prior EM data 

Setup af 'forward' type hdf5 that defines the type of forward

    FORWARD.h5:/type:'tdem'
    FORWARD.h5:/method:'ga-aem'
    FORWARD.h5:/gex:'filename.gex'
    FORWARD.h5:/im:1 --> points to the resistivity parameter in PRIOR.h5, gere '/M1'
    
    >> intergrate_update_prior_data(prior_h5, forward_h5, im, id)

    PRIOR.h5:/M1
    PRIOR.h5:/M2
    PRIOR.h5:/M2
    PRIOR.h5:/D1 --> EM data

## Setup data

    DATA.h5:/UTMX
    DATA.h5:/UTMY
    DATA.h5:/LINE
    DATA.h5:/ELEVATION
    DATA.h5:/D1/d_obs
    DATA.h5:/D1/d_std

## Perform inversion 

    >> integrate_rejection()
    >> integrate_posterior_statistics()


## Make some plots    







