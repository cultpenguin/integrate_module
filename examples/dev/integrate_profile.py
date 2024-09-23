# integrate_profile
import integrate as ig

import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

f_data_h5 = 'DAUGAARD_AVG.h5'
f_prior_data_h5 = 'prior_detailed_outvalleys_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5'
f_post_h5 = ig.integrate_rejection(f_prior_data_h5, f_data_h5, N_use = 10000, parallel=1, updatePostStat=0, showInfo=1, Nproc=8)


profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumulative')
stats.print_stats()