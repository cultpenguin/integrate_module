import cProfile
import pstats
import io
from pstats import SortKey
import integrate as ig

def profile_integrate():
    pr = cProfile.Profile()
    pr.enable()
    
    f_data_h5 = 'DAUGAARD_AVG.h5'
    f_prior_data_h5 = 'prior_detailed_outvalleys_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5'
    f_post_h5 = ig.integrate_rejection(f_prior_data_h5, f_data_h5, N_use=1000, parallel=0, updatePostStat=0, showInfo=1, Nproc=8)
    
    pr.disable()
    
    # Save stats to a file
    pr.dump_stats('integrate_profile.prof')
    
    # Print text output
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(20)  # Print top 20 time-consuming functions
    print(s.getvalue())

if __name__ == "__main__":
    profile_integrate()
