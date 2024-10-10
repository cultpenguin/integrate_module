#!/usr/bin/env python3

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Integration tool")
    parser.add_argument("f_data_h5", type=str, help="Path to data H5 file")
    parser.add_argument("f_prior_h5", type=str, help="Path to prior H5 file")
    parser.add_argument("--method", type=str, help="Method")
    parser.add_argument("--N_use", type=int, help="Number of samples to use")
    parser.add_argument("--f_post_h5", type=str, help="Path to output posterior H5 file")
    parser.add_argument("--showInfo", type=int, help="Show information during inversion")
    parser.add_argument("--parallel", type=int, help="Run in parallel")

    args = parser.parse_args()

    # Print out the arguments to verify they're being parsed correctly
    #print(f"Data H5 file: {args.f_data_h5}")
    #print(f"Prior H5 file: {args.f_prior_h5}")

    f_prior_h5 = args.f_prior_h5
    f_data_h5 = args.f_data_h5
    method = args.method


    if args.showInfo:
        #print(f"ShowInfo: {args.showInfo}")
        showInfo = args.showInfo
    else:
        showInfo = 0

    if args.method:
        if showInfo>0:
            print(f"Method: {args.method}")
        method = args.method
    else:
        method = 'inversion'
        
    if args.N_use:
        if showInfo>0:
            print(f"N_use: {args.N_use}")
        N_use = args.N_use
    else:
        N_use = 1e+9

    if args.f_post_h5:
        if showInfo>0:
            print(f"Posterior H5 file: {args.f_post_h5}")
        f_post_h5 = args.f_post_h5
    else:
        f_post_h5 = 'posterior.h5'

    if args.parallel:
        if showInfo>0:
            print(f"Parallel: {args.parallel}")
        parallel = 1==args.parallel
    else:
        parallel = False




    # Here you would add the actual integration logic
    # For example:
    # perform_integration(args.type, args.f_data_h5, args.f_prior_h5, args.N_use, args.f_post_h5)

    if method.lower()=='inversion':
        #print("Starting inversion")
        import integrate as ig
        import time as time
        t0 = time.time()
        ig.integrate_rejection(args.f_prior_h5, args.f_data_h5, 
                            f_post_h5=f_post_h5,
                            parallel=parallel, 
                            Ncpu=8,
                            N_use = N_use,
                            showInfo=showInfo
                            )
        t1 = time.time()
        print("%s:Time: %3.1f" % (f_post_h5,t1-t0))

    #print("Integration complete")

if __name__ == "__main__":
    main()