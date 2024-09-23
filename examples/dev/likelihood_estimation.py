#%% 
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

N_obs = 11693
N = 100000
Nd = 40

# Simulated data
D = np.random.rand(N, Nd)
# Observed data
D_obs = np.random.rand(N_obs, Nd)

# Diagonal data covariance matrix Nd,Nd with random values
C = np.diag(np.random.rand(Nd))
# Inverse of C matrix
C_inv = np.linalg.inv(C)


def likelihood(d, d_obs, C_inv):
    dd = d-d_obs
    return np.dot(dd, np.dot(C_inv, dd))

def likelihood_vectorized(D, d_obs, C_inv):
    dd = D - d_obs
    return np.sum(np.dot(dd, C_inv) * dd, axis=1)

def likelihood_data(D, d_obs,C_inv):
    N = D.shape[0]
    L = np.zeros(N)

    L = likelihood_vectorized(D, d_obs, C_inv)

    #for id in range(N):
    #    L[id]=likelihood(D[id], d_obs, C_inv)
    
    return L

#%% Likelihood estimation
t0 = time.time()
for iobs in tqdm(range(N_obs)):
    d_obs = D_obs[iobs]
    L = likelihood_data(D, d_obs, C_inv)
t2 = time.time()
t_seq = t2 - t0

print("Time for sequential loop:", t_seq)

#%% Parallel

def process_likelihoods(iobs_range, D, D_obs, C_inv):
    L_subset = []
    for iobs in iobs_range:
        d_obs = D_obs[iobs]
        L = likelihood_data(D, d_obs, C_inv)
        L_subset.append(L)
    return L_subset


print("__name__:", __name__)
if __name__ == '1__main__':
    t0 = time.time()
    num_processes = 8#mp.cpu_count()  # Adjust based on your system
    chunk_size = N_obs // num_processes

    with mp.Pool(processes=num_processes) as pool:
        iobs_ranges = []
        for i in range(num_processes):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, N_obs)  # Ensure end does not exceed N_obs
            iobs_ranges.append((start, end))

        results = pool.starmap(process_likelihoods, [(iobs_range, D, D_obs, C_inv) for iobs_range in iobs_ranges])


    
    # Combine results from different processes
    L_all = np.concatenate(results)
    t2 = time.time()
    t_par = t2 - t0

    print("Time for par loop:", t_par)


# %%
#plt.plot(L)
# %%
