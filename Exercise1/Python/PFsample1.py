# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
'''
    PFsample1.py
    Sample Program for Particle Filter
    1. 2-D Random Walk
'''

# import modules
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# fix seed
RANDOM_SEED = 0
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# number of time points & particles
Nt = 100
Np = 10000

# variances of system noise (unknown)
tau_x = 0.5
tau_y = 0.2
v_cov = np.array([[tau_x,0],[0,tau_y]])

# variances of observation noise (known)
sigma_x = 0.1
sigma_y = 0.1
w_cov = np.array([[sigma_x,0],[0,sigma_y]])

# synthetic observation data
mean0 = np.array([0,0])
vt = np.random.multivariate_normal(mean0, v_cov, Nt)
wt = np.random.multivariate_normal(mean0, w_cov, Nt)
yt = np.cumsum(vt+wt, axis=0)

# +
# preparation for sequential estimation
x0 = np.zeros([Np,2])
xtp = np.zeros([Nt,Np,2])
xtf = np.zeros([Nt,Np,2])
llh = np.zeros([Nt,Np])
res = np.zeros(Np, dtype=int)

# sequential estimation using particle filter
for t in tqdm(range(Nt)):
    
    x = xtf[t-1] if t>0 else x0
    
    # one-step ahead prediction
    xtp[t] = x + np.random.multivariate_normal(mean0, v_cov, Np)
    
    # log-likelihood at this time point
    llh[t] = -np.log(2*np.pi) - 0.5*(
        np.log(sigma_x) + np.log(sigma_y)
        + (yt[t,0] - xtp[t,:,0])**2 / sigma_x
        + (yt[t,1] - xtp[t,:,1])**2 / sigma_y
    )
    
    # weight of each particle
    weight = np.exp(llh[t] - np.max(llh[t])) / np.sum(np.exp(llh[t] - np.max(llh[t])))
    
    # filtering based on the systematic residual resampling
    u = np.random.rand() / Np
    
    for p in range(Np):
        res[p] = np.floor((weight[p] - u) * Np) + 1
        u += res[p] / Np - weight[p]
    
    k = 0
    for i in range(Np):
        for j in range(res[i]):
            xtf[t,k] = xtp[t,i]
            k += 1

# log-likelihood of each model
llh_model = np.sum(llh, axis=1)
# -

# figures
plt.plot(yt[:,0], yt[:,1], label='true')
plt.plot(np.mean(xtf[:,:,0],axis=1), np.mean(xtf[:,:,1],axis=1), label='prediction')
plt.legend(loc='best')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Particle Filter for 2-D Random Walk')
plt.show()
