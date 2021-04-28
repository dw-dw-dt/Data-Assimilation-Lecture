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
    2. Time-series Decomposition
'''

# import modules
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

# fix seed
RANDOM_SEED = 0
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# observation data
yt = []
with open('Input_Data.txt', mode='r') as f:
    for line in f:
        l = line.replace('\n','').split('\t')
        yt.append(list(map(float, l))[-1])
yt = np.array(yt)

# parameters
Nt = len(yt)
Np = 100000
torder = 2
period = 12

# sample particles
tau2u = np.random.rand(Np)
tau2s = np.random.rand(Np)
sigma2 = np.random.rand(Np)*10

# +
# preparation for sequential estimation
dimx = torder + period - 1
x0 = np.zeros(dimx)
xtp = np.zeros([Np,dimx])
xtf = np.zeros([Np,dimx])
res = np.zeros(Np, dtype=int)
ut = np.array([])
st = np.array([])
llh = np.zeros(Np)
llh_model = np.array([])

# initial state vector
fit = LinearRegression().fit(
    np.array(range(period+1)).reshape(-1,1),
    yt[:period+1].reshape(-1,1)
)
xtf[:,0] = fit.intercept_[0]
xtf[:,1] = fit.intercept_[0] - fit.coef_[0,0]
for i in range(2,dimx):
    xtf[:,i] = yt[period+1-i] - fit.predict([[period+1-i]])[0,0]
# -

# sequential estimation using particle filter
for t in tqdm(range(Nt)):
    
    # one-step ahead prediction
    xtp[:,0] = 2 * xtf[:,0] - xtf[:,1] + np.random.normal(0, np.sqrt(tau2u), Np)
    xtp[:,1] = xtf[:,0]
    xtp[:,2] = - np.sum(xtf[:,2:dimx],axis=1) + np.random.normal(0, np.sqrt(tau2s), Np)
    xtp[:,3:dimx] = xtf[:,2:dimx-1]
        
    # log-likelihood at this time point
    llh = -0.5 * (np.log(2*np.pi*sigma2) + (yt[t] - xtp[:,0] - xtp[:,2])**2 / sigma2)
    
    # weight of each particle
    weight = np.exp(llh - np.max(llh)) / np.sum(np.exp(llh - np.max(llh)))
    
    # filtering based on the systematic residual resampling
    u = np.random.rand() / Np
    
    for p in range(Np):
        res[p] = np.floor((weight[p] - u) * Np) + 1
        u += res[p] / Np - weight[p]
    
    k = 0
    for i in range(Np):
        for j in range(res[i]):
            xtf[k] = xtp[i]
            k += 1
    
    # trend component & seasonal component
    ut = np.append(ut, np.mean(xtf[:,0]))
    st = np.append(st, np.mean(xtf[:,2]))
    
    # log-likelihood of each model
    llh_model = np.append(llh_model, np.sum(llh))


# +
# figures
time = np.arange(Nt)

plt.figure(figsize=(7,3))
plt.plot(time, yt, label='true')
plt.plot(time, ut+st, label='prediction')
# plt.plot(time, ut, label='trend component', ls='--')
# plt.plot(time, st, label='seasonal component', c='tab:green')
plt.legend(loc='best')
plt.xlabel('time')
plt.ylabel('sea level')
plt.title('Particle Filter for Time-series Decomposition')
plt.show()
