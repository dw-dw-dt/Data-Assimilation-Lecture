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
    3. Data Assimilation on Advection-Diffusion Equation
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

# parameters
u0 = 8.0
v0 = 5.0
tau20 = 1.0
sigma2 = 16.0
um = 5.0
usd = 3.0
vm = 6.0
vsd = 2.0
tau2m = 0.8
tau2sd = 0.5

# grid spacing
dt = 0.5
Nt = 1200
t = np.arange(Nt) * dt
dx = 10.0
Nx = 400
x = np.arange(Nx) * dx
xl = np.array([(i+1)%Nx for i in range(Nx)])
xr = np.array([(i-1)%Nx for i in range(Nx)])
Np = 1000


# +
# function for simulation
def simulation(u, v, tau2):
    
    C = np.zeros([Nt,Nx])
    
    for k in range(1,Nt):
        A = -u * (C[k-1,xl] - C[k-1,xr]) / (2 * dx)
        K = max(k-2, 0)
        S = v * (C[K,xl] - 2 * C[K,:] + C[K,xr]) / (dx * dx)
        W = np.sin(2 * np.pi * t[k-1] / 120) * np.sin(np.pi * x / 60) + np.random.normal(0, np.sqrt(tau2), Nx)
        C[k] = 2 * dt * (A + S + W)
        if k>1:
            C[k] = C[k-2] + C[k]
    
    return C

# synthetic observation data
yt0 = simulation(u0, v0, tau20)
yts = simulation(um, vm, tau2m)
yt = np.zeros([Nt,Nx]) + np.nan
observedT = np.arange(9, Nt, 15)
observedX = np.arange(2, Nx, 20)
for t_index in observedT:
    for x_index in observedX:
        yt[t_index,x_index] = yt0[t_index,x_index]

yt = yt + np.random.normal(0, np.sqrt(sigma2), len(yt)).reshape(-1, 1)

# variables for DA
xtp = np.zeros([Np,2,Nx])
xtf = np.zeros([Np,2,Nx])
xtfm = np.zeros([Nt,Nx])
llh = np.zeros(Np)
res = np.zeros(Np, dtype=int)

# sample particles from priors
u = np.random.normal(um, usd, Np)
v = np.random.normal(vm, vsd, Np)
tau2 = np.random.lognormal(np.log(tau2m), tau2sd, Np)
uPrior = u.copy()
vPrior = v.copy()
tau2Prior = tau2.copy()
# -

# particle filter
for k in tqdm(range(1,Nt)):
    
    # one-step ahead prediction
    tau2_expand = np.array([])
    for _ in range(Nx):
        tau2_expand = np.append(tau2_expand, tau2)
    A = - np.diag(u) @ (xtf[:,0,xl] - xtf[:,0,xr]) / (2 * dx)
    S = np.diag(v) @ (xtf[:,1,xl] - 2 * xtf[:,1,:] + xtf[:,1,xr]) / (dx * dx)
    W = np.sin(2 * np.pi * t[k] / 120) * np.sin(np.pi * x / 60) + np.random.normal(0, np.sqrt(tau2_expand), Np*Nx).reshape(Np, Nx)
    xtp[:,0,:] = xtf[:,1,:] + 2 * dt * (A + S + W)
    xtp[:,1,:] = xtf[:,0,:]
    
    # skip filtering step if no observation
    if all(observedT!=k):
        xtf = xtp.copy()
        xtfm[k] = np.mean(xtf[:,0,:], axis=0)
        continue
    
    # log-likelihood at this time point
    dy = yt[k,observedX] - xtp[:,0,observedX]
    llh = -0.5 * (np.log(2 * np.pi * sigma2) + np.sum(dy*dy, axis=1) / sigma2)
    
    # filtering using the residual systematic resampling
    weight = np.exp(llh - np.max(llh)) / np.sum(np.exp(llh - np.max(llh)))
    r = np.random.rand() / Np
    uTemp = u.copy()
    vTemp = v.copy()
    tau2Temp = tau2.copy()
    
    for p in range(Np):
        res[p] = np.floor((weight[p] - r) * Np) + 1
        r += res[p] / Np - weight[p]

    j = 0
    for i in range(Np):
        for _ in range(res[i]):
            xtf[j] = xtp[i]
            u[j] = uTemp[i]
            v[j] = vTemp[i]
            tau2[j] = tau2Temp[i]
            j += 1
    
    xtfm[k] = np.mean(xtf[:,1,:], axis=0)

# prior & posterior distributions of the model parameters
plt.figure(figsize=(10,5))
plt.suptitle('Prior & Posterior Distributions of the Model Parameters')
plt.subplot(2,3,1)
plt.hist(uPrior, bins=20)
plt.xlabel('$u$')
plt.ylabel('$p(u)$')
plt.subplot(2,3,2)
plt.hist(vPrior, bins=20)
plt.xlabel('$v$')
plt.ylabel('$p(v)$')
plt.subplot(2,3,3)
plt.hist(tau2Prior, bins=20)
plt.xlabel('$\\tau^2$')
plt.ylabel('$p(\\tau^2)$')
plt.subplot(2,3,4)
plt.hist(u, bins=20, color='tab:orange')
plt.xlabel('$u$')
plt.ylabel('$p(u|y)$')
plt.subplot(2,3,5)
plt.hist(v, bins=20, color='tab:orange')
plt.xlabel('$v$')
plt.ylabel('$p(v|y)$')
plt.subplot(2,3,6)
plt.hist(tau2, bins=20, color='tab:orange')
plt.xlabel('$\\tau^2$')
plt.ylabel('$p(\\tau^2|y)$')
plt.tight_layout()
plt.show()

# true, simulated, observed & assimilated concentrations
plt.figure(figsize=(8,8))
plt.subplot(2,2,1)
plt.gca().set_title('True')
plt.imshow(yt0, aspect='auto', origin='lower', cmap='autumn')
plt.xticks(np.arange(0, Nx+1, Nx//4), np.arange(0, Nx*dx+1, Nx*dx//4, dtype=int))
plt.xlabel('$x$')
plt.yticks(np.arange(0, Nt+1, Nt//6), np.arange(0, Nt*dt+1, Nt*dt//6, dtype=int))
plt.ylabel('$t$')
plt.subplot(2,2,2)
plt.gca().set_title('Simulation')
plt.imshow(yts, aspect='auto', origin='lower', cmap='autumn')
plt.xticks(np.arange(0, Nx+1, Nx//4), np.arange(0, Nx*dx+1, Nx*dx//4, dtype=int))
plt.xlabel('$x$')
plt.yticks(np.arange(0, Nt+1, Nt//6), np.arange(0, Nt*dt+1, Nt*dt//6, dtype=int))
plt.ylabel('$t$')
plt.subplot(2,2,3)
plt.gca().set_title('Observation')
plt.imshow(yt, aspect='auto', origin='lower', cmap='autumn')
plt.xticks(np.arange(0, Nx+1, Nx//4), np.arange(0, Nx*dx+1, Nx*dx//4, dtype=int))
plt.xlabel('$x$')
plt.yticks(np.arange(0, Nt+1, Nt//6), np.arange(0, Nt*dt+1, Nt*dt//6, dtype=int))
plt.ylabel('$t$')
plt.subplot(2,2,4)
plt.gca().set_title('Assimilation')
plt.imshow(xtfm, aspect='auto', origin='lower', cmap='autumn')
plt.xticks(np.arange(0, Nx+1, Nx//4), np.arange(0, Nx*dx+1, Nx*dx//4, dtype=int))
plt.xlabel('$x$')
plt.yticks(np.arange(0, Nt+1, Nt//6), np.arange(0, Nt*dt+1, Nt*dt//6, dtype=int))
plt.ylabel('$t$')
plt.tight_layout()
plt.show()
