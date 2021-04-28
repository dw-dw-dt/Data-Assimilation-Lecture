#
# EnKFadvection.R
#
# Data Assimilation on an advection equation using EnKF

# install modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# parameters
u0      = -4.0     # advection coefficient (true/unknown)
sigma2  =  0.1     # variance of observation noise (true/known)
um      =  0.0     # mean of prior related to advection coefficient
usd     =  2.0     # std of prior related to advection coefficient


# grid spacing
dt = 0.5            # time interval for simulation
Nt = 500            # number of time steps for simulation
t  = np.array([i*dt for i in range(Nt)]) # temporal coordinate
dx = 10.0           # interval for spatial difference
Nx = 400            # number of spatial grids
x  = np.array([i*dx for i in range(Nx)])  # spatial coordinate

x1  = np.array([i for i in range(Nx)])          # set of spatial indices
xl1 = np.array([i for i in range(1,Nx)] + [0])      # leftward shifted of x1
xr1 = np.array([Nx-1] + [i for i in range(Nx-1)]) # rightward shifted of x1

Np   = 1000          # number of ensemble members in EnKF

xini = [i for i in range(49,150)] # set of indices for initial condition

umax =  0.8*dx/dt # upper limit of u determined by CFL condition
umin = -0.8*dx/dt # lower limit of u determined by CFL condition

def exact(u): #analytical solution
    C = np.zeros( (Nx, Nt) )
    C[xini,0] = 1
    tmp = np.concatenate( [C[:,0],C[:,0]] )
    print(tmp.shape)
    for k in range(1,Nt):
        rshift = np.floor( (k*u*dt/dx)%(Nx) )
        xshift = (x1 + Nx - rshift).astype(int)
        C[x1,k] = tmp[xshift]
    return C



def simulation(u):
    C = np.zeros( (Nx, Nt) )
    C[xini,0] = 1
    for k in range(1,Nt):
        A = - u * ( C[xl1,k-1] - C[xr1,k-1] ) / ( 2 * dx )    # advection
        S = abs( u ) * ( C[xl1,k-1] + C[xr1,k-1] - 2 * C[:,k-1] ) / ( 2 * dx )
        C[:,k] = C[:,k-1] + ( A + S ) * dt
    return C

# synthetic observation data
yte = exact( u0 )
yt0 = exact( u0 )  # true
yts = simulation( um )  # simulation using initial guess parameters



yt  = np.full( (Nx,Nt), np.nan)  # observed concentration
observedX = np.arange(0,Nx,20)             # observed places
observedT = np.arange(9,Nt,10)            # observed time

#observedX = np.arange(Nx)             # observed places
#observedT = np.arange(Nt)              # observed time

obs_index = np.ix_(observedX, observedT) # observed places * time
yt[obs_index] = yt0[obs_index]
for k in observedT:
    yt[ observedX , k ] += np.random.normal( size = len(observedX), loc=0, scale=np.sqrt(sigma2) )


xtfini  = np.full( (Nx,Nt), np.nan)
xtfini[obs_index] = yts[obs_index]


# observation matrix
Ht = np.zeros( (len(observedX),Nx+1))
Ht[:,observedX] = np.eye(len(observedX))



# variables for DA
xtp  = np.zeros( (Nx+1, Np) )  # predictive distribution
xtf  = np.zeros( (Nx+1, Np) )  # filter distribution
xtpm = np.zeros( (Nx+1, Nt) )  # mean of predictive distribution
xtfm = np.zeros( (Nx+1, Nt) )  # mean of filter distribution
xtfv = np.zeros( (Nx+1, Nt) )  # variance of filter distribution


llh  = -Nx*Nt*np.log(2*np.pi*sigma2)
llhm = np.full( Nt, np.nan )

uenm = np.full( (2, Nt), np.nan )    # Ensemble average & standard deviation of u


# sample particles from priors
xtf[xini,:] = 1
xtf[Nx,:] = np.random.normal( size=Np, loc=um, scale=usd )


xtf[Nx,xtf[Nx,:]>umax] = umax
xtf[Nx, xtf[Nx,:]<umin ] = umin
uPrior = xtf[Nx,:]


llhbak = np.full( Nt, np.nan )
zz = 0.0
for k in range(Nt,0,-1):
    if k in observedT:
        llhbak[k] = zz
        res = yt[ observedX, k ] - xtfini[ observedX, k ]
        zz = zz - res @ res /(2*sigma2)

uenm[0,0] <- np.mean(xtf[Nx,:])
uenm[1,0] <- np.std(xtf[Nx,:])




# EnKF
for k in range(1,Nt):

    print("Assimilating t={}".format(k))

    # one-step ahead prediction
    A = - ( xtf[xl1,:] - xtf[xr1,:] ) @ np.diag( xtf[Nx,:] ) / ( 2 * dx )
    S = ( xtf[xl1,:] + xtf[xr1,:] - 2 * xtf[x1,:] ) @ np.diag( np.abs(xtf[Nx,:]) ) / ( 2 * dx )
    v =  np.random.normal(loc = 0, scale = 0.01, size = (Nx, Np))
    xtp[x1,:] = xtf[x1,:] + ( A + S ) * dt + v[x1,:]
    xtp[Nx,:] = xtf[Nx,:] # + np.random.normal(size =  Np, loc=0, scale=0.1 ) # no system noise
    xtpm[:,k] = np.mean( xtp, axis=1)

    # skip filtering step if no observation
    if (observedT==k).sum() == 0:
        xtf = xtp
        xtfm[:,k] = np.mean(xtf, axis=1)
        continue


    # filtering
    dxtp = xtp - xtpm[:,k].reshape(-1,1)
    wt = np.random.normal(loc=0, scale=np.sqrt(sigma2), size=(len(observedX), Np))
    dw  = wt - np.mean( wt, axis=1).reshape(-1,1)
    bb  = np.linalg.pinv( Ht @ dxtp @ dxtp.T @ Ht.T + wt@ wt.T)
    xtf = xtp + xtp @ dxtp.T @ Ht.T @ bb @ ( (yt[observedX,k].reshape(-1,1) + dw) - Ht @ xtp )


 #
    xtf[Nx,xtf[Nx,:]>umax] = umax
    xtf[Nx, xtf[Nx,:]<umin ] = umin



    uenm[0,k] = np.mean(xtf[Nx,:])
    uenm[1,k] = np.std(xtf[Nx,:])
    print( "Estimated u = {}".format(uenm[0,k]))
    print( "Standard deviation of u = {}".format(uenm[1,k]))


    xtfm[:,k] = np.mean( xtf, axis=1 )


# log-likelihood at this time point
    if ( observedT==k ).sum() != 0:
        res = yt[observedX, k] - Ht @ xtfm[:,k]
        llh = llh - res @ res /(2*sigma2)
        llhm[k] = llh + llhbak[k]
        print( "Log-likelihood = {}".format(llhm[k]))
    print("\n")


# figures
plt.rcParams["font.size"] = 18
# prior & posterior distributions of the model parameters
fig = plt.figure(figsize=(20,20))
ax1 = fig.add_subplot(211, title=r'Prior PDF $p(u)$', xlabel=r'$u$', ylabel='Frequency')
ax2 = fig.add_subplot(212, title=r"Posterior PDF $p(\mu|y)$",xlabel=r'$u$', ylabel='Frequency')
ax1.hist(uPrior, bins='scott')
ax2.hist(xtf[Nx],bins='scott')
fig.tight_layout()
fig.savefig("dist_EnKF.png")

# true, simulated, observed & assimilated concentrations
fig = plt.figure(figsize=(40,20))
ax1 = fig.add_subplot(221, title='True', ylabel='t')
ax2 = fig.add_subplot(222, title="Simulation", ylabel='t')
ax3 = fig.add_subplot(223, title='Observed',  ylabel='t')
ax4 = fig.add_subplot(224, title="Assimilation", ylabel='t')
sns.heatmap(data=pd.DataFrame(data=yte.T, index=dt*t, columns=dx*x), ax=ax1, xticklabels=100,yticklabels=100)
sns.heatmap(data=pd.DataFrame(data=yts.T, index=dt*t, columns=dx*x), ax=ax2, xticklabels=100,yticklabels=100)
sns.heatmap(data=pd.DataFrame(data=yt.T, index=dt*t, columns=dx*x), ax=ax3, xticklabels=100,yticklabels=100)
sns.heatmap(data=pd.DataFrame(data=xtfm[x1].T, index=dt*t, columns=dx*x), ax=ax4, xticklabels=100,yticklabels=100)
fig.tight_layout()
ax1.invert_yaxis()
ax2.invert_yaxis()
ax3.invert_yaxis()
ax4.invert_yaxis()
fig.savefig("concentration_EnKF.png")


# time series of estimated value
utr = np.full( Nt, u0 )
observedt = observedT*dt


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, xlabel="Time", ylabel='Estimated Value')
#ax.plot(uenm[0,observedT],observedt,color='green')
ax.plot(t, utr, color='red')
ax.errorbar(x=observedt,y=uenm[0,observedT],yerr=2*uenm[1,observedT],color='green')
fig.savefig("timeseries_estimated_EnKF.png")

llhm[0] = llhm[1]
# time series of log-likelihood
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, xlabel='Time', ylabel = 'Log-likelihood')
ax.plot(observedt, llhm[observedT],color='blue')
fig.savefig("timeseries_loglikelihood_EnKF.png")
