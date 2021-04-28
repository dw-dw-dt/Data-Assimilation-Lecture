########################################
# Decomp_EnKF.R                        #
#                                      #
# Time-series decomposition using EnKF #
########################################

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# observation data
yt = np.array(pd.read_csv( "../Input_Data.txt" , sep="\s+", header=None, usecols=[2])).reshape(-1)

# parameters
Nt = len(yt)  # number of time points
#Np <- 1000        # number of ensemble members
Np = 50        # number of ensemble members
torder = 2       # trend order
period = 12      # period of seasonal component

# system noises
tau2u  = 25  # variance of system noise of trend component
tau2s  = 25  # variance of system noise of seasonal component
sigma2 = 25  # variance of observation noise

# preparation for sequential estimation
dimx = torder + period - 1               # dimension of state vector
xtp  = np.full( (dimx, Np, Nt), np.nan )  # predictive distribution
xtf  = np.full( (dimx, Np, Nt), np.nan )  # filter distribution

# observation matrix (vector)
Ht = np.full( (1,dimx), np.nan )
Ht[0] = np.array([1,0,1] + [0 for i in range(dimx-3)])


# variables for DA
x0   = np.zeros( (dimx, Np) )  # initial state vector
xtpm = np.zeros( (dimx, Nt) )  # mean of predictive distribution
xtpv = np.zeros( (dimx, Nt) )  # variance of predictive distribution
xtfm = np.zeros( (dimx, Nt) )  # mean of filter distribution
xtfv = np.zeros( (dimx, Nt) )  # variance of filter distribution


# initial state vector
fit = LinearRegression()    # regression line
fit.fit( np.array([i for i in range(period+1)]).reshape(-1,1) , yt[:period+1] )
x0[0] = fit.intercept_ + np.random.normal(loc=0, scale=np.sqrt(tau2u), size=Np)
x0[1] = fit.intercept_ - fit.coef_[0]
residuals = yt[:period+1] - fit.predict(np.array([i for i in range(period+1)]).reshape(-1,1))
x0[2] = residuals[period-1] + np.random.normal(loc=0, scale=np.sqrt(tau2s), size=Np)
x0[3:dimx] = residuals[period-2:0:-1].reshape(period-2,-1)


# EnKF

# EnKF
for t in range(Nt):

    print("Assimilating t={}".format(t))

    # one-step ahead prediction
    if t>0:
        xtp[0,:,t] = 2 * xtf[0,:,t-1] - xtf[1,:,t-1] \
           +np.random.normal(loc=0, scale=np.sqrt(tau2u), size=Np )
        xtp[1,:,t] = xtf[0,:,t-1]
        xtp[2,:,t] = - np.sum( xtf[2:dimx,:,t-1], axis=0 ) \
           + np.random.normal( loc=0, scale=np.sqrt( tau2s ), size=Np )
        xtp[3:dimx,:,t] = xtf[2:(dimx-1),:,t-1]
    else:
        xtp[0,:,t] = 2 * x0[0] - x0[1] + np.random.normal( loc=0, scale=np.sqrt( tau2u ), size=Np )
        xtp[1,:,t] = x0[0]
        xtp[2,:,t] = - np.sum( x0[2:dimx], axis=0 ) \
           + np.random.normal( loc=0, scale=np.sqrt( tau2s ), size=Np )
        for i in range(3,dimx): xtp[i,:,t] = x0[i-1]

    xtpm[:,t] = np.mean( xtp[:,:,t], axis=1 )

    # filtering
    dxtp = xtp[:,:,t] - xtpm[:,t].reshape(-1,1)
    wt  = np.random.normal( loc=0, scale=np.sqrt(sigma2), size=Np )
    dw  = wt - np.mean( wt )
    bb  = 1 / ( Ht @ ( dxtp @ dxtp.T ) @ Ht.T + np.mean(wt*wt) )
    xtf[:,:,t] = xtp[:,:,t] + ( dxtp @ dxtp.T ) @ Ht.T @ bb @ ( yt[t] + dw - Ht @ xtp[:,:,t] )

    xtfm[:,t] = np.mean( np.mean( xtf, axis=1 ), axis=1)



# figures
ut = np.mean( xtf[0,:,:], axis=0 )
st = np.mean( xtf[2,:,:], axis=0 )
fig = plt.figure(figsize=(10,40))
ax1 = fig.add_subplot(311, xlabel='Time', ylabel=r'$y_t$ & $u_t$')
ax2 = fig.add_subplot(312 ,xlabel='Time', ylabel=r'$s_t$')
ax3 = fig.add_subplot(313, xlabel='Time', ylabel=r'$y_t - u_t - s_t$')
ax1.plot(yt, color='black', label=r'$y_t$')
ax1.plot(ut, color='red', label=r'$u_t$')
ax2.plot(st, color='black', label=r'$s_t$')
ax3.plot(yt - ut - st, color='black', label=r'$y_t - u_t - s_t$')
ax1.legend()
fig.subplots_adjust(hspace=0.5)
plt.show()
