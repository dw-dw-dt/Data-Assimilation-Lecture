# python 3.5 or greater

import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.optimize import minimize, shgo
from scipy.optimize import LinearConstraint as LS
import pandas as pd

def KalmanFilter(a0,P0,dt,ct,Tt,Zt,HHt,GGt,yt):
    ## Check classes of arguments

    stor_type = {"a0" : a0, "P0" : P0, "dt" : dt,
                   "ct" : ct, "Tt" : Tt, "Zt" : Zt,
                   "HHt" : HHt, "GGt" : GGt, "yt" : yt}
    not_ndarray = [key for key,value in stor_type.items()
    if not isinstance(value,np.ndarray)]
    if(not_ndarray):
        sys.exit("'" + ', '.join(not_ndarray) +
        "' must be class 'numpy.ndarray'")

    ## Check the storage type: Must be 'float64' ('float') for all arguments

    not_float = [key for key,value in stor_type.items() if not value.dtype=='float']
    if(not_float):
        sys.exit("Storage mode of variable(s) '" + ', '.join(not_float)
        + "' is not double!")

    ## Check compatibility of dimensions
    #### ! we do not cosider here time-varying system
    n = yt.shape[1]
    d = yt.shape[0]
    m = len(a0)
    if(P0.shape[1] != m or P0.shape[0] != m or dt.shape[0] != m or
       Zt.shape[1] != m or HHt.shape[0] != m or HHt.shape[1] != m  or
       Tt.shape[0] != m  or Tt.shape[1] != m):
       sys.exit("Some of P0.shape[1], P0.shape[0], dt.shape[0],\n"+
             "Zt.shape[1], HHt.shape[0], HHt.shape[1],\n"+
             "Tt.shape[0] or Tt.shape[1] is/are not equal to 'm'!\n")

    if(ct.shape[0] != d or Zt.shape[0] != d  or
       GGt.shape[0]!= d  or GGt.shape[1] != d  or yt.shape[0] != d):
        sys.exit("Some of ct.shape[0], Zt.shape[0], GGt.shape[0],\n"+
             "GGt.shape[1] or yt.shape[0] is/are not equal to 'd'!\n")

    ## simulation
    ### initialization
    at = np.zeros((m,n+1)); at[:,0] = a0 # predicted state variables
    Pt = np.zeros((m,m,n+1)); Pt[:,:,0] = P0 # variances of at
    vt = np.zeros((d,n)) # prediction errors
    Ft = np.zeros((d,d,n)) # variances of vt
    Kt = np.zeros((m,d,n)) # kalman gain
    att = np.zeros((m,n)) # filtered state variables
    Ptt = np.zeros((m,m,n)) # variance of at
    logLik = -1/2 * yt.shape[1] * d * np.log(2*np.pi) # constant part of logLik
    for i in range(n):
        y_pred = Zt @ at[:,i] + ct
        #updating...
        vt[:,i] = yt[:,i] - y_pred #prediction error
        Ft[:,:,i] = Zt @ Pt[:,:,i] @ Zt.T + GGt # variances of vt
        Kt[:,:, i] = Pt[:,:,i] @ Zt.T @ np.linalg.inv(Ft[:,:, i]) #kalman gain
        att[:,i] =  at[:,i] + Kt[:,:,i] @ vt[:,i] # filtered state variables
        Ptt[:,:,i] = Pt[:,:,i] - Pt[:,:,i] @ Zt.T @ Kt[:,:,i].T # filetered state variables
        logLik += -1/2 * ( np.log(np.linalg.det(Ft[:,:,i])) + vt[:,i].T @ np.linalg.inv(Ft[:,:,i]) @ vt[:,i])
        #prediction
        at[:,i+1] = dt + Tt @ att[:,i] #prediction
        Pt[:,:,i+1] = Tt @ Ptt[:,:,i] @ Tt.T + HHt #prediction


    return {"att":att,"at":at,"Ptt":Ptt,"Pt":Pt,"vt":vt,"Ft":Ft,"Kt":Kt,"logLik":logLik}


order  = 2                  # order of trend component
period = 12                 # period of seasonal component
dim_a = order + period - 1  # dimension of state vector
P0    = np.eye(dim_a )    # initial covariance matrix (V0)
dt    = np.zeros(dim_a)  # mean vector of system noise
ct    = np.zeros(1)           # mean vector of observation noise

# transition matrix (Ft)
Tt           = np.zeros((dim_a,dim_a))
Tt[0,:2]     = [2,-1]
Tt[1,0]      = 1
Tt[2,2:dim_a]= -1
Tt[3:dim_a,2:(dim_a-1)] = np.eye(dim_a-3)


# observation matrix (Ht)
Zt      = np.zeros((1,dim_a))
Zt[0,0] = 1
Zt[0,2] = 1


# initial variance of system noise
HHt0 = [0.20,0.10]

# initial variance of observation noise
GGt0 = 0.20

# observation data
yt = pd.read_csv( "Input_Data.txt" , sep='\t', header=None).to_numpy()[:,2:3].T
  # uses the third column in the data file

# initial state vector (x0)
a0 = np.append( [yt[0,0]], np.zeros(dim_a-1) )

# objective function to be minimized
def J(x,dim_a,a0,P0,dt,ct,Tt,Zt,yt):
  HHt = np.zeros((dim_a,dim_a))
  HHt[0,0] = x[0]
  HHt[2,2] = x[1]
  GGt      = np.array([[x[2]]])
  return -KalmanFilter(a0,P0,dt,ct,Tt,Zt,HHt,GGt,yt)["logLik"]

# optimize the model parameters
constraint = LS(np.array([[1,0,0],[0,1,0],[0,0,1]]), 0, np.inf)
xopt = minimize( J, np.array([HHt0[0], HHt0[1], GGt0]), args = (dim_a,
              a0, P0, dt, ct, Tt, Zt, yt) ,constraints=constraint)

print(f"a0,HHt,GGtの推測値{xopt.x}")
HHt = np.zeros((dim_a,dim_a))
HHt[0,0] = xopt.x[0]
HHt[2,2] = xopt.x[1]
KF = KalmanFilter( a0, P0, dt, ct, Tt, Zt, HHt, np.array([[xopt.x[2]]]), yt)
print(KF["logLik"])

# figures
fig = plt.figure()
ax1 = fig.add_subplot(2,1,1,xlabel="Time", ylabel="yt & trend",
 xlim = (0,70),ylim=(np.floor(np.min(yt[0])), np.ceil(np.max(yt[0]))))
ax2 = fig.add_subplot(2,1,2,xlabel="Time",ylabel="seasonal",xlim = (0,70))
#plt.xlim(0,yt.shape[1])

ax1.plot( yt[0])  # observation data
ax1.plot( KF["att"][0,:-1], color="red" )  # trend component

ax2.plot( KF["att"][2],color="blue" ) # seasonal component
plt.show()
