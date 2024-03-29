#########################################
#  Sample Program using Kalman Filter   #
#                                       #
#  1. One-time Kalman filter            #
#########################################

library(FKF)        # load FKF package

Nt <- 100           # number of time points
a0 <- c(0.0)        # initial state vector (x0)
P0 <- matrix(0.01)  # initial covariance matrix (V0)
dt <- matrix(0.0)   # mean vector of system noise
ct <- matrix(0.0)   # mean vector of observation noise
Tt <- matrix(1.0)   # transition matrix (Ft)
Zt <- matrix(1.0)   # observation matrix (Ht)

HHt_T <- 1.0        # variance of system noise (true)
HHt_F <- 0.2        # variance of system noise (false)
#HHt_F <- 1.0        # variance of system noise (false)

GGt_T <- 4.0        # variance of observation noise (true)
GGt_F <- 1.0        # variance of observation noise (false)
#GGt_F <- 1.0        # variance of observation noise (false)

Ht <- matrix(HHt_F) # covariance matrix of system noise (Qt)
Gt <- matrix(GGt_F) # covariance matrix of observation noise (Rt)

# synthetic data
xt <- cumsum( rnorm( Nt, mean=0, sd=sqrt(HHt_T) ) )      # state vector
yt <- rbind( xt + rnorm( Nt, mean=0, sd=sqrt(GGt_T) ) )  # observation data

# apply Kalman filter
a0_F <- c(50.0)
#KF <- fkf( a0, P0, dt, ct, Tt, Zt, Ht, Gt, yt )
KF <- fkf( a0_F, P0, dt, ct, Tt, Zt, Ht, Gt, yt )


# figures
x_range <- c(1,Nt)
y_range <- c( ceiling( max(yt[1,]) ), floor( min(yt[1,]) ) )

plot( yt[1,], xlim=x_range, ylim=y_range, type="p", ylab="" )  # observation data
par(new=T)
plot( KF$att[1,], xlim=x_range, ylim=y_range, type="l", col="red",
      xlab="", ylab="" )  # mean of filtering distribution
par(new=T)
plot( KF$att[1,]+sqrt(KF$Ptt[1,1,]), xlim=x_range, ylim=y_range,
      type="l", lty=3, col="red", xlab="", ylab="" )
par(new=T)
plot( KF$att[1,]-sqrt(KF$Ptt[1,1,]), xlim=x_range, ylim=y_range,
      type="l", lty=3, col="red", xlab="", ylab="" )