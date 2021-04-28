#########################################
#  Sample Program using Kalman Filter   #
#                                       #
#  2. Optimization of Model Parameters  #
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
GGt_T <- 4.0        # variance of observation noise (true)
GGt_F <- 1.0        # variance of observation noise (false)

Ht <- matrix(HHt_F) # covariance matrix of system noise (Qt)
Gt <- matrix(GGt_F) # covariance matrix of observation noise (Rt)

# synthetic data
xt <- cumsum( rnorm( Nt, mean=0, sd=sqrt(HHt_T) ) )      # state vector
yt <- rbind( xt + rnorm( Nt, mean=0, sd=sqrt(GGt_T) ) )  # observation data

# objective function to be minimized
#J <- function( par, ... ) {
#  -fkf( HHt = matrix( par[1] ), GGt = matrix( par[2] ), ... )$logLik
#}

J <- function( par, ... ) {
  -fkf( HHt = matrix( par[1] ), GGt = matrix( par[2] ), a0 = c(par[3]), ... )$logLik
}


# optimize the model parameters
#opt <- optim( c( HHt=HHt_F, GGt=GGt_F ), fn=J,
#              a0=a0, P0=P0, dt=dt, ct=ct, Tt=Tt, Zt=Zt, yt=yt )

a0_F <- c(10.0)

opt <- optim( c( HHt=HHt_F, GGt=GGt_F, a0=a0_F ), fn=J,
              P0=P0, dt=dt, ct=ct, Tt=Tt, Zt=Zt, yt=yt )

# Kalman filter using the optimzed parameters
#KF <- fkf( a0, P0, dt, ct, Tt, Zt, matrix(opt$par[1]), matrix(opt$par[2]), yt )
KF <- fkf( opt$par[3], P0, dt, ct, Tt, Zt, matrix(opt$par[1]), matrix(opt$par[2]), yt )


# figures
x_range <- c(1,Nt)
y_range <- c( floor( min(yt[1,]) ), ceiling( max(yt[1,]) ) )

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