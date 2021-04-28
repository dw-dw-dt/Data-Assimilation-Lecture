####################################################
#  Sample Program using Kalman Filter              #
#                                                  #
#  3. Extraction of Trend & Seasonal Variations    #
####################################################

library(FKF)                 # load FKF package

order  <- 2                  # order of trend component
period <- 12                 # period of seasonal component
dim_a <- order + period - 1  # dimension of state vector
P0    <- diag( 1, dim_a )    # initial covariance matrix (V0)
dt    <- matrix( 0, dim_a )  # mean vector of system noise
ct    <- matrix(0)           # mean vector of observation noise

# transition matrix (Ft)
Tt            <- matrix( 0, nrow=dim_a, ncol=dim_a )
Tt[1,1:2]     <- c(2,-1)
Tt[2,1]       <- 1
Tt[3,3:dim_a] <- -1
diag(Tt[4:dim_a,3:(dim_a-1)]) <- 1

# observation matrix (Ht)
Zt      <- matrix( 0, nrow=1, ncol=dim_a )
Zt[1,1] <- 1
Zt[1,3] <- 1

# initial variance of system noise
HHt0 <- c(0.20,0.10)

# initial variance of observation noise
GGt0 <- 0.20

# observation data
yt <- t( as.matrix( read.table( "Input_Data.txt" ) )[,3] )
  # uses the third column in the data file

# initial state vector (x0)
a0 <- c( yt[1,1], rep(0,dim_a-1) )

# objective function to be minimized
J <- function( par, dim_a, ... ) {
  HHt <- matrix( 0, nrow=dim_a, ncol=dim_a )
  HHt[1,1] <- par[1]
  HHt[3,3] <- par[2]
  GGt      <- matrix( par[3] )
  return( -fkf( HHt=HHt, GGt=GGt, ... )$logLik )
}

# optimize the model parameters
opt <- optim( c( HHt=HHt0, GGt=GGt0 ), fn=J, dim_a=dim_a,
              a0=a0, P0=P0, dt=dt, ct=ct, Tt=Tt, Zt=Zt, yt=yt )

# Kalman filter using the optimzed parameters
HHt      <- matrix( 0, nrow=dim_a, ncol=dim_a )
HHt[1,1] <- opt$par[1]
HHt[3,3] <- opt$par[2]
GGt      <- matrix( opt$par[3] )
KF <- fkf( a0, P0, dt, ct, Tt, Zt, HHt, GGt, yt )

# figures
x_range <- c(1,dim(yt)[2])
y_range <- c( floor( min(yt[1,]) ), ceiling( max(yt[1,]) ) )
par( mfrow=c(2,1), oma=c(0,0,0,0) )

plot( yt[1,], xlim=x_range, ylim=y_range, type="l",
      xlab="Time", ylab="yt & trend" )  # observation data
lines( KF$att[1,], type="l", col="red" )  # trend component
#lines( KF$att[1,]+sqrt(KF$Ptt[1,1,]), lty=3, col="red" )
#lines( KF$att[1,]-sqrt(KF$Ptt[1,1,]), lty=3, col="red" )

plot( KF$att[3,], xlim=x_range, type="l", col="blue",
      xlab="Time", ylab="seasonal" ) # seasonal component
#lines( KF$att[3,]+sqrt(KF$Ptt[3,3,]), lty=3, col="blue" )
#lines( KF$att[3,]-sqrt(KF$Ptt[3,3,]), lty=3, col="blue" )