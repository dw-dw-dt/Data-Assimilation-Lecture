########################################
# Decomp_EnKF.R                        #
#                                      #
# Time-series decomposition using EnKF #
########################################

#library(MASS)

# observation data
yt <- read.table( "../Input_Data.txt" )[ ,3]

# parameters
Nt <- length(yt)  # number of time points
#Np <- 1000        # number of ensemble members
Np <- 50        # number of ensemble members
torder <- 2       # trend order
period <- 12      # period of seasonal component

# system noises
tau2u  <- 25  # variance of system noise of trend component
tau2s  <- 25  # variance of system noise of seasonal component
sigma2 <- 25  # variance of observation noise

# preparation for sequential estimation
dimx <- torder + period - 1               # dimension of state vector
xtp  <- array( NA, dim=c(dimx,Np,Nt) )  # predictive distribution
xtf  <- array( NA, dim=c(dimx,Np,Nt) )  # filter distribution

# observation matrix (vector)
Ht <- matrix( NA, nrow=1, ncol=dimx )
Ht[1, ] <- c( 1, 0, 1, rep(0,dimx-3) )

# variables for DA
x0   <- matrix( 0, nrow=dimx, ncol=Np )  # initial state vector
xtpm <- matrix( 0, nrow=dimx, ncol=Nt )  # mean of predictive distribution
xtpv <- matrix( 0, nrow=dimx, ncol=Nt )  # variance of predictive distribution
xtfm <- matrix( 0, nrow=dimx, ncol=Nt )  # mean of filter distribution
xtfv <- matrix( 0, nrow=dimx, ncol=Nt )  # variance of filter distribution

# initial state vector
fit     <- lm( yt[1:(period+1)] ~ c(1:(period+1)) )   # regression line
x0[1, ] <- fit$coefficients[1] + rnorm( Np, mean=0, sd=sqrt( tau2u ) )
x0[2, ] <- fit$coefficients[1] - fit$coefficients[2]
x0[3, ] <- fit$residuals[period] + rnorm( Np, mean=0, sd=sqrt( tau2s ) )
x0[4:dimx, ] <- fit$residuals[(period-1):2]

# EnKF
for ( t in 1:Nt ) {
  message( paste( "Assimilating t=", t, sep="" ) )

  # one-step ahead prediction
  if (t>1) {
    xtp[1, ,t] <- 2 * xtf[1, ,t-1] - xtf[2, ,t-1] +
                  rnorm( Np, mean=0, sd=sqrt( tau2u ) )
    xtp[2, ,t] <- xtf[1, ,t-1]
    xtp[3, ,t] <- - apply( xtf[3:dimx, ,t-1], 2, sum ) + rnorm( Np, mean=0, sd=sqrt( tau2s ) )
    xtp[4:dimx, ,t] <- xtf[3:(dimx-1), ,t-1]
  } else {
    xtp[1, ,t] <- 2 * x0[1, ] - x0[2, ] + rnorm( Np, mean=0, sd=sqrt( tau2u ) )
    xtp[2, ,t] <- x0[1, ]
    xtp[3, ,t] <- - apply( x0[3:dimx, ], 2, sum ) + rnorm( Np, mean=0, sd=sqrt( tau2s ) )
    for ( i in 4:dimx ) { xtp[i, ,t] <- x0[i-1, ] }
  }

  xtpm[ ,t] <- apply( xtp[ , ,t], 1, mean )

  # filtering
  dxtp <- xtp[ , ,t] - xtpm[ ,t]
  wt  <- rnorm( Np, mean=0, sd=sqrt(sigma2) )
  dw  <- wt - mean( wt )
  bb  <- 1 / ( Ht %*% ( dxtp %*% t(dxtp) ) %*% t(Ht) + mean(wt*wt) )
  xtf[ , ,t] <- xtp[ , ,t] + ( dxtp %*% t(dxtp) ) %*% t(Ht) %*% bb %*% ( yt[t] + dw - Ht %*% xtp[ , ,t] )

  xtfm[ ,t] <- apply( xtf, 1, mean )
}


# figures
ut <- apply( xtf[1, , ], 2, mean )
st <- apply( xtf[3, , ], 2, mean )

par( mfrow=c(3,1), oma=c(0,0,0,0) )
plot ( yt, xlab="Time", ylab="yt & ut", type="l", col="black" )
lines( ut, type="l", col="red" )
plot ( st, type="l", col="black", ylim=c(-150,150) )
plot ( yt - ut - st, type="l", col="black", ylim=c(-1,1) )