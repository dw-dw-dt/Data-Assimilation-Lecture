######################################
# PFsample2.R                        #
# Sample Program for Particle Filter #
#                                    #
# 2. Time-series decomposition       #
######################################

# observation data
yt <- read.table( "Input_Data.txt" )[ ,3]

# parameters
Nt <- length(yt)  # number of time points
Np <- 100000        # number of particles
torder <- 2       # trend order
period <- 12      # period of seasonal component

# sample particles
tau2u  <- runif( Np, min=0, max=1 )   # variance of system noise of trend component
tau2s  <- runif( Np, min=0, max=1 )   # variance of system noise of seasonal component
sigma2 <- runif( Np, min=0, max=10 )  # variance of observation noise

# preparation for sequential estimation
dimx <- torder + period - 1             # dimension of state vector
x0   <- rep( NA, length=dimx )          # initial state vector
xtp  <- array( NA, dim=c(dimx,Np,Nt) )  # predictive distribution
xtf  <- array( NA, dim=c(dimx,Np,Nt) )  # filter distribution
llh  <- matrix( NA, nrow=Np, ncol=Nt )  # log-likelihood at each time
res  <- c( NA, length=Np )              # number of particles to be resampled

# initial state vector
fit   <- lm( yt[1:(period+1)] ~ c(1:(period+1)) )   # regression line
x0[1] <- fit$coefficients[1]
x0[2] <- fit$coefficients[1] - fit$coefficients[2]
x0[3:dimx] <- fit$residuals[period:2]

# sequential estimation using particle filter
for ( t in 1:Nt ) {
  # one-step ahead prediction
  if (t>1) {
    xtp[1, ,t] <- 2 * xtf[1, ,t-1] - xtf[2, ,t-1] +
                  rnorm( Np, mean=0, sd=sqrt( tau2u ) )
    xtp[2, ,t] <- xtf[1, ,t-1]
    xtp[3, ,t] <- - apply( xtf[3:dimx, ,t-1], 2, sum ) +
                  rnorm( Np, mean=0, sd=sqrt( tau2s ) )
    xtp[4:dimx, ,t] <- xtf[3:(dimx-1), ,t-1]
  } else {
    xtp[1, ,t] <- 2 * x0[1] - x0[2] + rnorm( Np, mean=0, sd=sqrt( tau2u ) )
    xtp[2, ,t] <- x0[1]
    xtp[3, ,t] <- sum( x0[3:dimx] ) + rnorm( Np, mean=0, sd=sqrt( tau2s ) )
    for ( i in 4:dimx ) { xtp[i, ,t] <- x0[i-1] }
  }

  # log-likelihood at this time point
  llh[ ,t] <- - 0.5 * ( log(2*pi*sigma2) + ( yt[t] - xtp[1, ,t] - xtp[3, ,t] )**2 / sigma2 )

  # weight of each particle
  weight <- exp( llh[ ,t] - max(llh[ ,t]) ) / sum( exp( llh[ ,t] - max(llh[ ,t]) ) )

  # filtering based on the systematic residual resampling
  u <- runif(1) / Np

  for ( p in 1:Np ) {
    res[p] <- floor( ( weight[p] - u ) * Np ) + 1
    u <- u + res[p] / Np - weight[p]
  }

  k <- 0
  for ( i in 1:Np ) {
  for ( j in 1:res[i] ) {
    if ( res[i]==0 ) break
    k <- k + 1
    xtf[ ,k,t] <- xtp[ ,i,t]
  }
  }
}

# log-likelihood of each model
llh_model <- apply( llh, 1, sum )

# figures
ut <- apply( xtf[1, , ], 2, mean )
st <- apply( xtf[3, , ], 2, mean )

par( mfrow=c(3,1), oma=c(0,0,0,0) )
plot ( yt, xlab="Time", ylab="yt & ut", type="l", col="black" )
lines( ut, type="l", col="red" )
plot ( st, type="l", col="blue" )
plot ( yt - ut - st, type="l", col="green" )