######################################
# PFsample1.R                        #
# Sample Program for Particle Filter #
#                                    #
# 1. 2-D random walk                 #
######################################

# parameters
Nt <- 100             # number of time points
Np <- 10000           # number of particles
tau2_T <- c(0.5,0.2)  # variances of system noise (unknown)
sigma2 <- c(0.1,0.1)  # variances of observation noise (known)

# synthetic observation data
yt <- rbind( cumsum( rnorm( Nt, mean=0, sd=sqrt(tau2_T[1]) ) ),
             cumsum( rnorm( Nt, mean=0, sd=sqrt(tau2_T[2]) ) ) ) +
      rbind( rnorm( Nt, mean=0, sd=sqrt(sigma2[1]) ),
             rnorm( Nt, mean=0, sd=sqrt(sigma2[2]) ) )

# sample particles
tau2 <- rbind( runif( Np, min=0, max=1 ), runif( Np, min=0, max=1 ) )

# preparation for sequential estimation
x0  <- c(0,0)                         # initial state
xtp <- array( NA, dim=c(2,Np,Nt) )    # predictive distribution
xtf <- array( NA, dim=c(2,Np,Nt) )    # filter distribution
llh <- matrix( NA, nrow=Np, ncol=Nt ) # log-likelihood at each time
res <- c( NA, length=Np )             # number of particles to be resampled

# sequential estimation using particle filter
for ( t in 1:Nt ) {
  if (t>1) { x <- xtf[ , ,t-1] } else { x <- x0 }

  # one-step ahead prediction
  xtp[ , ,t] <- x + rbind( rnorm( Np, mean=0, sd=sqrt( tau2[1, ] ) ),
                           rnorm( Np, mean=0, sd=sqrt( tau2[2, ] ) ) )

  # log-likelihood at this time point
  llh[ ,t] <- - log(2*pi) - 0.5 * (
                           log( sigma2[1] ) + log( sigma2[2] )
                         + ( yt[1,t] - xtp[1, ,t] )**2 / sigma2[1]
                         + ( yt[2,t] - xtp[2, ,t] )**2 / sigma2[2] )

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
llh_model <- apply( llh, 2, sum )

# figures
plot( yt[1, ], yt[2, ], xlab="x", ylab="y", type="l", col="black" )
lines( apply( xtf[1, , ], 2, mean ), apply( xtf[2, , ], 2, mean ),
       type="l", col="red" )
