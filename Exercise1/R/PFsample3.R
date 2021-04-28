#####################################################
# PFsample3.R                                       #
#                                                   #
# Data Assimilation on Advection-Diffusion Equation #
#####################################################

# parameters
u0     <- 8.0    # advection velocity (true/unknown)
v0     <- 5.0    # diffusion coefficient (true/unknown)
tau20  <- 1.0    # variance of external force (true/unknown)
sigma2 <- 16.0   # variance of observation noise (true/known)

um     <- 5.0    # mean of prior of u
usd    <- 3.0    # standard deviation of prior of u
vm     <- 6.0    # mean of prior of u
vsd    <- 2.0    # standard deviation of prior of v
tau2m  <- 0.8    # mean of prior of u
tau2sd <- 0.5    # log-standard deviation of prior of tau2

# grid spacing
dt <- 0.5            # grid spacing in time
Nt <- 1200           # number of grids in time
t  <- 0:(Nt-1) * dt  # real coordinate in time
dx <- 10.0           # grid spacing in space
Nx <- 400            # number of grids in space
x  <- 0:(Nx-1) * dx  # real coordinate in space
xl <- c(2:Nx,1)      # vector letting 1:Nx shift leftward
xr <- c(Nx,1:(Nx-1)) # vector letting 1:Nx shift rightward
Np <- 1000           # number of particles

# function for simulation
simulation <- function( u, v, tau2 ) {
  C <- matrix( 0, nrow=Nx, ncol=Nt )

  for ( k in 2:Nt ) {
    A <- - u * ( C[xl,k-1] - C[xr,k-1] ) / ( 2 * dx )          # advection
    K <- max(k-2,1)
    S <- v * ( C[xl,K] - 2 * C[ ,K] + C[xr,K] ) / ( dx * dx )  # diffusion
    W <- sin( pi * x / 60 ) * sin( 2 * pi * t[k-1] / 120 ) +
           rnorm( Nx, mean=0, sd=sqrt(tau2) )                  # external force
    C[ ,k] <- 2 * dt * ( A + S + W )
    if ( k>2 ) C[ ,k] <- C[ ,k-2] + C[ ,k]
  }

  return( C )
}

# synthetic observation data
yt0 <- simulation( u0, v0, tau20 )  # true
yts <- simulation( um, vm, tau2m )  # simulation using guess parameters

yt  <- matrix( NA, nrow=Nx, ncol=Nt )  # observed concentration
observedX <- seq( 3,Nx,20)             # observed places
observedT <- seq(10,Nt,15)             # observed time
yt[ observedX, observedT ] <- yt0[ observedX, observedT ]
yt <- yt + rnorm( length(yt), mean=0, sd=sqrt(sigma2) )
                            # add observation noise

# variables for DA
xtp  <- array( 0, dim=c(Nx,2,Np) )     # predictive distribution
xtf  <- array( 0, dim=c(Nx,2,Np) )     # filter distribution
xtfm <- matrix( 0, nrow=Nx, ncol=Nt )  # mean of predictive distribution
llh  <- rep( 0, length=Np )            # log-likelihood
res  <- rep( 0, length=Np )            # number of resamples

# sample particles from priors
u    <- rnorm( Np, mean=um, sd=usd )
v    <- rnorm( Np, mean=vm, sd=vsd )
tau2 <- rlnorm( Np, meanlog=log(tau2m), sdlog=tau2sd )

uPrior <- u; vPrior <- v; tau2Prior <- tau2

# particle filter
for ( k in 2:Nt ) {
  # one-step ahead prediction
  A <- - ( xtf[xl,1, ] - xtf[xr,1, ] ) %*% diag(u) / ( 2 * dx )
  S <- ( xtf[xl,2, ] - 2 * xtf[ ,2, ] + xtf[xr,2, ] ) %*% diag(v) / ( dx * dx )
  W <- sin( pi * x / 60 ) * sin( 2 * pi * t[k] / 120 ) +
         matrix( rnorm( Nx*Np, mean=0, sd=sqrt(tau2) ), nrow=Nx, byrow=T )
  xtp[ ,1, ] <- xtf[ ,2, ] + 2 * dt * ( A + S + W )
  xtp[ ,2, ] <- xtf[ ,1, ]

  # skip filtering step if no observation
  if ( all( observedT!=k ) ) {
    xtf <- xtp
    xtfm[ ,k] <- apply( xtf[ ,1, ], 1, mean )
    next
  } 

  # log-likelihood at this time point
  dy  <- yt[observedX,k] - xtp[observedX,1, ]
  llh <- - 0.5 * ( log( 2 * pi * sigma2 ) + apply( dy*dy, 2, sum ) / sigma2 )

  # filtering using the residual systematic resampling
  weight <- exp( llh - max(llh) ) / sum( exp( llh - max(llh) ) )
  r <- runif(1) / Np
  uTemp <- u; vTemp <- v; tau2Temp <- tau2

  for ( p in 1:Np ) {
    res[p] <- floor( ( weight[p] - r ) * Np ) + 1
    r <- r + res[p] / Np - weight[p]
  }

  for ( p in which(res>0) ) {
    prange <- ( sum( res[p-1] ) + 1 ) : sum( res[p] )
    xtf[ , ,prange] <- xtp[ , ,p]
    u[prange] <- uTemp[p]
    v[prange] <- vTemp[p]
    tau2[prange] <- tau2Temp[p]
  }

  xtfm[ ,k] <- apply( xtf[ ,1, ], 1, mean )
}

# figures
# prior & posterior distributions of the model parameters
jpeg( file="paramDist.jpg", quality=100,
      width=1000, height=1000, pointsize=24 )
par( mfcol=c(3,2) )
hist( uPrior, breaks=seq( floor(min(uPrior)), ceiling(max(uPrior)), 0.5 ),
      main="p(u)", xlab="u", col="gray" )
hist( vPrior, breaks=seq( floor(min(vPrior)), ceiling(max(vPrior)), 0.5 ),
      main="p(v)", xlab="v", col="gray" )
hist( tau2Prior, breaks=seq( floor(min(tau2Prior)), ceiling(max(tau2Prior)), 0.1 ),
      main="p(tau2)", xlab="tau2", col="gray" )
hist( u, breaks=seq( floor(min(uPrior)), ceiling(max(uPrior)), 0.5 ),
      main="p(u|y)", xlab="u", col="gray" )
hist( v, breaks=seq( floor(min(vPrior)), ceiling(max(vPrior)), 0.5 ),
      main="p(v|y)", xlab="v", col="gray" )
hist( tau2, breaks=seq( floor(min(tau2Prior)), ceiling(max(tau2Prior)), 0.1 ),
      main="p(tau2|y)", xlab="tau2", col="gray" )
dev.off()

# true, simulated, observed & assimilated concentrations
jpeg( file="concentration.jpg", quality=100,
      width=1200, height=1000, pointsize=24 )
par( mfrow=c(2,2), mar=c(4,5,2,2) )
image( x, t, yt0, xlim=c(0,Nx*dx), ylim=c(0,Nt*dt),
       main="True", xlab="", ylab="t" )
image( x, t, yts, xlim=c(0,Nx*dx), ylim=c(0,Nt*dt),
       main="Simulation", xlab="", ylab="t" )
image( x, t, yt , xlim=c(0,Nx*dx), ylim=c(0,Nt*dt),
       main="Observation", xlab="x", ylab="t" )
image( x, t, xtfm, xlim=c(0,Nx*dx), ylim=c(0,Nt*dt),
       main="Assimilation", xlab="x", ylab="t" )
dev.off()
