#
# EnKFadvection.R
#
# Data Assimilation on an advection equation using EnKF
#
#             ∂C       ∂C
#             -- + u * -- = 0
#             ∂t       ∂x
#
#
#

library(MASS)

# parameters
u0      <- -4.0     # 真の移流速度 (true/unknown)
sigma2  <-  0.1     # 真の観測ノイズの分散 (true/known)
um      <-  0.0     # 移流速度の事前分布の平均
usd     <-  2.0     # 移流速度の事前分布の標準偏差


# grid spacing
dt <- 0.5            # シミュレーションの時間ステップ間隔
Nt <- 500            # シミュレーションの全時間ステップ数
t  <- 0:(Nt-1) * dt  # 実時間の座標
dx <- 10.0           # 空間差分の間隔
Nx <- 400            # 空間の全分割数
x  <- 0:(Nx-1) * dx  # 実空間の座標

x1  <- 1:Nx           # 空間の分割要素の集合
xl1 <- c(2:Nx,1)      # 空間の分割要素の集合を左へシフトした集合 周期境界条件を適用
xr1 <- c(Nx,1:(Nx-1)) # 空間の分割要素の集合を右へシフトした集合 周期境界条件を適用

Np   <- 2000          # EnKFのアンサンブルメンバーの数

xini <- 50:150 # 初期条件の要素集合

umax <-  0.8*dx/dt # CFL条件から決まるuの上限
umin <- -0.8*dx/dt # CFL条件から決まるuの下限



exact <- function( u ) { # 厳密解
    C <- matrix( 0, nrow=Nx, ncol=Nt )
    C[xini,1] <- 1
    tmp <- c( C[ ,1],C[ ,1] )
    for ( k in 2:Nt ) {
        rshift <- floor( (k*u*dt/dx)%%(Nx) )
        xshift <- x1 + Nx - rshift
        C[x1,k] <- tmp[xshift]
    }
    return( C )
}




simulation <- function( u ) { # 風上差分を使ったシミュレーションの解
  C <- matrix( 0, nrow=Nx, ncol=Nt )
  C[xini,1] <- 1
  for ( k in 2:Nt ) {
      A <- - u * ( C[xl1,k-1] - C[xr1,k-1] ) / ( 2 * dx )    # advection
      S <- abs( u ) * ( C[xl1,k-1] + C[xr1,k-1] - 2 * C[ ,k-1] ) / ( 2 * dx )
      C[ ,k] <- C[ ,k-1] + ( A + S ) * dt
  }
  return( C )
}



# synthetic observation data
yte <- exact( u0 )
yt0 <- exact( u0 )  # true
yts <- simulation( um )  # simulation using initial guess parameters



yt  <- matrix( NA, nrow=Nx, ncol=Nt )  # observed concentration
observedX <- seq( 1,Nx,20)             # observed places
observedT <- seq( 10,Nt,10)            # observed time
#observedX <- 1:Nx              # observed places
#observedT <- 1:Nt              # observed time
yt[ observedX, observedT ] <- yt0[ observedX, observedT ]
for (k in observedT){
    yt[ observedX, k ] <- yt[ observedX, k ] + rnorm( length(observedX), mean=0, sd=sqrt(sigma2) )
}

xtfini  <- matrix( NA, nrow=Nx, ncol=Nt )
xtfini[ observedX, observedT ] <- yts[ observedX, observedT ]



# observation matrix
Ht <- matrix( 0, nrow=length(observedX), ncol=Nx+1 )
diag( Ht[ ,observedX] ) <- 1



# variables for DA
xtp  <- matrix( 0, nrow=Nx+1, ncol=Np )  # predictive distribution
xtf  <- matrix( 0, nrow=Nx+1, ncol=Np )  # filter distribution
xtpm <- matrix( 0, nrow=Nx+1, ncol=Nt )  # mean of predictive distribution
xtfm <- matrix( 0, nrow=Nx+1, ncol=Nt )  # mean of filter distribution
xtfv <- matrix( 0, nrow=Nx+1, ncol=Nt )  # variance of filter distribution

llh  <- -Nx*Nt*log(2*pi*sigma2)
llhm <- rep( NA, length=Nt )

uenm <- matrix( NA, nrow=2, ncol=Nt )    # Ensemble average & standard deviation of u





# sample particles from priors
xtf[xini, ] <- 1
xtf[Nx+1, ] <- rnorm( Np, mean=um, sd=usd )

#
idx <- which( xtf[Nx+1, ] > umax )
xtf[Nx+1,idx] <- umax
idx <- which( xtf[Nx+1, ] < umin )
xtf[Nx+1,idx] <- umin

uPrior <- xtf[Nx+1, ]


llhbak <- rep( NA, length=Nt )
zz <- 0.0
for (k in Nt:1){
    if ( any( observedT==k ) ) {
        llhbak[k] <- zz
        res <- yt[ observedX, k ] - xtfini[ observedX, k ]
        zz <- zz - c(res) %*% c(res) /(2*sigma2)
    }
}

uenm[1,1] <- mean(xtf[Nx+1, ])
uenm[2,1] <- sd(xtf[Nx+1, ])




# EnKF
for ( k in 2:Nt ) {
    message( paste( "Assimilating t=", k, sep="" ) )

  # one-step ahead prediction
  A <- - ( xtf[xl1, ] - xtf[xr1, ] ) %*% diag( xtf[Nx+1, ] ) / ( 2 * dx )
  S <- ( xtf[xl1, ] + xtf[xr1, ] - 2 * xtf[x1, ] ) %*% diag( abs( xtf[Nx+1, ] ) ) / ( 2 * dx )
  v <-  matrix( rnorm(length(x1)*Np,mean=0, sd=0.01 ),ncol=Np)
  xtp[x1, ] <- xtf[x1, ] + ( A + S ) * dt + v[x1,]
  xtp[Nx+1, ] <- xtf[Nx+1, ] # + rnorm( Np, mean=0, sd=0.1 ) #システムノイズなし
  xtpm[ ,k] <- apply( xtp, 1, mean )

# skip filtering step if no observation
  if ( all( observedT!=k ) ) {
    xtf <- xtp
    xtfm[ ,k] <- apply( xtf, 1, mean )
    next
  }

# filtering
    dxtp <- xtp - xtpm[ ,k]
    wt <- matrix( rnorm( length(observedX)*Np, mean=0, sd=sqrt(sigma2) ),
                nrow=length(observedX) )
    dw  <- wt - apply( wt, 1, mean )
    bb  <- ginv( Ht %*% dxtp %*% t(dxtp) %*% t(Ht) + wt%*%t(wt) )
    xtf <- xtp + xtp %*% t(dxtp) %*% t(Ht) %*% bb %*% ( yt[observedX,k] + dw - Ht %*% xtp )


 #
    idx <- which( xtf[Nx+1, ] > umax )
    xtf[Nx+1,idx] <- umax
    idx <- which( xtf[Nx+1, ] < umin )
    xtf[Nx+1,idx] <- umin


    uenm[1,k] <- mean(xtf[Nx+1, ])
    uenm[2,k] <- sd(xtf[Nx+1, ])
    message( paste( "Estimated u = ", uenm[ 1,k ] ,sep=""))
    message( paste( "Standard deviation of u = ", uenm[ 2,k ] ,sep=""))


    xtfm[ ,k] <- apply( xtf, 1, mean )


# log-likelihood at this time point
    if ( any( observedT==k ) ) {
        res <- yt[ observedX, k ] - Ht %*% xtfm[ ,k]
        llh <- llh - c(res) %*% c(res) /(2*sigma2)
        llhm[k] <- llh + llhbak[k]
        message( paste( "Log-likelihood = ", llhm[k], sep="" ) )
    }
    message(" ")
}


# figures
# prior & posterior distributions of the model parameters
jpeg( file="dist_EnKF.jpg", quality=100,
      width=1000, height=1000, pointsize=24 )
par( mfcol=c(2,1) )
hist( uPrior, breaks="Scott", main="Prior PDF p(u)", xlab="u", col="gray")
hist( xtf[Nx+1, ], breaks="Scott", main="Posterior PDF p(u|y)", xlab="u", col="gray")
dev.off()

# true, simulated, observed & assimilated concentrations
jpeg( file="concentration_EnKF.jpg", quality=100,
      width=1200, height=1000, pointsize=24 )
par( mfrow=c(2,2), mar=c(4,5,2,2) )
image( x, t, yte, xlim=c(0,Nx*dx), ylim=c(0,Nt*dt),
       main="True", xlab="", ylab="t" )
image( x, t, yts, xlim=c(0,Nx*dx), ylim=c(0,Nt*dt),
       main="Simulation", xlab="", ylab="t" )
image( x, t, yt , xlim=c(0,Nx*dx), ylim=c(0,Nt*dt),
       main="Observation", xlab="x", ylab="t" )
image( x, t, xtfm[x1, ], xlim=c(0,Nx*dx), ylim=c(0,Nt*dt),
       main="Assimilation", xlab="x", ylab="t" )
dev.off()

# time series of estimated value
utr <- rep( u0, length=Nt )
observedt <- observedT*dt

jpeg( file="timeseries_estimated_EnKF.jpg", quality=100, width=1200, height=600, pointsize=24 )
    plot ( 0, 0, type="l",xlim=range(t), ylim=c( min(uenm[1,observedT]-2*uenm[2,observedT]), max(uenm[1,observedT]+2*uenm[2,observedT])), xlab="Time", ylab="Estimated value")
    lines( t, utr, col="red")
    points( observedt, uenm[1,observedT], col="blue")
    arrows( observedt, uenm[1,observedT]+uenm[2,observedT], observedt, uenm[1,observedT]-uenm[2,observedT], angle = 90, length=0.1, col="blue")
    arrows( observedt, uenm[1,observedT]-uenm[2,observedT], observedt, uenm[1,observedT]+uenm[2,observedT], angle = 90, length=0.1, col="blue")
dev.off()

llhm[1]  <- llhm[2]
# time series of log-likelihood
jpeg( file="timeseries_loglikelihood_EnKF.jpg", quality=100, width=1200, height=600, pointsize=24 )
plot ( 0, 0, type="b",xlim=range(t), ylim=range(llhm[observedT]), xlab="Time", ylab="Log-likelihood")
points( observedt, llhm[observedT], col="blue")
dev.off()
