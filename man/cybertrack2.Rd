\name{cybertrack2}
\alias{cybertrack2}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
parameter estimation
}
\description{
Return stochastic EM algorithm output and time-series mixture propotions for mixtures of zero-inflated multivariate Gaussian distribution.
}
\usage{
cybertrack2(Y, L, D, P, Wini, piini, alphaini, muini, Sigmaini, tau, nu, xi, Lambda, num_iter, num_iter_refine, wis_iter, t_id)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{Y}{
List of data for each case. 
}
  \item{L}{
Number of clusters.
}
  \item{D}{
Number of cases.
}
  \item{piini}{
Initial value of mixture proportion.
}
  \item{alphaini}{
Initial value of persistency parameter.
}
  \item{muini}{
Initial value of mean for multivariate Gaussian distribution.
}
  \item{Sigmaini}{
Initial value of variance matrix for multivariate Gaussian distribution.
}
  \item{tau}{
Hyper parameter of mu prior.
}
  \item{nu}{
Hyper parameter of Sigma prior.
}
  \item{Lambda}{
Hyper parameter of Sigma prior.
}
  \item{num_iter}{
Number of iterations.
}
  \item{t_id}{
List of timepoint index for each sample in each case.
}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
%%  ~Describe the value returned
\item{pi}{The final mixture proportion.}
\item{alpha}{The final persistency parameter.}
\item{Sigma}{The final variance matrix.}
\item{mu}{The final mean.}
\item{W}{The latent cluster of each sample.}
\item{loglik}{The history of log-likelihood.}
}
\references{
%% ~put references to the literature/web site here ~
}
\author{
%%
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
library(mvtnorm)
set.seed(1234)
sample <- 1000000

L <- 3 #number of clusters 
K <- 2 #number of variables 
N <- 1000 #samples at each timepoint
T <- 5 #number of timepoints
D <- 2 #number of cases

true_pi <- list()
true_pi[[1]] <- matrix(c(0.3,0.3,0.4,
                         0.3,0.3,0.4,
                         0.2,0.5,0.3,
                         0.2,0.5,0.3,
                         0.8,0.1,0.1),L,T,byrow=FALSE)
true_pi[[2]] <- matrix(c(0.4,0.3,0.3,
                         0.1,0.1,0.8,
                         0.1,0.1,0.8,
                         0.2,0.2,0.6,
                         0.2,0.2,0.6),L,T,byrow=FALSE)

true_mu <- matrix(c(-3,2,
                    2,1,
                    0,-3),L,K)
true_sigma <- array(,dim=c(K,K,L))
for(i in 1:L) true_sigma[,,i] <- diag(K)

Y <- t_id <- list()
for(d in 1:D){
  mn <- list()
  for(i in 1:L) mn[[i]] <- mvrnorm(sample,true_mu[i,],true_sigma[,,i]) 
  junk <- pi_id <- id <- list()
  for(t in 1:T){
    junk[[t]] <- matrix(NA_real_,N,K)
    pi_id[[t]] <- sample(1:L,N,replace=TRUE,prob=true_pi[[d]][,t])
    id[[t]] <- sample(1:sample,N,replace=TRUE)
    for(i in 1:N){
      junk[[t]][i,] <- mn[[pi_id[[t]][i]]][id[[t]][i],] 
    }
  }  
  Y[[d]] <- do.call(rbind,junk)
  t_id[[d]] <- rep(1:T,each=N)
}

kminit <- function(y,L,seed = sample.int(.Machine$integer.max, 1)){
  set.seed(seed)
  kmres <- kmeans(y,L,iter.max=100,nstart=3,algorithm="Lloyd")
  list(mean=t(kmres$centers),
       var=simplify2array(lapply(split(as.data.frame(y),kmres$cluster),var)),
       cluster=kmres$cluster)
}
kmY <- do.call(rbind,Y)

num_iter <- 100
tau <- 0.1
nu <- K+1
Lambda <- diag(K)
kmpar <- kminit(kmY,L,123)
piini <- matrix(1/L,T,L)
alphaini <- c(rep(1,T))
muini <- kmpar$mean
Sigmaini <- kmpar$var

result <- cybertrack(Y,L,D,piini,alphaini,muini,Sigmaini,tau,nu,Lambda,num_iter,t_id)
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
