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
  \item{P}{
Number of Gaussian fixed at onece during weighted iterative sampling.  
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
  \item{xi}{
Proportion of samples drawn during weighted iterative sampling. 
}
  \item{Lambda}{
Hyper parameter of Sigma prior.
}
  \item{num_iter}{
Number of iterations for stochastic EM.
}
  \item{num_iter_refine}{
Number of iterations for parameter refinement after weighted iterative sampling.
}
  \item{num_iter_refine}{
Number of iterations for weighted iterative sampling.
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
\item{Z}{Imputed data by Gibbs sampling.}
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
Kodai Minoura, Ko Abe
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
library(Rcpp)
library(RcppArmadillo)
library(MASS)
library(clusterGeneration)

library(CYBERTRACK2)

set.seed(1)
#simulation
L <- 3 #cluster
K <- 10 #variable 
N <- 2000 #samples at each t
T <- 5 #t
D <- 1

true_pi <- matrix(c(0.30,0.20,0.50,
                    0.30,0.20,0.50,
                    0.30,0.20,0.50,
                    0.50,0.25,0.25,
                    0.50,0.25,0.25),T,L,byrow=TRUE)

true_mu <- matrix(runif(K*L,-2,8),K,L,byrow=TRUE)

true_sigma <- array(0,dim=c(K,K,L))
for(l in 1:L) true_sigma[,,l] <- genPositiveDefMat(dim=K,covMethod="unifcorrmat",
                                                   rangeVar=c(0,1))$Sigma


kminit <- function(y,L,seed = sample.int(.Machine$integer.max, 1)){
  set.seed(seed)
  kmres <- kmeans(y,L,iter.max=50,nstart=5,algorithm="Lloyd")
  list(mean=t(kmres$centers),
       var=simplify2array(lapply(split(as.data.frame(y),kmres$cluster),var)),
       cluster=kmres$cluster)
}

Y <- t_id <- clus_id <- list()
for(d in 1:D){
  junk <- tmp_id <- list()
  for(t in 1:T){
    junk[[t]] <- matrix(0,N,K)
    ids <- c()
    for(n in 1:N){
      id <- sample(1:L,1,prob=true_pi[t,])
      ids[n] <- id
      junk[[t]][n,] <- mvrnorm(1,true_mu[,id],true_sigma[,,id])
    }
    tmp_id[[t]] <- ids
  }
  clus_id[[d]] <- unlist(tmp_id)
  Y[[d]] <- do.call(rbind,junk)
  t_id[[d]] <- rep(1:T,each=N)
  Y[[d]][which(Y[[d]]<0)] <- 0
}
kmY <- do.call(rbind,Y)

num_iter <- 100
num_iter_refine <- 50
wis_iter <- 50

tau <- 1e-5
nu <- K+2
Lambda <- diag(K)
kmpar <- kminit(kmY,L,123)
Wini <- list()
for(d in 1:D){
  Wini[[d]] <- matrix(0,N*T,L)
  Wini[[d]][,1] <- 1 
}
piini <- list()
for(d in 1:D) piini[[d]] <- matrix(1/L,T,L)
alphaini <- c(rep(1,T))
muini <- kmpar$mean
Sigmaini <- kmpar$var
for(l in 1:L) Sigmaini[,,l] <- Sigmaini[,,l]+(1e-5*diag(K))

xi=0.10
P <- 1

result <- cybertrack2(Y,L,D,P,Wini,piini,alphaini,muini,Sigmaini,tau,nu,xi,Lambda,num_iter,num_iter_refine, wis_iter, t_id)

}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
