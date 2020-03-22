# CYBERTRACK2

###Depends:

R(>=3.6.0)

Rcpp, RcppARmadillo

###Authors:

Kodai Minoura and Ko Abe

Contact: minoura.kodai[at]e.mbox.nagoya-u.ac.jp and ko.abe[at]med.nagoya-u.ac.jp

## Installation

Install the latest version of this package from Github by pasting in the following.

~~~R
devtools::install_github("kodaim1115/CYBERTRACK2")
~~~

## An example of synthetic data

~~~R
library(Rcpp)
library(RcppArmadillo)
library(MASS)
library(clusterGeneration)

library(CYBERTRACK2)

set.seed(1)
#simulation
L <- 3 #cluster
K <- 10 #variables 
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
  kmres <- kmeans(y,L,iter.max=50,nstart=3,algorithm="Lloyd")
  list(mean=t(kmres$centers),
       var=simplify2array(lapply(split(as.data.frame(y),kmres$cluster),var)),
       cluster=kmres$cluster)
}

#Generate synthetic longitudinal mass cytometry data.
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

xi=0.10 #sampling proportion for weighted iterative sampling.
P <- 1 #number of clusters to be fixed ata once.

result <- cybertrack2(Y,L,D,P,Wini,piini,alphaini,muini,Sigmaini,tau,nu,xi,Lambda,num_iter,num_iter_refine, wis_iter, t_id)
~~~

## Genral overview

## Reference
