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

% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
%\keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
