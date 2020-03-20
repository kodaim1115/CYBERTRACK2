// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>
//#include <progress.hpp>
using namespace Rcpp;

const double log2pi = std::log(2.0 * M_PI);

arma::rowvec rcate(const arma::rowvec & p){
  int K = p.n_cols;
  arma::rowvec cump = cumsum(p);
  arma::rowvec x(K);
  x.fill(0);
  double U = R::runif(0,1);
  if(U<=cump[0]){
    x[0] = 1;
  }else{
    for(int k=1; k<K; k++){
      if(cump[k-1]<U & U<=cump[k]){
        x[k] = 1;
      }
    }
  }
  return(x);
}


// sub1 returns a matrix x[-i,-i]
arma::mat sub1(arma::mat x, int i) {
  x.shed_col(i);
  x.shed_row(i);
  return x;
}

// sub2 returns a matrix x[a,-b]
arma::mat sub2(arma::mat x, int a, int b){
  x.shed_col(b);
  return(x.row(a));
}

// negSubCol returns a column vector x[-i]
arma::vec negSubCol(arma::vec x, int i){
  x.shed_row(i);
  return(x);
}

// negSubRow returns a row vector x[-i]
arma::rowvec negSubRow(arma::rowvec x, int i){
  x.shed_col(i);
  return(x);
}

// [[Rcpp::export]]
arma::vec myrunif(int d){
  return arma::randu(d);
}

// [[Rcpp::export]]
arma::mat rtmvnorm_gibbs(arma::vec mu, arma::mat omega, arma::vec init_state){
  // Rprintf("Start gibbs\n");
  int d = mu.n_elem; //check dimension of target distribution
  
  //draw from U(0,1)
  arma::vec U = arma::randu(d);
  
  //calculate conditional standard deviations
  double var;
  arma::vec x = init_state;
  for(int i=0; i<d; i++){
    if(init_state[i]<=0){
      var = 1/omega(i,i);
      double mu_i = mu(i) - var*arma::as_scalar(sub2(omega,i,i)*(negSubCol(x,i)-negSubCol(mu,i)));
      //transformation
      double Fb = R::pnorm5(0,mu_i,std::sqrt(var),true,false);
      x(i) = mu_i + std::sqrt(var) * R::qnorm5(U(i) * Fb + 1e-100,0.0,1.0,1,0); 
    }
  }
  return x;
}

double logsumexp(const arma::rowvec & x){
  double maxx = max(x);
  double out = maxx + std::log(sum(exp(x-maxx)));
  return out;
}

// [[Rcpp::export]]
arma::rowvec softmax(const arma::rowvec & x){
  double den = logsumexp(x);
  arma::rowvec res = x;
  if(arma::is_finite(den)){
    res = exp(res - den);
  }else{
    res.fill(0);
    res.elem(arma::find(x==max(x))).fill(1);
    res = res/sum(res);
  }
  return res;
}

arma::vec colSums(const arma::mat & X){
  int nCols = X.n_cols;
  arma::vec out(nCols);
  for(int i = 0; i < nCols; i++){
    out(i) = sum(X.col(i));
  }
  return(out);
}

arma::vec colMeans(const arma::mat & X){
  int nCols = X.n_cols;
  int nRows = X.n_rows;
  arma::vec out(nCols);
  for(int i = 0; i < nCols; i++){
    out(i) = sum(X.col(i))/nRows;
  }
  return(out);
}

//[[Rcpp::export]]
arma::mat weighted_colMeans(Rcpp::List Y, Rcpp::List W, const double & tau, int L, int D){
  
  arma::mat tmpY = Y[0];
  int K = tmpY.n_cols; //variable number
  arma::mat out(K,L);
  for(int l=0;l<L;l++){
    for(int i=0;i<K;i++){
      double den = 0;
      double Sum = 0;
      for(int d=0;d<D;d++){
        arma::mat tmpY = Y[d];
        arma::mat tmpW = W[d];
        den += arma::sum(tmpW.col(l));
        Sum += arma::sum(tmpW.col(l) % tmpY.col(i));
      }
      out(i,l) = Sum/(den+tau);
    }  
  }
  return(out);
}

//[[Rcpp::export]]
arma::mat weighted_colMeans_fix(Rcpp::List Y, Rcpp::List W, arma::mat premu,
                                   const double & tau, int L, int D, arma::uvec fix_id){
  
  arma::mat tmpY = Y[0];
  int K = tmpY.n_cols; //variable number
  arma::mat out(K,L);
  //arma::uvec id = arma::find(fix_id == 1);
  for(int l=0;l<L;l++){
    if(fix_id(l) == 1){
      out.col(l) = premu.col(l);
    } else{
      for(int i=0;i<K;i++){
        double den = 0;
        double Sum = 0;
        for(int d=0;d<D;d++){
          arma::mat tmpY = Y[d];
          arma::mat tmpW = W[d];
          den += arma::sum(tmpW.col(l));
          Sum += arma::sum(tmpW.col(l) % tmpY.col(i));
        }
        out(i,l) = Sum/(den+tau);
      }
    }  
  }
  return(out);
}

double mvnorm_lpdf_inv_det(arma::vec x,
                           arma::vec mean,
                           arma::mat invSigma,
                           double rootdet){
  int xdim = x.n_rows;
  arma::mat out;
  double constants = -(static_cast<double>(xdim)/2.0) * log2pi;
  arma::vec A = x - mean;
  out  = constants - 0.5 * A.t()*invSigma*A + rootdet;
  return(arma::as_scalar(out));
}

// [[Rcpp::export]]
Rcpp::List simZ(const arma::mat & z_pre ,const arma::mat & mu,
                const arma::cube & invSigma, const int & L,
                const arma::mat & w){
  // Rprintf("Start ");
  int N = z_pre.n_rows;
  arma::mat z = z_pre;
  double ll = 0;
  arma::vec rootdet(L);
  for(int l=0; l<L; l++){
    rootdet(l) = std::log(det(invSigma.slice(l)))/2.0;
  }
  for(int n=0; n<N; n++){
    int ind = as_scalar(arma::find(w.row(n)==1));
    arma::vec tmpz = rtmvnorm_gibbs(mu.col(ind),invSigma.slice(ind),z_pre.row(n).t());
    z.row(n) = tmpz.t();
    ll += mvnorm_lpdf_inv_det(tmpz,mu.col(ind),invSigma.slice(ind),rootdet(ind));
  }
  return List::create(z,ll);
}

// [[Rcpp::export]]
arma::mat simW(arma::mat Y, arma::mat pi, arma::mat mu, arma::cube invSigma, 
               int L, arma::mat minmax_id, int T){
  int N = Y.n_rows;
  arma::rowvec lp(L);
  arma::mat W(N,L);
  arma::vec rootdet(L);
  for(int l=0; l<L; l++){
    rootdet(l) = std::log(det(invSigma.slice(l)))/2.0;
  }
  for(int t=0;t<T;t++){
    int start = minmax_id(0,t);
    int end = minmax_id(1,t)+1;
    for(int n=start; n<end; n++){
      for(int l=0; l<L; l++){
        lp(l) = mvnorm_lpdf_inv_det(Y.row(n).t(),mu.col(l),invSigma.slice(l),rootdet(l))+
          std::log(pi(t,l)); 
      }
      W.row(n) = rcate(softmax(lp)); //Gibbs sampling
    }
  }
  return W;
}

// [[Rcpp::export]]
arma::mat calc_E(arma::mat Y, arma::mat pi, arma::mat mu, arma::cube invSigma, 
                 int L, arma::mat minmax_id, int T){
  
  int N = Y.n_rows;
  arma::rowvec lp(L);
  arma::mat E(N,L);
  arma::vec rootdet(L);
  for(int l=0; l<L; l++){
    rootdet(l) = std::log(det(invSigma.slice(l)))/2.0;
  }
  for(int t=0;t<T;t++){
    int start = minmax_id(0,t);
    int end = minmax_id(1,t)+1;
    for(int n=start; n<end; n++){
      for(int l=0; l<L; l++){
        lp(l) = mvnorm_lpdf_inv_det(Y.row(n).t(),mu.col(l),invSigma.slice(l),rootdet(l))+
          std::log(pi(t,l)); 
      }
      E.row(n) = softmax(lp); 
    }
  }
  return E;
}

// [[Rcpp::export]]
double calc_weight(arma::rowvec x, arma::uvec fix_id){
  
  //Rprintf(".");
  arma::uvec id = arma::find(fix_id == 1);
  double fix_prob = arma::sum(x(id));
  double out = 1-fix_prob;
  return(out);
}

// [[Rcpp::export]]
arma::uvec combine_list(Rcpp::List x){
  
  int list_size = x.length();
  arma::uvec vec_size(list_size);
  for(int i=0; i<list_size; i++){
    arma::uvec tmpvec = x[i];
    vec_size(i) = tmpvec.n_elem;
  }
  arma::uvec cumsum = arma::cumsum(vec_size);
  int total_elem = cumsum(list_size-1);
  arma::uvec out(total_elem);
  for(int i=0; i<list_size; i++){
    arma::uvec tmpvec = x[i];
    for(int j=0; j<tmpvec.n_elem; j++){
      if(i==0){
        out(j) = tmpvec(j);
      } else{
        out(j + cumsum(i-1)) = tmpvec(j);
      }
    }
  }
  return(out);
}

// [[Rcpp::export]]
arma::uvec rsample(arma::uvec x,int N,arma::vec prob){
  
  int M = x.n_elem;
  arma::uvec out(N); 
  arma::vec tmp_prob = prob;
  
  for(int n=0; n<N; n++){
    arma::vec cump = arma::cumsum(tmp_prob);
    double U = R::runif(0,1);
    int id;
    if(U<=cump(0)){
      id = 0;
      out(n) = x(id);
    }else{
      for(int m=1; m<M; m++){
        if(cump(m-1)<U & U<=cump(m)){
          id = m;
          out(n) = x(id);
        }
      }
    }
    tmp_prob(id) = 0;
    double sum = arma::sum(tmp_prob);
    tmp_prob = tmp_prob/sum;
  }
  return(out);
}

// [[Rcpp::export]]
arma::uvec weighted_iterative_sampling(arma::mat Y,int L,int D, arma::mat Wini, double xi, arma::mat pi,   
                                           const arma::mat mu, const arma::cube Sigma, 
                                           arma::rowvec t_id, arma::uvec fix_id, int wis_iter){
  
  arma::cube invSigma = Sigma;
  if(Sigma.has_nan()){
    Rprintf("nan");
  }
  for(int l=0; l<L; l++){
    invSigma.slice(l) = arma::inv(Sigma.slice(l));
  }
  arma::rowvec unique_time = arma::unique(t_id);
  int T = unique_time.n_cols;
  arma::mat minmax_id(2,T);
  for(int t=0; t<T; t++){
    arma::uvec id = arma::find(t_id == unique_time(t));
    minmax_id(0,t) = arma::min(id);
    minmax_id(1,t) = arma::max(id);
  }
  
  arma::mat W = Wini;
  arma::mat Z = Y;
  for(int i=0; i<wis_iter; i++){
    //Draw latent cluster
    W = simW(Z,pi,mu,invSigma,L,minmax_id,T);
    
    //Draw unobserved data
    List LZ(2);
    LZ = simZ(Z,mu,invSigma,L,W);
    arma::mat tmpZ = LZ[0];
    
    Z = tmpZ;
  }
  
  //Calculate posterior
  arma::mat E = calc_E(Z,pi,mu,invSigma,L,minmax_id,T);
  Rcpp::List out(T);
  for(int t=0;t<T;t++){
    int start = minmax_id(0,t);
    int end = minmax_id(1,t);
    int N = end-start+1;
    arma::uvec samples(N);
    arma::vec weight(N);weight.fill(0);
    arma::vec prob(N);
    for(int n=0; n<N; n++){
      samples(n) = start + n;
      arma::mat tmpE = E.submat(start,0,end,L-1);
      weight(n) = calc_weight(tmpE.row(n),fix_id);
    }
    arma::uvec zero = arma::find(weight == 0);
    double num_zero = zero.n_elem;
    weight(zero).fill(1/num_zero);
    
    double sum = arma::sum(weight);
    prob = weight/sum;
    
    arma::mat sample_size(1,1);
    sample_size(0) = N*xi; 
    int S = arma::as_scalar(arma::round(sample_size));
    arma::uvec id(S);
    id = rsample(samples,S,prob);
    out[t] = id;
  }
  arma::uvec res = combine_list(out);
  return(res);
}

// [[Rcpp::export]]
arma::rowvec pi_update(arma::mat W, arma::rowvec pre_pi, arma::vec minmax_id,
                       double alpha, int L, int T){
  
  int start = minmax_id(0);
  int end = minmax_id(1);
  arma::rowvec pi(L);
  arma::rowvec nl_t(L);
  for(int l=0; l<L; l++){
    nl_t(l) = arma::sum(W(arma::span(start,end),l));
  }
  int Nt = arma::sum(nl_t);
  for(int l=0; l<L; l++){
    pi(l) = (nl_t(l)+alpha*pre_pi(l))/(Nt+alpha);
  }
  return pi;
}

// [[Rcpp::export]]
double alpha_update(arma::mat W, double pre_alpha, arma::rowvec pre_pi, 
                    int L,arma::vec minmax_id){
  
  int start = minmax_id(0);
  int end = minmax_id(1);
  int Nt = end-start+1;
  double alpha;
  double tmp;
  arma::vec nl_t(L);
  
  for(int l=0;l<L;l++){
    nl_t(l) = arma::sum(W(arma::span(start,end),l));
  }
  tmp = 0;
  for(int l=0;l<L;l++){
    tmp += pre_pi(l)*(R::digamma(nl_t(l)+pre_alpha*pre_pi(l))-R::digamma(pre_alpha*pre_pi(l)));
  } 
  alpha = pre_alpha*tmp/(R::digamma(Nt+pre_alpha)-R::digamma(pre_alpha));
  return alpha;
}

// [[Rcpp::export]]
arma::cube sigma_update(arma::mat mu, Rcpp::List Y, Rcpp::List W, int L, int D,
                           double nu, double tau, arma::mat Lambda){
  int K = mu.n_rows;
  arma::cube Sigma(K,K,L);
  Sigma.fill(0);
  arma::vec nl(L);
  nl.fill(0);
  for(int l=0; l<L; l++){
    for(int d=0; d<D; d++){
      arma::mat tmpY = Y[d];
      arma::mat tmpW = W[d];
      nl(l) += arma::sum(tmpW.col(l));
      int N = tmpY.n_rows;
      for(int n=0; n<N; n++){
        arma::vec d = tmpY.row(n).t()-mu.col(l);
        Sigma.slice(l) += tmpW(n,l)*d*d.t();
      }
    }
  }
  for(int l=0; l<L; l++){
    Sigma.slice(l) = (Lambda + Sigma.slice(l) + tau*mu.col(l)*mu.col(l).t())/(nl(l) + nu - K - 1);
  }
  return Sigma;
}

// [[Rcpp::export]]
arma::cube sigma_update_fix(arma::mat mu, Rcpp::List Y, Rcpp::List W, arma::cube preSigma,
                               int L, int D, double nu, double tau, arma::mat Lambda, arma::uvec fix_id){
  int K = mu.n_rows;
  arma::cube Sigma(K,K,L);
  Sigma.fill(0);
  arma::vec nl(L);
  nl.fill(0);
  for(int l=0; l<L; l++){
    if(fix_id(l) == 1){
      Sigma.slice(l) = preSigma.slice(l);
    } else{
      for(int d=0; d<D; d++){
        arma::mat tmpY = Y[d];
        arma::mat tmpW = W[d];
        nl(l) += arma::sum(tmpW.col(l));
        int N = tmpY.n_rows;
        for(int n=0; n<N; n++){
          arma::vec d = tmpY.row(n).t()-mu.col(l);
          Sigma.slice(l) += tmpW(n,l)*d*d.t();
        }
      }
    }
  }
  for(int l=0; l<L; l++){
    if(fix_id(l) == 1){
      continue;
    }else{
      Sigma.slice(l) = (Lambda + Sigma.slice(l) + tau*mu.col(l)*mu.col(l).t())/(nl(l) + nu - K - 1);
    }
  }
  return Sigma;
}

// [[Rcpp::export]]
arma::vec weighted_means(int D, int L, int T, Rcpp::List pi, Rcpp::List t_id){
  
  int N = 0;
  arma::rowvec out(L);
  out.fill(0);
  
  arma::rowvec x = t_id[0];
  arma::rowvec unique_time = arma::unique(x);
  
  for(int d=0; d<D; d++){
    arma::rowvec tmpt_id = t_id[d];
    arma::mat tmp_pi = pi[d];
    for(int t=0; t<T; t++){
      arma::uvec id = arma::find(tmpt_id == unique_time(t));
      int start = arma::min(id);
      int end = arma::max(id);
      int tmpN = end - start + 1;
      
      out += tmp_pi.row(t) * tmpN;
      N += tmpN;
    }
  }
  out = out/N;
  arma::vec out1 = out.t();
  return(out1);
}

// [[Rcpp::export]]
arma::uvec find_popular(int P, Rcpp::List all_sample_pi, arma::uvec fix_id, Rcpp::List t_id){
  
  int L = fix_id.n_elem;
  int D = all_sample_pi.length();
  arma::mat junk = all_sample_pi[0];
  int T = junk.n_rows;
  arma::vec find_popular(L);
  
  find_popular = weighted_means(D,L,T,all_sample_pi,t_id);
  
  arma::uvec id = arma::find(fix_id == 1);
  arma::vec zero(id.n_elem); zero.fill(0); find_popular.rows(id) = zero;
  arma::uvec sort = arma::sort_index(find_popular,"descend");
  arma::uvec res = fix_id;
  for(int p=0; p<P; p++){
    res(sort(p)) = 1;
  }
  return(res);
}

// [[Rcpp::export]]
Rcpp::List fix(int P, Rcpp::List all_sample_pi, Rcpp::List pi, arma::uvec fix_id, Rcpp::List t_id){
  
  
  Rcpp::List out = all_sample_pi;
  int D = out.length();
  //arma::uvec res = fix_id;
  arma::mat tmppi = pi[0];
  int T = tmppi.n_rows;
  
  arma::uvec id = arma::find(fix_id == 1);
  arma::uvec nonfix_id = arma::find(fix_id != 1);
  int F = id.n_elem;
  int J = nonfix_id.n_elem;
  
  for(int d=0; d<D; d++){
    arma::mat all_sample_fix_pi(T,F);
    arma::mat fix_pi(T,F);
    arma::vec sum_all_sample_fix_pi(T);
    arma::vec sum_nonfix_pi(T);
    
    arma::mat tmp_all_sample_pi = all_sample_pi[d];
    arma::mat tmp_pi = pi[d];
    for(int f=0; f<F; f++){
      all_sample_fix_pi.col(f) = tmp_all_sample_pi.col(id(f));
      fix_pi.col(f) = tmp_pi.col(id(f));
    }
    for(int t=0; t<T; t++){
      sum_all_sample_fix_pi(t) = arma::sum(all_sample_fix_pi.row(t));
      sum_nonfix_pi(t) = 1-arma::sum(fix_pi.row(t));
      tmp_pi.row(t) = tmp_pi.row(t)*(1-sum_all_sample_fix_pi(t))/sum_nonfix_pi(t);
    }
    arma::mat tmp_out = out[d];
    for(int j=0; j<J; j++){
      tmp_out.col(nonfix_id(j)) = tmp_pi.col(nonfix_id(j));
    }
    out[d] = tmp_out;
  }
  
  arma::uvec res = find_popular(P, all_sample_pi, fix_id, t_id);
  return List::create(out,res);
}

// [[Rcpp::export]]
Rcpp::List stochasticEM_fix(Rcpp::List & Y, const int & L, const int & D, Rcpp::List Wini, Rcpp::List piini, arma::rowvec alphaini, 
                    const arma::mat & muini, const arma::cube & Sigmaini,const double & tau, const double & nu, 
                    const arma::mat & Lambda, const int & num_iter, Rcpp::List t_idini, Rcpp::List id, arma::uvec fix_id){
  
  Rcpp::List pi(D);
  Rcpp::List alpha(D);
  Rcpp::List Z(D);
  Rcpp::List W(D);
  Rcpp::List t_id(D);
  for(int d=0;d<D;d++){
    arma::uvec tmpid = id[d];
    pi[d] = piini[d];
    alpha[d] = alphaini;
    arma::mat tmpY = Y[d]; tmpY = tmpY.rows(tmpid);
    Z[d] = tmpY;
    W[d] = Wini[d];
    arma::mat tmpW = W[d]; tmpW = tmpW.rows(tmpid); W[d] = tmpW;
    arma::rowvec tmpt_id = t_idini[d]; tmpt_id = tmpt_id.cols(tmpid); t_id[d] = tmpt_id;
  }
  
  int K = muini.n_rows;
  arma::mat mu = muini;
  arma::cube Sigma = Sigmaini;
  arma::mat llhist(num_iter-1,D);
  
  arma::uvec non_fix = arma::find(fix_id != 1);
  int num_non_fix = non_fix.n_elem;
  
  arma::vec one(K); one.fill(1);
  arma::mat iden_mat = arma::diagmat(one);
  for(int i=0; i<num_non_fix; i++){
    arma::mat tmpZ = Z[0];
    mu.col(non_fix(i)) = tmpZ.row(i).t();
    Sigma.slice(non_fix(i)) = iden_mat;
  }

  for(int h=1;h<num_iter;h++){
    for(int d=0;d<D;d++){
      arma::mat pre_Z = Z[d];
      arma::mat tmpW = W[d];
      arma::mat tmppi = pi[d];
      arma::rowvec tmpalpha = alpha[d];
      arma::rowvec tmpt_id = t_id[d];
      arma::rowvec unique_time = arma::unique(tmpt_id);
      int T = unique_time.n_cols;
      arma::mat minmax_id(2,T);
      for(int t=0; t<T; t++){
        arma::uvec id = arma::find(tmpt_id == unique_time(t));
        minmax_id(0,t) = arma::min(id);
        minmax_id(1,t) = arma::max(id);
      }
      if(Sigma.has_nan()){
        break;
      }
      arma::cube invSigma = Sigma;
      for(int l=0; l<L; l++){
        invSigma.slice(l) = arma::inv(Sigma.slice(l));
      }
      
      //Gibbs sampling of latent variable
      tmpW = simW(pre_Z,tmppi,mu,invSigma,L,minmax_id,T); 
      
      //Gibbs sampling of z
      List LZ(2);
      LZ = simZ(pre_Z,mu,invSigma,L,tmpW);
      Z[d] = LZ[0];
      arma::mat tmpZ = Z[d];
      
      llhist(h-1,d) = LZ[1];
      
      //Caluclate alpha and pi
      for(int t=0;t<T;t++){
        if(t==0){
          tmpalpha(0) = 1;
          int start = minmax_id(0,t);
          int end = minmax_id(1,t);
          int Nt = end-start+1;
          arma::rowvec nl(L);
          nl.fill(0);
          for(int l=0;l<L;l++){
            nl(l) = arma::sum(tmpW(arma::span(start,end),l));
          }
          tmppi.row(0) = (nl+tmpalpha(0))/(Nt+L*tmpalpha(0));
        } else{
          tmpalpha(t) = alpha_update(tmpW,tmpalpha(t),tmppi.row(t-1),L,minmax_id.col(t));
          tmppi.row(t) = pi_update(tmpW,tmppi.row(t-1),minmax_id.col(t),tmpalpha(t),L,T);
        }
      }
      pi[d] = tmppi;
      alpha[d] = tmpalpha;
      W[d] = tmpW;
    }
    mu = weighted_colMeans_fix(Z,W,mu,tau,L,D,fix_id);
    Sigma = sigma_update_fix(mu,Z,W,Sigma,L,D,nu,tau,Lambda,fix_id);
  }
  return Rcpp::List::create(Rcpp::Named("pi")=pi,_["alpha"]=alpha,_["Sigma"]=Sigma,
                            _["mu"]=mu,_["W"]=W,_["loglik"]=llhist);
}

// [[Rcpp::export]]
Rcpp::List stochasticEM(Rcpp::List & Y, const int & L, const int & D, Rcpp::List Wini, Rcpp::List piini, arma::rowvec alphaini, 
                         const arma::mat & muini, const arma::cube & Sigmaini,const double & tau, const double & nu, 
                         const arma::mat & Lambda, const int & num_iter, Rcpp::List t_id){
  
  Rcpp::List pi(D);
  Rcpp::List alpha(D);
  Rcpp::List Z(D);
  Rcpp::List W(D);
  for(int d=0;d<D;d++){
    pi[d] = piini[d];
    alpha[d] = alphaini;
    Z[d] = Y[d];
    W[d] = Wini[d];
  }
  arma::mat mu = muini;
  arma::cube Sigma = Sigmaini;
  arma::mat llhist(num_iter-1,D);
  for(int h=1;h<num_iter;h++){
    for(int d=0;d<D;d++){
      arma::mat tmpY = Y[d];
      arma::mat pre_Z = Z[d];
      arma::mat tmpW = W[d];
      arma::mat tmppi = pi[d];
      arma::rowvec tmpalpha = alpha[d];
      arma::rowvec tmpt_id = t_id[d];
      arma::rowvec unique_time = arma::unique(tmpt_id);
      int T = unique_time.n_cols;
      arma::mat minmax_id(2,T);
      for(int t=0; t<T; t++){
        arma::uvec id = arma::find(tmpt_id == unique_time(t));
        minmax_id(0,t) = arma::min(id);
        minmax_id(1,t) = arma::max(id);
      }
      if(Sigma.has_nan()){
        break;
      }
      arma::cube invSigma = Sigma;
      for(int l=0; l<L; l++){
        invSigma.slice(l) = arma::inv(Sigma.slice(l));
      }
      
      //Gibbs sampling of latent variable
      tmpW = simW(pre_Z,tmppi,mu,invSigma,L,minmax_id,T); 
      
      //Gibbs sampling of z
      List LZ(2);
      LZ = simZ(pre_Z,mu,invSigma,L,tmpW);
      Z[d] = LZ[0];
      arma::mat tmpZ = Z[d];
      
      llhist(h-1,d) = LZ[1];
      
      //Caluclate alpha and pi
      for(int t=0;t<T;t++){
        if(t==0){
          tmpalpha(0) = 1;
          int start = minmax_id(0,t);
          int end = minmax_id(1,t);
          int Nt = end-start+1;
          arma::rowvec nl(L);
          nl.fill(0);
          for(int l=0;l<L;l++){
            nl(l) = arma::sum(tmpW(arma::span(start,end),l));
          }
          tmppi.row(0) = (nl+tmpalpha(0))/(Nt+L*tmpalpha(0));
        } else{
          tmpalpha(t) = alpha_update(tmpW,tmpalpha(t),tmppi.row(t-1),L,minmax_id.col(t));
          tmppi.row(t) = pi_update(tmpW,tmppi.row(t-1),minmax_id.col(t),tmpalpha(t),L,T);
        }
      }
      pi[d] = tmppi;
      alpha[d] = tmpalpha;
      W[d] = tmpW;
    }
    mu = weighted_colMeans(Z,W,tau,L,D);
    Sigma = sigma_update(mu,Z,W,L,D,nu,tau,Lambda);
  }
  return Rcpp::List::create(Rcpp::Named("Z")=Z,_["pi"]=pi,_["alpha"]=alpha,_["Sigma"]=Sigma,
                            _["mu"]=mu,_["W"]=W,_["loglik"]=llhist);
}

// [[Rcpp::export]]
Rcpp::List cybertrack2(Rcpp::List & Y, const int & L, const int & D, int & P, Rcpp::List Wini, Rcpp::List piini, arma::rowvec alphaini, 
                    const arma::mat & muini, const arma::cube & Sigmaini,const double & tau, const double & nu, const double & xi,
                    const arma::mat & Lambda, const int & num_iter, int num_iter_refine, int wis_iter, Rcpp::List t_id){
  
  Rprintf("start ");
  Rcpp::List out1; Rcpp::List out2;
  Rcpp::List pi(D); pi.fill(piini[0]);
  Rcpp::List all_sample_pi = pi;
  Rcpp::List alpha(D); alpha.fill(alphaini);
  Rcpp::List W = Wini;
  arma::mat mu = muini;
  arma::cube Sigma = Sigmaini;
  
  int fix_number;
  arma::uvec fix_id(L); fix_id.fill(0);
  Rcpp::List id(D);
  
  do{
    for(int d=0; d<D; d++){
      arma::uvec tmp_id = weighted_iterative_sampling(Y[d],L,D,W[d],xi,pi[d],mu,Sigma,t_id[d],fix_id,wis_iter);
      id[d] = tmp_id;
    }
    
    out1 = stochasticEM_fix(Y,L,D,W,piini,alphaini,mu,Sigma,tau,nu,Lambda,num_iter,t_id,id,fix_id);
    Rcpp::List subsample_pi = out1[0];
    arma::mat tmp_mu = out1[3]; mu = tmp_mu;
    arma::cube tmp_Sigma = out1[2]; Sigma = tmp_Sigma;
    
    out2 = fix(P,all_sample_pi,subsample_pi,fix_id,t_id);
    pi = all_sample_pi = out2[0];
    arma::uvec tmp_fix_id = out2[1]; fix_id = tmp_fix_id;
    arma::uvec fix_vec = arma::find(fix_id == 1);
    fix_number = fix_vec.n_elem;
  } while (fix_number < L);
  
  Rcpp::List out;
  out = stochasticEM(Y,L,D,Wini,pi,alphaini,mu,Sigma,tau,nu,Lambda,num_iter_refine,t_id);
  Rprintf("end");
  return(out);
}
