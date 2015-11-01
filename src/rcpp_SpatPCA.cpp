// includes from the plugin
// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppParallel.h>
#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <iostream>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace std;
using namespace arma;

arma::mat cubicmatrix(const vec z1){
  
  //vec h = diff(z1);
  vec h(z1.n_elem-1);
  for(unsigned i = 0; i < z1.n_elem-1; i++)
    h[i] = z1[i+1]-z1[i];
  arma::mat Q, R;
  int p = z1.n_elem;
  Q.zeros(p-2,p);
  R.zeros(p-2,p-2);
  
  for(unsigned int  j = 0; j < p-2; ++j){
    Q(j,j) = 1/h[j];
    Q(j,j+1) = -1/h[j]-1/h[j+1];
    Q(j,j+2) = 1/h[j+1];
    R(j,j) = (h[j]+h[j+1])/3;
  }
  for(unsigned int  j = 0; j < p-3; ++j)
    R(j,j+1) = R(j+1,j) = h[j+1]/6;
  
  return(Q.t()*solve(R,Q));
}

using namespace arma;
using namespace Rcpp;
using namespace std;

struct tpm: public RcppParallel::Worker {
  const mat& P;
  mat& L;  
  int p;
  int d;
  tpm(const mat &P, mat& L, int p, int d) : P(P), L(L), p(p), d(d){}
  void operator()(std::size_t begin, std::size_t end) {
    for(std::size_t i = begin; i < end; i++){
      for(unsigned int j = 0; j < p; ++j){
        if(j >i){
          if(d==2){  
            double r  = sqrt(pow(P(i,0)-P(j,0),2)+(pow(P(i,1)-P(j,1),2)));
            L(i,j) = r*r*log(r)/(8.0*arma::datum::pi);
          }
          else{
            double r  = sqrt(pow(P(i,0)-P(j,0),2));
            L(i,j) = sqrt(2)/(16*sqrt(arma::datum::pi))*pow(r,3);
          }  
        }
      }  
      
      L(i,p) = 1;
      for(unsigned int k = 0; k < d; ++k){
        L(i,p+k+1) = P(i,k);
      }
    }
  }
};

arma::mat tpmatrix(const arma::mat P){
   arma::mat L, Lp;
  int p = P.n_rows, d = P.n_cols;

  L.zeros(p+d+1, p+d+1);
  tpm tpm(P,L,p,d);
  parallelFor(0, p,tpm);
  L = L + L.t();
  Lp = inv(L);   
  Lp.shed_cols(p,p+2);
  Lp.shed_rows(p,p+2);
  L.shed_cols(p,p+2);
  L.shed_rows(p,p+2);
  arma::mat result = Lp.t()*(L*Lp);
  return(result);
}

// [[Rcpp::export]]
arma::mat tpm2(const arma::mat z,const arma::mat P, const arma::mat Phi){
  arma::mat L;//, Lp;
  int p = P.n_rows, d = P.n_cols, K = Phi.n_cols;
  L.zeros(p+d+1, p+d+1);
  tpm tpm(P,L,p,d);
  parallelFor(0, p,tpm);
  L = L + L.t();
 // Lp = inv(L);  
  arma::mat Phi_star, para(p+d+1, K);
  Phi_star.zeros(p+d+1, K);
  Phi_star.rows(0,p-1) = Phi;
  para = solve(L, Phi_star);
  int pnew = z.n_rows;
  arma::mat eigen_fn(pnew, K);
  double psum, r;
  for(unsigned newi = 0; newi < pnew ; newi++){
    for(unsigned i = 0; i < K; i++){
      psum = 0;
      for(unsigned j = 0; j < p; j++){
         if(d==2){
            r  = sqrt(pow(z(newi,0)-P(j,0),2)+(pow(z(newi,1)-P(j,1),2)));
            if(r!=0)
             psum += para(j,i)* r*r*log(r)/(8.0*arma::datum::pi);
         }
         else{
           r = norm(z.row(newi)-P.row(j),'f');
           if(r!=0)
             psum += para(j,i)*((sqrt(2)/(16*sqrt(arma::datum::pi)))*pow(r,3));
         }
      }
      if(d==1)
        eigen_fn(newi,i) = psum + para(p+1,i)*z(newi,0) + para(p,i);
      else
        eigen_fn(newi,i) = psum + para(p+1,i)*z(newi,0) + para(p+2,i)*z(newi,1) + para(p,i); 
    }
  }
  return(eigen_fn);
}


// user includes
using namespace arma;
using namespace Rcpp;
using namespace std;
void spatpcacore(const mat Y, mat& Phi, mat& R,  mat& C,  mat& Lambda1, mat& Lambda2, const mat Omega,const double tau1, const double tau2, double rho,const double rhoincre,const int maxit,const double tol){
  
  int p = Phi.n_rows;
  int K = Phi.n_cols;
  int iter = 0;
  arma::mat Ip, Sigtau1, temp, tau2one, zero, one;
  arma::vec er(4);
  Ip.eye(p,p);
  zero.zeros(p,K);
  one.ones(p,K);
  tau2one = tau2*one;
  Sigtau1 = tau1*Omega - arma::trans(Y)*Y;
  
  arma::mat U;
  arma::vec S;
  arma::mat V;
  arma::mat Phiold = Phi;
  arma::mat Rold = R;
  arma::mat Cold = C;
  arma::mat Lambda1old = Lambda1;
  arma::mat Lambda2old = Lambda2;
  arma::mat tempinv = arma::inv_sympd(Sigtau1 + (rho*Ip));
  for (iter = 0; iter < maxit; iter++){
    // Phi =0.5*arma::solve(Sigtau1 + rho*Ip,(rho*(Rold+Cold)-Lambda1old-Lambda2old));
    Phi =0.5*tempinv*(rho*(Rold+Cold)-(Lambda1old+Lambda2old));
    R = arma::sign(((Lambda1old/rho)+Phi))%arma::max(zero, arma::abs(((Lambda1old/rho)+Phi)) - tau2one/rho);
    temp = Phi+Lambda2old/rho;
    arma::svd_econ(U, S, V,temp);
    C = U.cols(0,V.n_cols-1)*V.t();
    
    Lambda1 = Lambda1old +rho*(Phi-R);
    Lambda2 = Lambda2old +rho*(Phi-C);
    
    
    er[1] = arma::norm(Phi-R,"fro")/sqrt(p/1.0);
    er[2] = arma::norm((R-Rold),"fro")/sqrt(p/1.0);
    er[3] = arma::norm(Phi-C,"fro")/sqrt(p/1.0);
    er[4] = arma::norm((C-Cold),"fro")/sqrt(p/1.0);
    
    if(max(er) <= tol)
      break;
    Phiold = Phi;
    Rold = R;
    Cold = C;
    Lambda1old = Lambda1;
    Lambda2old = Lambda2;
  }
  
  iter++;
  if(iter == maxit)
    Rcpp::Rcout<<"Not converge at tau1="<<tau1<<" tau2="<<tau2<<"\n"<<std::endl;
}
void spatpcacore2(const mat YY, mat& Phi,  mat& C, mat& Lambda2, const mat Omega,const double tau1, const double rho,const int maxit,const double tol){
  
  int p = Phi.n_rows;
  int K = Phi.n_cols;
  int iter = 0;
  arma::mat Ip, Sigtau1, temp;
  arma::vec er(2);
  Ip.eye(p,p);
  Sigtau1 = tau1*Omega - YY;
  
  arma::mat U;
  arma::vec S;
  arma::mat V;
  arma::mat Cold = C;
  arma::mat Lambda2old = Lambda2;
  arma::mat tempinv = arma::inv_sympd(2*Sigtau1 + rho*Ip);
  for (iter = 0; iter < maxit; iter++){
    Phi = tempinv*((rho*Cold)-Lambda2old);
    temp = Phi + (Lambda2old/rho);
    arma::svd_econ(U, S, V,temp);
    C = U.cols(0,V.n_cols-1)*V.t();
    
    Lambda2 = Lambda2old +rho*(Phi-C);
    
    er[0] = arma::norm(Phi-C,"fro")/sqrt(p*K/1.0);
    er[1] = arma::norm((C-Cold),"fro")/sqrt(p*K/1.0);
    
    if(max(er) <= tol)
      break;
    Cold = C;
    Lambda2old = Lambda2;
  }
  
  iter++;
  if(iter == maxit)
    Rcpp::Rcout<<"Not converge at tau1="<<tau1<<"\n"<<std::endl;
}
using namespace arma;
using namespace Rcpp;
using namespace std;

arma::mat spatpcacore2p(const arma::mat YY, arma::mat& C, arma::mat& Lambda2, const arma::mat Omega,const double tau1, const double rho,const int maxit,const double tol){
  
  int p = C.n_rows;
  int K = C.n_cols;
  int iter = 0;
  arma::mat Ip, Sigtau1, temp;
  arma::vec er(2);
  Ip.eye(p,p);
  Sigtau1 = tau1*Omega - YY;
  
  arma::mat U;
  arma::vec S;
  arma::mat V;
  arma::mat Phi;
  arma::mat Cold = C;
  arma::mat Lambda2old = Lambda2;
  arma::mat tempinv = arma::inv_sympd(2*Sigtau1 + rho*Ip);
  for (iter = 0; iter < maxit; iter++){
    Phi = tempinv*((rho*Cold)-Lambda2old);
    temp = Phi + (Lambda2old/rho);
    arma::svd_econ(U, S, V,temp);
    C = U.cols(0,V.n_cols-1)*V.t();
    
    Lambda2 = Lambda2old +rho*(Phi-C);
    
    er[0] = arma::norm(Phi-C,"fro")/sqrt(p*K/1.0);
    er[1] = arma::norm((C-Cold),"fro")/sqrt(p*K/1.0);
    
    if(max(er) <= tol)
      break;
    Cold = C;
    Lambda2old = Lambda2;
  }
  
  iter++;
  if(iter == maxit)
    Rcpp::Rcout<<"Not converge at tau1="<<tau1<<"\n"<<std::endl;
  // cout<<"iter"<<iter<<endl;
  return(Phi);
}

void spatpcacore3(const arma::mat tempinv, arma::mat& Phi, arma::mat& R,  arma::mat& C,  arma::mat& Lambda1, arma::mat& Lambda2, const double tau2, double rho,const int maxit,const double tol){
  
  int p = Phi.n_rows;
  int K = Phi.n_cols;
  int iter = 0;
  arma::mat temp, tau2onerho, zero, one;
  arma::vec er(4);
  
  zero.zeros(p,K);
  one.ones(p,K);
  tau2onerho = tau2*one/rho;
  //Sigtau1 = tau1*Omega - arma::trans(Y)*Y;
  
  arma::mat U;
  arma::vec S;
  arma::mat V;
  arma::mat Phiold = Phi;
  arma::mat Rold = R;
  arma::mat Cold = C;
  arma::mat Lambda1old = Lambda1;
  arma::mat Lambda2old = Lambda2;
  //  arma::mat tempinv = arma::inv_sympd(Sigtau1 + (rho*Ip));
  for (iter = 0; iter < maxit; iter++){
    // Phi =0.5*arma::solve(Sigtau1 + rho*Ip,(rho*(Rold+Cold)-Lambda1old-Lambda2old));
    Phi =0.5*tempinv*(rho*(Rold+Cold)-(Lambda1old+Lambda2old));
    R = arma::sign(((Lambda1old/rho)+Phi))%arma::max(zero, arma::abs(((Lambda1old/rho)+Phi)) - tau2onerho);
    temp = Phi+Lambda2old/rho;
    arma::svd_econ(U, S, V,temp);
    C = U.cols(0,V.n_cols-1)*V.t();
    
    Lambda1 = Lambda1old +rho*(Phi-R);
    Lambda2 = Lambda2old +rho*(Phi-C);
    
    
    er[1] = arma::norm(Phi-R,"fro")/sqrt(p/1.0);
    er[2] = arma::norm((R-Rold),"fro")/sqrt(p/1.0);
    er[3] = arma::norm(Phi-C,"fro")/sqrt(p/1.0);
    er[4] = arma::norm((C-Cold),"fro")/sqrt(p/1.0);
    
    if(max(er) <= tol)
      break;
    Phiold = Phi;
    Rold = R;
    Cold = C;
    Lambda1old = Lambda1;
    Lambda2old = Lambda2;
  }
  
  iter++;
  if(iter == maxit)
    Rcpp::Rcout<<"Not converge at tau2="<<tau2<<"\n"<<std::endl;
}

using namespace arma;
using namespace Rcpp;
using namespace std;


struct spatpcacv_p: public RcppParallel::Worker {
  const mat& Y;
  int K;
  const mat& Omega;
  const vec& tau1;
  const vec& nk;
  int maxit;
  double tol;
  mat& output;
  
  spatpcacv_p(const mat& Y, int K, const mat& Omega, const vec& tau1, const vec& nk, int maxit, double tol, mat& output) : Y(Y), K(K), Omega(Omega), tau1(tau1), nk(nk),  maxit(maxit),tol(tol),output(output) {}
  void operator()(std::size_t begin, std::size_t end) {
    for(std::size_t k = begin; k < end; k++){
      
      arma::mat UPhi, Phiold, Phi, C, Lambda2;
      vec SPhi;
      mat Ytrain = Y.rows(arma::find(nk!=(k+1)));
      mat Yvalid = Y.rows(arma::find(nk==(k+1)));
      
      arma::svd_econ(UPhi, SPhi, Phiold, Ytrain, "right");
      
      double rho = 10*pow(SPhi[0],2.0);
      Phi = Phiold.cols(0,K-1);
      C = Phi;
      Lambda2 = Phi*(diagmat(rho -1/(rho-2*pow(SPhi.subvec(0,K-1),2))));
      output(k,0) = arma::norm(Yvalid*Phi,"fro");
      mat YYtrain = Ytrain.t()*Ytrain;
      for(uword  i = 1; i < tau1.n_elem; i++){
        spatpcacore2(YYtrain,Phi,C,Lambda2, Omega, tau1[i], rho, maxit,tol);
        output(k,i) = arma::norm(Yvalid*Phi,"fro");//pow(arma::norm(Yvalid*(Ip-Phi*Phi.t()),"fro"),2.0); 
      }
    }
    
  }
};

struct spatpcacv_p2: public RcppParallel::Worker {
  const mat& Y;
  int K;
  const mat& tau1Omega;
  const vec& tau2;
  const vec& nk;
  int maxit;
  double tol;
  mat& output;
  
  
  spatpcacv_p2(const mat& Y, int K, const mat& tau1Omega, const vec& tau2, const vec& nk, int maxit, double tol, mat& output) : Y(Y), K(K), tau1Omega(tau1Omega), tau2(tau2), nk(nk),  maxit(maxit),tol(tol),output(output) {}
  void operator()(std::size_t begin, std::size_t end) {
    for(std::size_t k = begin; k < end; k++){
      arma::mat UPhi, Phiold, Phi, R, C, Lambda1, Lambda2,Ip;
      vec SPhi;
      mat Ytrain = Y.rows(arma::find(nk!=(k+1)));
      mat Yvalid = Y.rows(arma::find(nk==(k+1)));
      Ip.eye(Y.n_cols,Y.n_cols);
      arma::svd_econ(UPhi, SPhi, Phiold, Ytrain, "right");
      double rho = 10*pow(SPhi[0],2.0);
      Phi = Phiold.cols(0,K-1);
      R = C = Phi;
      Lambda1 = 0*Phi;
      Lambda2 = Phi*(diagmat(rho -1/(rho-2*pow(SPhi.subvec(0,K-1),2))));
      output(k,0) = arma::norm(Yvalid*Phi,"fro");
      mat tempinv = arma::inv_sympd((tau1Omega) - (arma::trans(Ytrain)*Ytrain) + (rho*Ip));
      for(uword  i = 1; i < tau2.n_elem; i++){
        spatpcacore3(tempinv,Phi,R,C,Lambda1,Lambda2, tau2[i], rho, maxit,tol);
        output(k,i) = arma::norm(Yvalid*Phi,"fro");//pow(arma::norm(Yvalid*(Ip-Phi*Phi.t()),"fro"),2.0); 
      }
    }
    
  }
};


using namespace Rcpp;
using namespace arma;
using namespace std;
// [[Rcpp::export]]

List spatpcacv_rcpp(NumericMatrix  sxyr, NumericMatrix Yr, int M, int K,  NumericVector  tau1r, NumericVector  tau2r,  NumericVector  nkr, int maxit, double tol, NumericVector  l2r) {
  int n = Yr.nrow(), p = Yr.ncol(), d = sxyr.ncol();
  arma::mat Y(Yr.begin(), n, p, false);
  arma::mat sxy(sxyr.begin(), p, d, false);
  colvec tau1(tau1r.begin(), tau1r.size(),false);
  colvec tau2(tau2r.begin(), tau2r.size(), false);
  colvec nk(nkr.begin(), nkr.size(), false);
  colvec l2(l2r.begin(), l2r.size(), false);
  arma::mat cv(M,tau1.n_elem), out;
  arma::mat Omega;
  double cvtau1, cvtau2;
  arma::mat YYest = Y.t()*Y;
  arma::mat UPhiest, Phiest2;
  vec SPhiest;
  arma::svd_econ(UPhiest, SPhiest, Phiest2, Y, "right");
  double rhoest = 10*pow(SPhiest[0],2.0);
  arma::mat Phiest = Phiest2.cols(0,K-1); 
  arma::mat Cest = Phiest;
  arma::mat  Lambda2est = Phiest*(diagmat(rhoest -1/(rhoest-2*pow(SPhiest.subvec(0,K-1),2))));
  if(d ==2)
    Omega = tpmatrix(sxy);
  else
    Omega = cubicmatrix(sxy);
  
  if(tau1.n_elem > 1){  
    spatpcacv_p spatpcacv_p(Y, K, Omega,  tau1, nk,  maxit, tol, cv);
    RcppParallel::parallelFor(0, M, spatpcacv_p);
    uword  index1;
    (sum(cv,0)).max(index1);
    cvtau1=tau1[index1];  
    if(index1 > 0)
      Phiest = spatpcacore2p(YYest,Cest,Lambda2est, Omega, cvtau1, rhoest, maxit,tol);
  }
  else{
    cvtau1=max(tau1); 
    Phiest = spatpcacore2p(YYest,Cest,Lambda2est, Omega, cvtau1, rhoest, maxit,tol);
  }
  
  if(tau2.n_elem > 1){
    arma::mat cv2(M,tau2.n_elem);
    uword  index2;
    arma::mat Omega2 = cvtau1*Omega;
    spatpcacv_p2 spatpcacv_p2(Y, K, Omega2, tau2, nk,  maxit, tol, cv2); 
    RcppParallel::parallelFor(0, M, spatpcacv_p2);
    //out = join_cols(-sum(cv,0)/M, -sum(cv2,0)/M);
    (sum(cv2,0)).max(index2);
    cvtau2 = tau2[index2];
    mat Ip;
    Ip.eye(Y.n_cols,Y.n_cols);
    mat tempinv = arma::inv_sympd((cvtau1*Omega) - YYest + (rhoest*Ip));
    mat Rest= Phiest;
    mat Lambda1est = 0*Phiest;
    for(uword  i = 0; i <= index2; i++){
      spatpcacore3(tempinv,Phiest,Rest,Cest,Lambda1est,Lambda2est, tau2[i], rhoest, maxit,tol);
    }
  }
  else{  
    cvtau2 = max(tau2);
    out = -sum(cv,0)/M;
    if(cvtau2 > 0){
      mat Ip;
      Ip.eye(Y.n_cols,Y.n_cols);
      mat tempinv = arma::inv_sympd((cvtau1*Omega) - YYest + (rhoest*Ip));
      mat Rest= Phiest;
      mat Lambda1est = 0*Phiest;
      for(uword  i = 0; i < l2.n_elem; i++){
        spatpcacore3(tempinv,Phiest,Rest,Cest,Lambda1est,Lambda2est, l2[i], rhoest, maxit,tol);
      }
    }
  }
  
  
  return List::create(Named("cv") = out,Named("est") = Phiest, Named("cvtau1") = cvtau1,Named("cvtau2") = cvtau2);
}
using namespace arma;
using namespace Rcpp;
using namespace std;

using namespace arma;
using namespace Rcpp;
using namespace std;
// [[Rcpp::export]]
arma::mat spatpca_rcpp(const arma::mat  sxy, const arma::mat Y, const int K, const double l1, const arma::vec l2,  const int maxit, const double tol){
  
  int d = sxy.n_cols;
  double rho;
  arma::mat UPhi, Phiold, Phi, R, C, Lambda1, Lambda2;
  arma::vec SPhi;
  arma::mat Omega;
  if(d ==2)
    Omega = tpmatrix(sxy);
  else
    Omega = cubicmatrix(sxy);
  arma::svd_econ(UPhi, SPhi, Phiold, Y,"right");
  Phi = Phiold.cols(0,K-1);
  C = Phi;
  rho = 10*pow(SPhi[0],2.0);
  
  Lambda2= Phi*(diagmat(rho -1/(rho-2*pow(SPhi.subvec(0,K-1),2)))); 
  mat YY = Y.t()*Y;
  
  if(l1 > 0)
    Phi = spatpcacore2p(YY,C,Lambda2, Omega, l1, rho, maxit,tol);
  

  if(max(l2)!=0){
    mat Ip;
    Ip.eye(Y.n_cols,Y.n_cols);
    mat tempinv = arma::inv_sympd((l1*Omega) - YY + (rho*Ip));
    Lambda1 = 0*Phi;
    R = Phi;
    for (unsigned int j = 0; j < l2.n_elem; j++){
      spatpcacore3(tempinv,Phi,R,C,Lambda1,Lambda2, l2[j], rho, maxit,tol);
   }
  }
  return Phi;
}

using namespace arma;
using namespace Rcpp;
using namespace std;
// [[Rcpp::export]]
arma::vec spatpcacv_gamma(const arma::mat Y, const arma::mat Phi, const int M, const arma::vec gamma, const arma::vec nk ){
  
  int p = Y.n_cols;
  int K = Phi.n_cols;
  int tempL, k;
  double totalvar, err, temp, tempSc, tempSc2;
  arma::mat Ytrain, Yvalid, Vc, Vc2, covtrain, covvalid,covest, Ip;
  arma::vec Sc, Sc2, Sct, Sctz, cv;
  arma::mat eigenvalue;
  cv.zeros(gamma.n_elem);
  Ip.eye(p,p);
  
  for(k = 1; k <= M; k++){
    Ytrain = Y.rows(arma::find(nk!=(k)));
    Yvalid = Y.rows(arma::find(nk==(k)));
    covtrain = arma::trans(Ytrain)*Ytrain/Ytrain.n_rows;
    covvalid = arma::trans(Yvalid)*Yvalid  /Yvalid.n_rows;
    totalvar = arma::trace(covtrain);
    arma::eig_sym(Sc,Vc, trans(Phi)*covtrain*Phi);
    tempSc2 = accu(Sc);
    Sc2 = sort(Sc,"descend");
    Vc2 = Vc.cols(sort_index(Sc,"descend"));
    
    Sct.ones(Sc2.n_elem);
    Sctz.zeros(Sc2.n_elem);
    
    for(unsigned int gj = 0; gj < gamma.n_elem; gj++){
      tempSc = tempSc2;
      tempL = K;
      if(Sc2[0]> gamma[gj]){
        err = (totalvar - tempSc+K*gamma[gj])/(p-tempL);
        temp = Sc2[tempL-1];  
        while( temp-gamma[gj] < err){
          if(tempL == 1){
            err = (totalvar - Sc2[0] + gamma[gj])/(p-1);
            break;
          }
          tempSc += -Sc2[tempL-1];
          tempL--;
          err = (totalvar - tempSc+tempL*gamma[gj])/(p-tempL);
          temp = Sc2[tempL-1];
        }
        if(Sc2[0]-gamma[gj] < err)
          err = (totalvar)/(p);
      }
      else{
        err = (totalvar)/(p);
      }
      
      eigenvalue = arma::max(Sc2-(err+gamma[gj])*Sct,Sctz);
      covest =  Phi*Vc2*diagmat(eigenvalue)*trans(Vc2)*trans(Phi);
      cv[gj] += pow(arma::norm(covvalid-covest - err*Ip,"fro"),2.0);
    }
  }
  return(cv);
}
