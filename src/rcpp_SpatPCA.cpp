// includes from the plugin
#include <RcppArmadillo.h>
#include <Rcpp.h>
// [[Rcpp::depends(RcppArmadillo)]]


// user includes
using namespace arma;
using namespace Rcpp;
using namespace std;
void spatPCAcore(const mat Y, mat& Phi, mat& R,  mat& C,  mat& Lambda1, mat& Lambda2, const mat Omega,const double tau1, const double tau2, double rho,const double rhoincre,const int maxit,const double tol){
    
    int p = Phi.n_rows;
    int K = Phi.n_cols;
    int iter = 0;
    arma::mat Ip, Sigtau1, temp, zero, one;
    arma::vec er(4);
    Ip.eye(p,p);
    zero.zeros(p,K);
    one.ones(p,K);
    Sigtau1 = tau1*Omega - arma::trans(Y)*Y;
    
    
    
    if(tau2 ==0){
        arma::mat eigvec;
        arma::vec eigval;
        eig_sym( eigval, eigvec, Sigtau1);
        
        Phi = eigvec.cols(0,K-1);
        R = Phi;
        C = Phi;
        Lambda1 = zero;
        Lambda2 = -2*(Sigtau1)*Phi;
        if (eigval(K-1) > 0) {
            arma::mat U;
            arma::vec S;
            arma::mat V;
            arma::mat Phiold = Phi;
            arma::mat Rold = R;
            arma::mat Cold = C;
            arma::mat Lambda1old = Lambda1;
            arma::mat Lambda2old = Lambda2;
            
            for (iter = 0; iter < maxit; iter++){
                Phi =0.5*arma::solve(Sigtau1 + rho*Ip,(rho*(Rold+Cold)-Lambda1old-Lambda2old));
                R = arma::sign((Lambda1old/rho+Phi))%arma::max(zero, arma::abs((Lambda1old/rho+Phi)) - tau2*one/rho);
                temp = Phi+Lambda2old/rho;
                arma::svd_econ(U, S, V,temp);
                C = U.cols(0,V.n_cols-1)*V.t();
                
                Lambda1 = Lambda1old +rho*(Phi-R);
                Lambda2 = Lambda2old +rho*(Phi-C);
                
                if(rho*rhoincre <= pow(10.0,10.0))
                    rho = rho*rhoincre;
                else
                    rho=pow(10.0,10.0);
                
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
        }
    }
    else{
        arma::mat U;
        arma::vec S;
        arma::mat V;
        arma::mat Phiold = Phi;
        arma::mat Rold = R;
        arma::mat Cold = C;
        arma::mat Lambda1old = Lambda1;
        arma::mat Lambda2old = Lambda2;
        
        for (iter = 0; iter < maxit; iter++){
            Phi =0.5*arma::solve(Sigtau1 + rho*Ip,(rho*(Rold+Cold)-Lambda1old-Lambda2old));
            R = arma::sign((Lambda1old/rho+Phi))%arma::max(zero, arma::abs((Lambda1old/rho+Phi)) - tau2*one/rho);
            temp = Phi+Lambda2old/rho;
            arma::svd_econ(U, S, V,temp);
            C = U.cols(0,V.n_cols-1)*V.t();
            
            Lambda1 = Lambda1old +rho*(Phi-R);
            Lambda2 = Lambda2old +rho*(Phi-C);
            
            if(rho*rhoincre <= pow(10.0,10.0))
                rho = rho*rhoincre;
            else
                rho=pow(10.0,10.0);
            
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
    }
    iter++;
    if(iter == maxit)
        Rcpp::Rcout<<"Not converge at tau1="<<tau1<<" tau2="<<tau2<<"\n"<<std::endl;
}

// [[Rcpp::depends(RcppArmadillo)]]
// user includes

using namespace arma;
//using namespace Rcpp;
using namespace std;
// [[Rcpp::export]]
arma::mat spatPCAcv_rcpp(const arma::mat Y, const int M, const int K, const arma::mat Omega, const arma::vec tau1, const arma::vec tau2, const arma::vec nk, const double rhoincre, const int maxit, const double tol){
    
    int p = Y.n_cols, k;
    double rho;
    arma::mat Ytrain, Yvalid, UPhi, Phiold, Phi, R, C, Lambda1, Lambda2, Ip, eigenvalue, cv;
    arma::vec SPhi;
    cv.zeros(tau1.n_elem,tau2.n_elem);
    Ip.eye(p,p);
    
    
    for( k = 1; k <= M; k++){
        Ytrain = Y.rows(arma::find(nk!=(k)));
        Yvalid = Y.rows(arma::find(nk==(k)));
        arma::svd_econ(UPhi, SPhi, Phiold, Ytrain);
        rho = 10*pow(SPhi[0],2.0);
        Phi = Phiold.cols(0,K-1);
        R = Phi;
        C = Phi;
        Lambda1 = 0*Phi;
        Lambda2 = 0*Phi;
        for(unsigned int i = 0; i < tau1.n_elem; i++){
            for(unsigned int j = 0; j < tau2.n_elem; j++){
                spatPCAcore(Ytrain,Phi,R,C,Lambda1,Lambda2, Omega, tau1[i],tau2[j], rho, rhoincre, maxit,tol);
                cv(i,j) += pow(arma::norm(Yvalid*(Ip-Phi*Phi.t()),"fro"),2.0);
            }
        }
    }
    return(cv);
}

using namespace arma;
using namespace Rcpp;
using namespace std;
// [[Rcpp::export]]
arma::mat spatPCAcv_rcpp_parallel(const arma::mat Y, const int m, const int K, const arma::mat Omega, const arma::vec tau1, const arma::vec tau2, const arma::vec nk, const double rhoincre, const int maxit, const double tol){
    
    int p = Y.n_cols;
    double rho;
    arma::mat Ytrain, Yvalid, UPhi, Phiold, Phi, R, C, Lambda1, Lambda2, Ip, eigenvalue, cv;
    arma::vec SPhi;
    cv.zeros(tau1.n_elem,tau2.n_elem);
    Ip.eye(p,p);
    
    Ytrain = Y.rows(arma::find(nk!=(m)));
    Yvalid = Y.rows(arma::find(nk==(m)));
    arma::svd_econ(UPhi, SPhi, Phiold, Ytrain);
    rho = 10*pow(SPhi[0],2.0);
    Phi = Phiold.cols(0,K-1);
    R = Phi;
    C = Phi;
    Lambda1 = 0*Phi;
    Lambda2 = 0*Phi;
    for(unsigned int i = 0; i < tau1.n_elem; i++){
        for(unsigned int j = 0; j < tau2.n_elem; j++){
            spatPCAcore(Ytrain,Phi,R,C,Lambda1,Lambda2, Omega, tau1[i],tau2[j], rho, rhoincre, maxit,tol);
            cv(i,j) += pow(arma::norm(Yvalid*(Ip-Phi*Phi.t()),"fro"),2.0);
        }
    }
    
    return(cv);
}


using namespace arma;
using namespace Rcpp;
using namespace std;
// [[Rcpp::export]]
arma::mat spatPCA_rcpp(const arma::mat Y, const int K, const arma::mat Omega, const double tau1, const arma::vec l2, const double rhoincre, const int maxit, const double tol){
    
    double rho;
    arma::mat UPhi, Phiold, Phi, R, C, Lambda1, Lambda2;
    arma::vec SPhi;
    
    arma::svd_econ(UPhi, SPhi, Phiold, Y,"right");
    Phi = Phiold.cols(0,K-1);
    R = Phi;
    C = Phi;
    Lambda1 = 0*Phi;
    Lambda2 = 0*Phi;
    rho = 10*(svd(trans(Y)*Y)[0]);
    
    if(tau1!=0){
        spatPCAcore(Y, Phi, R, C, Lambda1, Lambda2, Omega, tau1, 0, rho, rhoincre, maxit, tol);
    }
    if(max(l2)!=0){
        for (unsigned int j = 0; j < l2.n_elem; j++){
            spatPCAcore(Y, Phi, R, C, Lambda1, Lambda2, Omega, tau1, l2[j], rho, rhoincre, maxit, tol);
        }
    }
    return Phi;
}




using namespace arma;
using namespace Rcpp;
using namespace std;
// [[Rcpp::export]]
arma::vec spatPCAcv_gamma(const arma::mat Y, const arma::mat Phi, const int M, const arma::vec gamma, const arma::vec nk ){
    
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
