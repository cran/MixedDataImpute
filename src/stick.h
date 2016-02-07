#ifndef stick_h
#define stick_h

//#include <cassert>

#include "common.h"
#include "rng.h"
#include "StatFunctions.h"
using namespace std;

template<class T>
double blocklik(double alpha, T &counts, vector<double> &ngreater, int &kstar) {
  double loglik = double(kstar+1)*log(alpha);
  for(int h=0; h<kstar+1; ++h) {
    loglik += R::lbeta(1.+counts[h], alpha+ngreater[h]);
  }
  return loglik;
}

class Stick {
  public:
  vector<double> log1mwt;
  vector<double> logprob;
  vector<double> prob;
  double alpha;
  int k;
  
  void initialize();

  template<class T>
  void update_trunc(T &counts, double n=0) {
    double b = 0.0, a = 0.0, lga, lgb, lgsum;
    double csumlog1mwt = 0.0; // cumsum(log(1-v_h))
    
    if(n==0.0) {
      n = accumulate(counts.begin(), counts.end(), 0.0);
    }
    //vector<double> cumsum(k);
    //partial_sum(counts.begin(), counts.end(), cumsum.begin());
    
    double ncur = 0.0;
    for(int h=0; h<(k-1); ++h) {
      ncur += counts[h];
      a = 1.0 + counts[h]; 
      b = alpha + n - ncur;
      lga = rloggamma(a); 
      lgb = rloggamma(b);
      lgsum = logsumexp(lga, lgb);
      logprob[h] = lga - lgsum + csumlog1mwt;
      
      
      if(logprob[h] != logprob[h]) {
        Rcout << "a " << a << " b " << b;
        Rcout << "lga " << lga << " lgsum " << lgsum << " csumlog1mwt " << csumlog1mwt;
        Rcout << endl << endl;
        for (int s=0; s< k; ++s) {
          Rcout << counts[s] << " " << endl;
        }
        Rcout << endl<<endl;
        throw std::range_error("a logprob is nan");
      }
      
      csumlog1mwt += lgb - lgsum;
      prob[h] = exp(logprob[h]);
    }
    logprob[k-1] = csumlog1mwt;
    prob[k-1] = exp(logprob[k-1]);
  }
  
  void update_conc(double &a, double &b);
  
  
  template<class T>
  void update_block(T &counts, double &a, double &b, double sigma2=0.1) {
    
    vector<double> ngreater(k);
    int kstar = 0;
    double ncur = 0.0;
    
    //if(n==0.0) {
    double n = accumulate(counts.begin(), counts.end(), 0.0);
    //}
    
    for(int h=0; h<k; ++h) {
      if(counts[h]>0) kstar = h;
      ncur += counts[h];
      ngreater[h] = n - ncur;
    }
    /*
    double newalpha = exp(R::rnorm(log(alpha), sqrt(sigma2)));
    
    double log_acc = (a-1)*log(newalpha) - b*newalpha + blocklik(newalpha, counts, ngreater, kstar) + log(newalpha);
    log_acc -= (a-1)*log(alpha) - b*alpha + blocklik(alpha, counts, ngreater, kstar) + log(alpha);
    */
    
    double newalpha = rgamma(a, 1)/b;
    double log_acc = blocklik(newalpha, counts, ngreater, kstar);
    log_acc -= blocklik(alpha, counts, ngreater, kstar);
    
    
    if(log(runif()) < log_acc) alpha = newalpha;
    
    update_trunc(counts);
  
  }
  
  
  Stick() {};
  Stick(int k, double alpha);
  
  double& operator()(const int h, bool log=true) {
    if (!(h >= 0 && h < k)) Rcpp::stop("index out of bounds");
    if(log) {
      return logprob[h];
    } else {
      return prob[h];
    }
  }
};

#endif
