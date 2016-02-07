#include "stick.h"

void Stick::initialize() {
  vector<double> c(k);
  std::fill(c.begin(), c.end(), 0.0);
  update_trunc(c);
}

void Stick::update_conc(double &a, double &b) {
  double afc = a + k - 1;
  double bfc = b - logprob[k-1];
  //std::vector<double> tmp(k);
  //std::transform(logwt.begin(), logwt.end(), tmp.begin(), ::exp);
  //for(int h=1; h<k; ++h) bfc -= (logprob[h]-logprob[h-1]);
  alpha = rgamma(afc, 1.0)/bfc;
  
  if(alpha != alpha) {
    Rcout << "a: " << afc << " b: " << bfc << endl << endl;
    Rcout << "logprob: ";
    for(int h=0; h<k; ++h) {
      Rcout << logprob[h] << endl;
    }
    Rcout << "prob: ";
    for(int h=0; h<k; ++h) {
      Rcout << prob[h] << endl;
    }
    throw std::range_error("alpha is nan");
  }
}



Stick::Stick(int k_, double alpha_) {
  k = k_;
  alpha = alpha_;
  log1mwt.resize(k);
  logprob.resize(k);
  prob.resize(k);
  initialize();
}
