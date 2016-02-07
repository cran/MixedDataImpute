#ifndef CLUSTERS_H
#define CLUSTERS_H

#include <RcppArmadillo.h>
#include "rng.h"

using namespace Rcpp;
using namespace arma;
using namespace std;

////////////////////////////////////////////////////////////////////////////////
// Product multinomial cluster
////////////////////////////////////////////////////////////////////////////////
class ClusterPMN {
  public: 
  vector<vector<double> > val;
  vector<vector<double> > cumsum;
  vector<vector<double> > logval;
  vector<vector<double> > counts;
  ivec sizes;
  double n;
  int p, maxc;
  
  template<class T>
  void insert(T &x) {
    n++;
    for(int j=0; j<p; ++j) counts[j][x[j]] += 1;
  }
  
  template<class T>
  void remove(T &x) {
    n--;
    for(int j=0; j<p; ++j) counts[j][x[j]] -= 1;
  }
  
  template<class T>
  double loglik(T &x) {
    double out = 0.0;
    for(int j=0; j<p; ++j) out += logval[j][x[j]];
    return(out);
  }
  
  template<class T>
  double lik(T &x) {
    double out = 1.0;
    for(int j=0; j<p; ++j) out *= val[j][x[j]];
    return(out);
  }
  
  void update_value(int &j, int &oldx, int &newx) {
    counts[j][oldx]--;
    counts[j][newx]++;
  }
  
  void clear_stats();
  void sample_par(vec &gam);
  void update_cumsum();
  void normalize();
  void update_log();
  void recalc();
  int sample(int &j);
  
  mat flatten();
  
  ClusterPMN() {};
  ClusterPMN(ivec cx);
};

////////////////////////////////////////////////////////////////////////////////
// Univariate normal cluster, semiconjugate prior
////////////////////////////////////////////////////////////////////////////////
class ClusterNor {
  public:
  double mu;
  double phi, rootsigma2, logphi;
  double sumy, sumy2;
  double n;
  
  void insert(double y);
  
  void remove(double y);
  
  double loglik(double y);
  
  void clear_stats();
  
  void sample_par(double &m, double &v, double &a, double &b);
  
  double sample();
  
  mat flatten();
  
  ClusterNor();
};

////////////////////////////////////////////////////////////////////////////////
// Multivariate normal categorical regression cluster
////////////////////////////////////////////////////////////////////////////////

class ClusterMVReg {
  public:
  int p, q, pstar;
  mat B;
  vec vecB;
  mat Phi;
  mat cholPhi;
  double logdetPhi;
  
  mat XtX;
  mat YtX;
  mat YtY;
  double n;
  
  lookup_t ta;
  
  void insert(vec &y, vector<int> &x);
  void remove(vec &y, vector<int> &x);
  double loglik(vec y, vector<int> x); 
  
  vec get_mean(vector<int> &x);
  
  void update_x_value(vec &y, vector<int> &oldx, vector<int> &newx) {
    remove(y, oldx);
    insert(y, newx);
  }
  
  void clear_stats();
  
  // Phi ~ W(df, S0), E(Phi) = df*S0
  void sample_par(mat &B0, mat &PhiB, double &df, mat &S0inv);
  void sample_par_col(mat &B0, vec &tau, double &df, mat &S0inv);
  
  ClusterMVReg() {};
  ClusterMVReg(int q_, int p_, int pstar_, lookup_t ta_);
};

////////////////////////////////////////////////////////////////////////////////
// Multivariate normal *centered* categorical regression cluster
////////////////////////////////////////////////////////////////////////////////

class ClusterMVRegCent {
  public:
  int p, q, pstar;
  mat B;
  vec vecB;
  mat Phi;
  mat cholPhi;
  double logdetPhi;
  
  mat XtX;
  mat YtX;
  mat YtY;
  double n;
  
  lookup_t ta;
  
  void insert(vec &y, vector<int> &x);
  void remove(vec &y, vector<int> &x);
  double loglik(vec y, vector<int> x); 
  
  vec get_mean(vector<int> &x);
  
  void update_x_value(vec &y, vector<int> &oldx, vector<int> &newx) {
    remove(y, oldx);
    insert(y, newx);
  }
  
  void clear_stats();
  
  // Phi ~ W(df, S0), E(Phi) = df*S0
  void sample_par(mat &B0, mat &PhiB, double &df, mat &S0inv);
  
  ClusterMVRegCent() {};
  ClusterMVRegCent(int q_, int p_, int pstar_, lookup_t ta_);
};

#endif
