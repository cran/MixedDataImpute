#ifndef npgenloc_pmn_h
#define npgenloc_pmn_h

#include "clusters.h"
#include "stick.h"

#include <boost/foreach.hpp>
#include <boost/unordered_map.hpp>
#define foreach BOOST_FOREACH

// objects to be held outside:
// mixture probabilities, X
class MixPMN_HCMMLD {
  public:
  int k, n, p;
  vector<vector<int> > X;
  imat Xmask; 
  
  vec gam;
  
  vector<ClusterPMN> clusters;
  
  ivec alloc;
  
  // TODO: hyperprior on gam?
  void update_hyperpar() {};

  void update_theta() {
    foreach(ClusterPMN &clus, clusters) {
      clus.sample_par(gam);
    }
  }
  
  void update_alloc(Stick &prob, vector<vector<int> > &X) {
    vec lik(k);
    for(int i=0; i<n; ++i) {
      //Rcout << X[i] << endl;
      for(int c=0; c<k; ++c) {
        lik(c) = prob(c, false) * clusters[c].lik(X[i]);
      }
      int newh = rdisc(lik);
      if(alloc(i) != newh) {
        clusters[alloc(i)].remove(X[i]);
        clusters[newh].insert(X[i]);
        alloc(i) = newh;
      }
    }
  }
  
  
  void update_alloc(vector<Stick> &prob, ivec &topalloc, vector<vector<int> > &Mx) {
    vec lik(k);
    for(int i=0; i<n; ++i) {
      //Rcout <<1;
      for(int c=0; c<k; ++c) {
        lik(c) = prob[topalloc(i)](c, false)*clusters[c].lik(X[i]);
      }
      //Rcout <<2;
      int newh = rdisc(lik);
      if(alloc(i) != newh) {
        //Rcout <<3;
        clusters[alloc(i)].remove(X[i]);
        clusters[newh].insert(X[i]);
        //Rcout <<4;
        Mx[topalloc(i)][alloc(i)]--;
        Mx[topalloc(i)][newh]++;
        //Rcout <<5;
        alloc(i) = newh;
      }
      //Rcout << endl;
    }
  }
  
  
  void init(arma::imat &X_, arma::imat &Xmask_, arma::ivec &cx, vec &gam_, int &k_) {
    k = k_;
  
    n = X_.n_rows;
    p = X_.n_cols;
    //Rcout << "init xmodel: ";
    //Rcout <<1;
    
    X.resize(n);
    Xmask = Xmask_;
    gam = gam_;
    //Rcout <<2;
    for(int i=0; i<n; ++i) {
      X[i].resize(p);
      for(int j=0; j<p; ++j) X[i][j] = X_(i,j);
    }
    //Rcout <<2;
    ClusterPMN c(cx);
    clusters.resize(k, c);
    //Rcout <<3;
    alloc = ivec(n);
    Stick prob = Stick(k, 2.0);
    //Rcout <<4;
    
    int i = 0;
    foreach(int &h, alloc) {
      h = rdisc(prob.prob, false);
      clusters[h].insert(X[i]); 
      ++i;
    }
    //Rcout <<5;
  }
  
  MixPMN_HCMMLD() {};
  
  MixPMN_HCMMLD(arma::imat X_, arma::imat Xmask_, arma::ivec cx, vec gam_, int k_){
    init( X_, Xmask_, cx, gam_, k_);
  };
  
};

#endif
