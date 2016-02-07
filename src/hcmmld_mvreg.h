#ifndef npgenloc_mvreg_h
#define npgenloc_mvreg_h

#include "clusters.h"
#include "stick.h"

#include <boost/foreach.hpp>
#include <boost/unordered_map.hpp>
#define foreach BOOST_FOREACH

// objects to be held outside:
// mixture probabilities, X
class MixMVReg_HCMMLD {
  public:
  int k, n, p, q, pstar;
  vector<vec> Y;
  boost::unordered_map<vector<int>, vector<int> > Ymis;
  
  lookup_t ta;
  vector<ClusterMVReg> clusters;
  
  ivec alloc;
  double alpha, df, tau_a, tau_b, qb;
  vec tau;
  mat B, PhiB, Sinv, S0inv, B0;
  double c, d;

  mat get_y() {
    return arma_rbind_all(Y);
  }
  
  // could optimize here
  void impute_y(vector<vector<int> > &X) {
    // ix = index of missing obs, nix = not missing
    uvec ix, nix;
    vec m(q);
    
    typedef pair<vector<int>, vector<int> > p_t;
    foreach(p_t pattern, Ymis) {
      vector<int> obs_idx = pattern.second;
      vector<int> tix;
      vector<int> tnix;
      for(int j=0; j<q; ++j) {
        if(pattern.first[j]==0) {
          tnix.push_back(j);
        } else {
          tix.push_back(j);
        }
      }
      
      if(tix.size()==q) {
        //all missing
        cube cholcov(q, q, k);
        for(int h=0; h<k; ++h) {
          cholcov.slice(h) = chol(inv(clusters[h].Phi));
        }
        vec z(q);
        
        foreach(int i, obs_idx) {
          clusters[alloc(i)].remove(Y[i], X[i]);
          
          m = clusters[alloc(i)].get_mean(X[i]);
          generate(z.begin(), z.end(), znorm);
          Y[i] = m + cholcov.slice(alloc(i))*z;
          
          clusters[alloc(i)].insert(Y[i], X[i]);
          
        }
        
      } else {
        
        ix  = conv_to< uvec >::from(tix);
        nix = conv_to< uvec >::from(tnix);
        
        cube cholcov(ix.size(), ix.size(), k);
        cube A(ix.size(), nix.size(), k);
        for(int h=0; h<k; ++h) {
          mat cPhi = clusters[h].Phi.submat(ix,ix);
          cholcov.slice(h) = chol(inv(cPhi));
          A.slice(h) = solve(cPhi, clusters[h].Phi.submat(ix, nix));
        }
        
        vec cmean(ix.size());
        vec z(ix.size());
        
        foreach(int i, obs_idx) {
          clusters[alloc(i)].remove(Y[i], X[i]);
          
          m = clusters[alloc(i)].get_mean(X[i]);
          cmean = m.elem(ix) - A.slice(alloc(i))*(Y[i].elem(nix) - m.elem(nix));
          generate(z.begin(), z.end(), znorm);
          Y[i].elem(ix) = cmean + cholcov.slice(alloc(i))*z;
          
          clusters[alloc(i)].insert(Y[i], X[i]);
          
        }
      }
    }
  }
  
  // vec B_h ~ N(B, (1/tau)*I), vec B ~ N(B0, Qb)
  void update_hyperpar() {
    //vec vecB0(B0.begin(), pstar*q);
    //mat Qbinv = 0.1*eye(q*pstar, q*pstar);
    
    mat Sscale_inv = S0inv;
    
    int kstar = 0;
    mat Bsum(pstar, q); Bsum.fill(0);
    mat B2sum(pstar, q); B2sum.fill(0);
    for(int h=0; h<k; ++h) {
      if(clusters[h].n>0) {
        Bsum += clusters[h].B;
        B2sum += square(clusters[h].B);
        
        Sscale_inv += clusters[h].Phi;
        
        ++kstar;
      }
    }
    
    /*
    mat fcprec = kstar*PhiB + Qbinv;
    vec m0(Bsum.begin(), pstar*q);
    vec m = PhiB*m0;
    vec vecB = rmvnorm_post(1, m, fcprec);
    B = mat(vecB.begin(), pstar, q);
    */
    
    
    for(int r=0; r<q; ++r) {
      for(int s=0; s<pstar; ++s) {
        double aa = 1.0/(qb + kstar*tau(r));
        B(s, r) = rnorm(aa*tau(r)*Bsum(s, r), aa);
      }
    }
    
    
    for(int r=0; r<q; ++r) {
      // for the hierarchical form of the half t prior
      //s_tau(r) = rgamma(0.5, 1.0)/(nu_tau*tau(r) + 1.0/(A_tau*A_tau));
      //double afc = 0.5*(nu_tau + kstar*pstar);
      
      double ssq = accu(B2sum.col(r)) - 2*accu(Bsum.col(r)%B.col(r)) + kstar*accu(square(B.col(r)));
      
      // for the hierarchical form of the half t prior
      //double bfc = 0.5*(2*nu_tau*s_tau(r) + ssq);
      
      double afc = 0.5*(tau_a + kstar*pstar);
      double bfc = 0.5*(tau_b + ssq);
      tau(r) = fmin(rgamma(afc, 1.0)/bfc, 1e10);
    }
    
    mat Sscale = inv(Sscale_inv);
    Sinv = rwish(Sscale, c+kstar*d);
    
  }

  void update_theta(bool bycol = true) {
    
    foreach(ClusterMVReg &clus, clusters) {
      if(bycol) {
        clus.sample_par_col(B, tau, c, Sinv);
      } else {
        mat PhiBmat = repmat(tau.t(), pstar, 1);
        PhiB = diagmat(vec(PhiBmat.begin(), q*pstar));
        clus.sample_par(B, PhiB, c, Sinv);
      }
    }
  }
  
  void update_alloc(Stick &prob, vector<vector<int> > &X) {
    vec loglik(k);
    for(int i=0; i<n; ++i) {
      for(int c=0; c<k; ++c) {
        loglik(c) = prob(c) + clusters[c].loglik(Y[i], X[i]);
      }
      int newh = rdisc_log_inplace(loglik);
      if(alloc(i) != newh) {
        clusters[alloc(i)].remove(Y[i], X[i]);
        clusters[newh].insert(Y[i], X[i]);
        alloc(i) = newh;
      }
    }
  }
  
  
  void update_alloc(vector<Stick> &prob, ivec &topalloc, vector<vector<int> > &X,
                    vector<vector<int> > &My) {
    vec loglik(k);
    for(int i=0; i<n; ++i) {
      for(int c=0; c<k; ++c) {
        loglik(c) = prob[topalloc(i)](c) + clusters[c].loglik(Y[i], X[i]);
      }
      int newh = rdisc_log_inplace(loglik);
      if(alloc(i) != newh) {
        clusters[alloc(i)].remove(Y[i], X[i]);
        clusters[newh].insert(Y[i], X[i]);
        
        My[topalloc(i)][alloc(i)]--;
        My[topalloc(i)][newh]++;
        
        alloc(i) = newh;
      }
    }
  }
  
  cube get_B() {
    cube res(pstar, q, k);
    for(int h=0; h<k; ++h) res.slice(h) = clusters[h].B;
    return(res);
  }
  
  cube get_Phi() {
    cube res(q,q,k);
    for(int h=0; h<k; ++h) res.slice(h) = clusters[h].Phi;
    return res;
  }
  
  double loglik_comp(int h, vec y, ivec x_) {
    vector<int> x(x_.size());
    for(int j=0; j<x_.size(); ++j) x[j] = x_(j);
    double out = clusters[h].loglik(y, x);
    return out;
  }
  
  void init(arma::mat &Y_, arma::imat &Ymask, arma::imat &X_, arma::ivec &cx, arma::imat &lookup_table, int &k_) {
    k = k_;
    ta = make_lookup(lookup_table, cx);

    tau_a = 1; tau_b = 1;
  
    n = X_.n_rows;
    p = X_.n_cols;
    q = Y_.n_cols;
    pstar = lookup_table(lookup_table.n_rows-1, 2) + 1;
    
    vector<vector<int> > X; X.resize(n);
    //Y.resize(n);
    for(int i=0; i<n; ++i) {
      X[i].resize(p);
      for(int j=0; j<p; ++j) X[i][j] = X_(i,j);
      vec ytmp = Y_.row(i).t();
      Y.push_back(ytmp);
      
      //missing_pat pattern;
      //uvec ix    = find(ytmp!=ytmp);
      //Ymis.push_back(ix);
      uvec u = Ymask.row(i).t() == 1;//(ytmp != ytmp);
      vector<int> ix; int nmis = 0;
      foreach(int ii, u) {
        nmis += ii;
        ix.push_back(ii);
      }
      if(nmis>0) {
        Ymis[ix].push_back(i);
      }
      
    }
    
    c = q + 1;
    d = q + 2;
    S0inv = ((d-q-1)/c)*(eye(q,q));
    Sinv = S0inv;
    qb = 0.1;
    
    B0 = mat(pstar, q);
    B0.fill(0);
    
    alpha = 1.0;
    B = mat(pstar, q); B.fill(0);
    tau = vec(q);
    tau.fill(1);
    
    mat PhiBmat = repmat(tau.t(), pstar, 1);
    PhiB = diagmat(vec(PhiBmat.begin(), q*pstar));
    
    
    ClusterMVReg c(q, p, pstar, ta);
    clusters.resize(k, c);
    alloc = ivec(n);
    Stick prob = Stick(k, alpha);
    
    int i=0;
    foreach(int &h, alloc) {
      h = rdisc(prob.prob, false);
      clusters[h].insert(Y[i], X[i]); 
      ++i;
    }
    
  }
  
  MixMVReg_HCMMLD() {};
  
  MixMVReg_HCMMLD(arma::mat Y_, arma::imat Ymask, arma::imat X_, arma::ivec cx, arma::imat lookup_table, int k_) {
    init(Y_, Ymask, X_, cx, lookup_table, k_) ;
  }

};

#endif
