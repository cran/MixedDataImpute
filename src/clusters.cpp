#include "clusters.h"

////////////////////////////////////////////////////////////////////////////////
// Product multinomial cluster
////////////////////////////////////////////////////////////////////////////////
void ClusterPMN::clear_stats() {
  n = 0;
  for(int j=0; j<p; ++j) { 
    for(int c=0; c<sizes[j]; c++) {
      counts[j][c] = 0;
    }
  }
}

void ClusterPMN::sample_par(vec &gam) {
  for(int j=0; j<p; ++j) {
    for(int c=0; c<sizes[j]; ++c) {
      val[j][c] = rgamma(counts[j][c] + gam[j], 1.0);
    }
  }
  recalc();
}

void ClusterPMN::update_cumsum() {
  for(int j=0; j<p; ++j) {
    cumsum[j][0] = val[j][0];
    for(int c=1; c<sizes[j]; ++c) {
      cumsum[j][c] = val[j][c] + cumsum[j][c-1];
    }
  }
}

void ClusterPMN::normalize() {
  for(int j=0; j<p; ++j) {
    for(int c=0; c<sizes[j]; ++c) {
      val[j][c] = val[j][c]/cumsum[j][sizes[j]-1];
    }
  }
}

void ClusterPMN::update_log() {
  for(int j=0; j<p; ++j) {
    for(int c=0; c<sizes[j]; ++c) {
      logval[j][c] = log(val[j][c]);
    }
  }
}

void ClusterPMN::recalc() {
  update_cumsum();
  normalize();
  update_log();
}

int ClusterPMN::sample(int &j) {
  double s = cumsum[j][sizes[j]-1];
  double u = s*runif();
  int x = 0;
  while((u>cumsum[j][x])) { ++x; }
  return(x);
}

mat ClusterPMN::flatten() {
  mat out = zeros(p, maxc);
  for(int j=0; j<p; ++j) {
    for(int c=0; c<sizes[j]; ++c) {
      out(j,c) = val[j][c];
    }
  }
  return(out);
}

ClusterPMN::ClusterPMN(ivec cx) {
  sizes = cx;
  p = cx.size();
  n = 0;
  maxc = *std::max_element(cx.begin(), cx.end());
  val.resize(p);
  logval.resize(p);
  cumsum.resize(p);
  counts.resize(p);
  for(int j=0; j<p; ++j) {
    val[j].resize(cx[j]);
    logval[j].resize(cx[j]);
    cumsum[j].resize(cx[j]);
    counts[j].resize(cx[j]);
    for(int c=0; c<cx[j]; ++c) {
      val[j][c] = 1.0/cx[j];
    }
  }
  recalc();
}

////////////////////////////////////////////////////////////////////////////////
// Univariate normal cluster, semiconjugate prior
////////////////////////////////////////////////////////////////////////////////
void ClusterNor::insert(double y) {
  n++;
  sumy  += y;
  sumy2 += y*y;
}

void ClusterNor::remove(double y) {
  n--;
  sumy  -= y;
  sumy2 -= y*y;
}

double ClusterNor::loglik(double y) {
  double out = 0.5*logphi - 0.5*phi*(y-mu)*(y-mu);
  return(out);
}

void ClusterNor::clear_stats() {
  n = 0;
  sumy = 0;
  sumy2 = 0;
}

void ClusterNor::sample_par(double &m, double &v, double &a, double &b) {
  double fc_var = 1/(1/v + n*phi);
  double fc_mean = fc_var*(m/v + phi*sumy);
  mu = rnorm(fc_mean, sqrt(fc_var));
  
  double ssq = sumy2 - 2.0*mu*sumy + n*mu*mu;
  phi = rgamma(0.5*(a+n), 1.0)*2.0/(ssq + b);
  
  logphi = log(phi);
  rootsigma2 = 1.0/sqrt(phi);
}

double ClusterNor::sample() {
  double x = rnorm(mu, rootsigma2);
  return(x);
}

mat ClusterNor::flatten() {
  mat out = zeros(1, 2);
  out(0,0) = mu;
  out(0,1) = phi;
  return(out);
}

ClusterNor::ClusterNor() {
  n = 0;
  clear_stats();
  mu = 0;//rnorm(0, 1);
  phi = 1;//rgamma(2.0, 2.0);
}


////////////////////////////////////////////////////////////////////////////////
// Multivariate normal categorical regression cluster
////////////////////////////////////////////////////////////////////////////////
void ClusterMVReg::insert(vec &y, vector<int> &x) {
  ++n;
  XtX(0,0) += 1.0;
  YtX.col(0) += y;
  YtY += y*y.t();
  for(int j=0; j<p; ++j) {
    if(x[j]>0) {
      int ix = ta[j][x[j]];
      YtX.col(ix) += y;
      XtX(ix, 0) += 1;
      XtX(0, ix) = XtX(ix, 0);
      XtX(ix,ix) += 1;
      
      for(int s=(j+1); s<p; ++s) {
        if(x[s]>0) {
          int ixs = ta[s][x[s]];
          XtX(ixs, ix) += 1;
          XtX(ix, ixs) = XtX(ixs, ix);
        }
      }
    }
  }
}

void ClusterMVReg::remove(vec &y, vector<int> &x) {
  --n;
  if(n==0) {
    clear_stats();
  } else {
    XtX(0,0) -= 1.0;
    YtX.col(0) -= y;
    YtY -= y*y.t();
    for(int j=0; j<p; ++j) {
      if(x[j]>0) {
        int ix = ta[j][x[j]];
        YtX.col(ix) -= y;
        XtX(ix, 0) -= 1;
        XtX(0, ix) = XtX(ix, 0);
        XtX(ix,ix) -= 1;
        
        for(int s=(j+1); s<p; ++s) {
          if(x[s]>0) {
            int ixs = ta[s][x[s]];
            XtX(ixs, ix) -= 1;
            XtX(ix, ixs) = XtX(ixs, ix);
          }
        }
      }
    }
  }
}

vec ClusterMVReg::get_mean(vector<int> &x) {
  vec m = B.row(0).t();
  for(int j=0; j<p; ++j) {
    if(x[j]>0) {
      m += B.row(ta[j][x[j]]).t();
    }
  }
  return(m);
}

double ClusterMVReg::loglik(vec y, vector<int> x) {
  vec m = get_mean(x);
  double out = 0.5*(logdetPhi - accu(square(cholPhi*(y-m))));
  //double out = as_scalar(0.5*log(det(Phi)) - 0.5*trans(y-m)*Phi*(y-m));
  return(out);
}

void ClusterMVReg::clear_stats() {
  n = 0;
  XtX.fill(0);
  YtX.fill(0);
  YtY.fill(0);
}

// Phi ~ W(df, S0), E(Phi) = df*S0
void ClusterMVReg::sample_par(mat &B0, mat &PhiB, double &df, mat &S0inv) {

  mat fc_prec = PhiB + kron(Phi, XtX);
  mat tmp = YtX.t()*Phi;
  vec m(tmp.begin(), pstar*q);
  
  //Rcout << endl << "B" << endl;
  
  vec vecB0(B0.begin(), pstar*q, false, false);
  m += PhiB*vecB0;
  vecB = rmvnorm_post(1, m, fc_prec);
  B = mat(vecB.begin(), B.n_rows, B.n_cols);
  
  //mat ssq = -YtX*B;
  //ssq  += ssq.t() + YtY + B.t()*XtX*B;
  mat tmp0 = -YtX*B;
  mat ssq  = tmp0 + tmp0.t() + YtY + B.t()*XtX*B;
  //Phi = eye(q,q);
  //Rcout << "Phi" << endl;
  //Rcout << ssq << endl;
  //Rcout << S0inv << endl;
  //Rcout << df<<" ";
  //Rcout << n<< endl;
  mat S = inv(S0inv + ssq);
  //Rcout << S <<endl;
  try {
    Phi = rwish(S, df + n);
  } catch (...) {
    Rcout << S << endl;
    Rcout << ssq << endl;
    Rcout << S0inv << endl;
    Rcpp::stop("");
  }
  //Rcout << "cholPhi" << endl;
  cholPhi = chol(Phi);
  logdetPhi = 2.0*accu(log(diagvec(cholPhi)));
}

// Gibbs w/ blocking on columns of B
// Phi ~ W(df, S0), E(Phi) = df*S0
void ClusterMVReg::sample_par_col(mat &B0, vec &tau, double &df, mat &S0inv) {
  
  //mat Btmp = B;
  int sweeps = 2;
  for(int s=0; s<sweeps; ++s) {
    for(int r=0; r<q; ++r) {
      //indexing uvec
      vector<int> vnotr;
      for(int rr=0; rr<q; ++rr) { if (rr!=r) { vnotr.push_back(rr); } }
      uvec unotr = conv_to< uvec >::from(vnotr);
      uvec ur(1); ur(0) = r;
      
      vec b = tau(r)*B0.cols(ur) + Phi(r,r)*YtX.rows(ur).t() + (YtX.rows(unotr).t() - XtX*B.cols(unotr))*Phi.submat(unotr, ur);
      //b *= Phi(r,r);
      mat fc_prec = tau(r)*eye(pstar, pstar) + XtX*Phi(r,r);
      try {
        B.col(r) = rmvnorm_post(1, b, fc_prec);
      } catch(...) {
        Rcout << "clusters 307" << endl;
        Rcout << Phi(r,r) <<endl;
        Rcout << XtX << endl;
        Rcout << tau(r) << endl;
        Rcout << "\n\n" << 310;
        throw(Rcpp::exception("chol(fc_prec) failed","clusters.cpp",310));
      }
    }
  }
  //B=Btmp;
  
  //Rcout << B << endl;
  
  //mat ssq = -YtX*B;
  mat tmp = -YtX*B;
  mat ssq  = tmp + tmp.t() + YtY + B.t()*XtX*B;
  //Phi = eye(q,q);
  //Rcout << "Phi" << endl;
  //Rcout << ssq << endl;
  //Rcout << S0inv << endl;
  //Rcout << df<<" ";
  //Rcout << n<< endl;
  mat S = inv(S0inv + ssq);
  //Rcout << S <<endl;
  
  try {
    Phi = rwish(S, df + n);
  } catch(...) {
    Rcout << S0inv << endl;
    Rcout << ssq << endl;
    Rcout << n << endl;
    Rcout << YtY << endl;
    Rcout << B.t()*XtX*B << endl;
    Rcout << "clusters 337" << endl;
    throw(Rcpp::exception("draw Phi failed","clusters.cpp",333));
  }
  
  //Rcout << "local Phi" << endl;
  Phi = rwish(S, df + n);
  //Rcout << "cholPhi" << endl;
  // cholPhi = chol(Phi);
  bool ok = chol(cholPhi, Phi);
  if(!ok) {
    Rcout << S0inv << endl;
    Rcout << ssq << endl;
    Rcout << n << endl;
    throw(Rcpp::exception("chol(Phi) failed","clusters.cpp",330));
  }
  logdetPhi = 2.0*accu(log(diagvec(cholPhi)));
  
  vecB = vec(B.begin(), q*pstar, false, false);
}



ClusterMVReg::ClusterMVReg(int q_, int p_, int pstar_, lookup_t ta_) {
  ta = ta_;
  q = q_;
  p = p_;
  pstar = pstar_;
  n = 0;
  
  B = mat(pstar, q); B.fill(0);
  vecB = vec(B.begin(), q*pstar, false, false);
  Phi = eye(q, q);
  
  XtX = mat(pstar, pstar);
  YtX = mat(q, pstar);
  YtY = mat(q, q);
  XtX.fill(0);
  YtY.fill(0);
  YtX.fill(0);
  
  mat B0(pstar, q); B0.fill(0);
  mat PhiB = eye(pstar*q, pstar*q);
  double df = q + 1;
  mat S0inv = df*eye(q,q);
  
  sample_par(B0, PhiB, df, S0inv);
  
}



////////////////////////////////////////////////////////////////////////////////
// Multivariate normal *centered* categorical regression cluster
////////////////////////////////////////////////////////////////////////////////
void ClusterMVRegCent::insert(vec &y, vector<int> &x) {
  ++n;
  YtY += y*y.t();
  for(int j=0; j<p; ++j) {
    int ix = ta[j][x[j]];
    YtX.col(ix) += y;
    XtX(ix,ix) += 1;
    
    for(int s=(j+1); s<p; ++s) {
      int ixs = ta[s][x[s]];
      XtX(ixs, ix) += 1;
      XtX(ix, ixs) = XtX(ixs, ix);
    }
  }
}

void ClusterMVRegCent::remove(vec &y, vector<int> &x) {
  --n;
  if(n==0) {
    clear_stats(); 
  } else {
    YtY -= y*y.t();
    for(int j=0; j<p; ++j) {
      int ix = ta[j][x[j]];
      YtX.col(ix) -= y;
      XtX(ix,ix) -= 1;
      
      for(int s=(j+1); s<p; ++s) {
        int ixs = ta[s][x[s]];
        XtX(ixs, ix) -= 1;
        XtX(ix, ixs) = XtX(ixs, ix);
      }
    }
  }
}

vec ClusterMVRegCent::get_mean(vector<int> &x) {
  vec m = B.row(0).t();
  for(int j=0; j<p; ++j) {
    m += B.row(ta[j][x[j]]).t();
  }
  return(m);
}

double ClusterMVRegCent::loglik(vec y, vector<int> x) {
  vec m = get_mean(x);
  double out = 0.5*(logdetPhi - accu(square(cholPhi*(y-m))));
  //double out = as_scalar(0.5*log(det(Phi)) - 0.5*trans(y-m)*Phi*(y-m));
  return(out);
}

void ClusterMVRegCent::clear_stats() {
  n = 0;
  XtX.fill(0);
  YtX.fill(0);
  YtY.fill(0);
}

// Phi ~ W(df, S0), E(Phi) = df*S0
void ClusterMVRegCent::sample_par(mat &B0, mat &PhiB, double &df, mat &S0inv) {

  mat fc_prec = PhiB + kron(Phi, XtX);
  mat tmp = YtX.t()*Phi;
  vec m(tmp.begin(), pstar*q);
  
  //Rcout << endl << "B" << endl;
  
  vec vecB0(B0.begin(), pstar*q, false, false);
  m += PhiB*vecB0;
  vecB = rmvnorm_post(1, m, fc_prec);
  B = mat(vecB.begin(), B.n_rows, B.n_cols);
  
  mat ssq = -YtX*B;
  ssq  += ssq.t() + YtY + B.t()*XtX*B;
  //Phi = eye(q,q);
  //Rcout << "Phi" << endl;
  //Rcout << ssq << endl;
  //Rcout << S0inv << endl;
  //Rcout << df<<" ";
  //Rcout << n<< endl;
  mat S = inv(S0inv + ssq);
  //Rcout << S <<endl;
  Phi = rwish(S, df + n);
  //Rcout << "cholPhi" << endl;
  cholPhi = chol(Phi);
  logdetPhi = 2.0*accu(log(diagvec(cholPhi)));
}

ClusterMVRegCent::ClusterMVRegCent(int q_, int p_, int pstar_, lookup_t ta_) {
  ta = ta_;
  q = q_;
  p = p_;
  pstar = pstar_;
  n = 0;
  
  B = mat(pstar, q); B.fill(0);
  vecB = vec(B.begin(), q*pstar, false, false);
  Phi = eye(q, q);
  
  XtX = mat(pstar, pstar);
  YtX = mat(q, pstar);
  YtY = mat(q, q);
  XtX.fill(0);
  YtY.fill(0);
  YtX.fill(0);
  
  mat B0(pstar, q); B0.fill(0);
  mat PhiB = eye(pstar*q, pstar*q);
  double df = q + 1;
  mat S0inv = df*eye(q,q);
  
  sample_par(B0, PhiB, df, S0inv);
  
}

