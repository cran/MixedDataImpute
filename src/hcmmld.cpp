#include "hcmmld_mvreg.h"
#include "hcmmld_pmn.h"

class HCMMLD {
  public:
  int kx, ky, k;
  int n;
  ivec cx;
  
  double alpha, xalpha, yalpha;
  double alpa, alpb;
  double beta_x_a, beta_x_b, beta_y_a, beta_y_b;
  double c, d;
  mat S0inv;
  vec gamma;
  double qb;
  
  MixPMN_HCMMLD Xmodel;
  MixMVReg_HCMMLD Ymodel;
  vector<Stick> Yprob;
  vector<Stick> Xprob;
  Stick prob;
  
  vector<vector<int> > My;
  vector<vector<int> > Mx;
  
  ivec alloc;
  
  void update_top_alloc() {
    vector<double> wt(k);
    for(int i=0; i<n; ++i) {
      for(int h=0; h<k; ++h) {
        wt[h] = prob(h, false)*Yprob[h](Ymodel.alloc(i),false)*Xprob[h](Xmodel.alloc(i),false);
      }
      int newh = rdisc(wt);
      if(newh != alloc(i)) {
        My[alloc(i)][Ymodel.alloc(i)]--;
        Mx[alloc(i)][Xmodel.alloc(i)]--;
        
        My[newh][Ymodel.alloc(i)]++;
        Mx[newh][Xmodel.alloc(i)]++;
        
        alloc(i) = newh;
      }
    }
  }
  void init_top_alloc() {
    vector<double> wt(k);
    for(int i=0; i<n; ++i) {
      for(int h=0; h<k; ++h) {
        wt[h] = prob(h, false)*Yprob[h](Ymodel.alloc(i),false)*Xprob[h](Xmodel.alloc(i),false);
      }
      int newh = rdisc(wt);
      My[newh][Ymodel.alloc(i)]++;
      Mx[newh][Xmodel.alloc(i)]++;
      
      alloc(i) = newh;
    }
  }
  void update_lower_model_par(bool bycol=true) {
    Xmodel.update_theta();
    Ymodel.update_theta(bycol);
    Ymodel.update_hyperpar();
  }
  void update_low_alloc() {
    Xmodel.update_alloc(Xprob, alloc, Mx);
    Ymodel.update_alloc(Yprob, alloc, Xmodel.X, My);
  }
  
  void update_stick() {

    for(int h=0; h<k; ++h) {
      Xprob[h].update_trunc(Mx[h]);
      Yprob[h].update_trunc(My[h]);
    }
    
    vec counts = zeros(k);
    for(int i=0; i<n; ++i) {
      counts(alloc(i)) += 1;
    }
    prob.update_trunc(counts);
    //double alpa = 1.5, alpb = 1.5;
    //double sigma2 = 100;
    //prob.update_block(counts, alpa, alpb, sigma2);
    //alpha = prob.alpha;
    
  }
  void update_conc() {
    vec counts = zeros(k);
    for(int i=0; i<n; ++i) {
      counts(alloc(i)) += 1;
    }
    
    double kstar = 0;
    for(int h=0; h<k; ++h) {
      if(counts[h]>0) ++kstar;
    }
    
    double xa = beta_x_a + kstar*(kx-1.0);
    double ya = beta_y_a + kstar*(ky-1.0);
    
    double xb = beta_x_b;
    double yb = beta_y_b;
    
    for(int h=0; h<k; ++h) {
      if(counts[h]>0) {
        xb -= Xprob[h].logprob[kx-1];
        yb -= Yprob[h].logprob[ky-1];
      }
    }
    
    xalpha = rgamma(xa, 1.0)/xb;
    yalpha = rgamma(ya, 1.0)/yb;
    
    for(int h=0; h<k; ++h) { 
      Xprob[h].alpha = xalpha;
      Yprob[h].alpha = yalpha;
    }
    
    prob.update_conc(alpa, alpb);
    alpha = prob.alpha;
    
  }
  
  void impute_Y() { Ymodel.impute_y(Xmodel.X); }
  
  void impute_X() {
    vec m(Ymodel.q);
    ClusterMVReg * yclus;
    ClusterPMN * xclus;
    
    //std::set<int> modified;
    //vector<int> xtmp(Xmodel.p);
    for(int j=0; j<Xmodel.p; ++j) {
      vec logwt(cx[j]);
      for(int i=0; i<n; ++i) {
        if(Xmodel.Xmask(i,j)>0) {
          //std::copy(X[i].begin(), X[i].end(), xtmp.begin());
          //xtmp[0] = j;
          
          yclus = &Ymodel.clusters[Ymodel.alloc(i)];
          xclus = &Xmodel.clusters[Xmodel.alloc(i)];
          //Rcout <<1;
          m = yclus->get_mean(Xmodel.X[i]);
          //Rcout <<2;
          if(Xmodel.X[i][j]>0) {
            //Rcout << Xmodel.X[i][j];
            //Rcout << endl << Ymodel.ta[j].size();
            //Rcout << Ymodel.ta[j][Xmodel.X[i][j]];
            m -= yclus->B.row(Ymodel.ta[j][Xmodel.X[i][j]]).t();
          }
          //Rcout <<3;
          // for x_j=0, beta = 0 so discrim is 0
          logwt(0) = xclus->logval[j][0];
          //Rcout <<4;
          for(int s=1; s<cx[j]; ++s) {
            logwt(s) = xclus->logval[j][s];
            logwt(s) += as_scalar(yclus->B.row(Ymodel.ta[j][s])*yclus->Phi*(Ymodel.Y[i]-m));
            logwt(s) -= 0.5*accu(square(yclus->cholPhi*yclus->B.row(Ymodel.ta[j][s]).t()));
          }
          //Rcout <<5;
          int newx = rdisc_log_inplace(logwt);
          //Rcout <<6;
          if(newx != Xmodel.X[i][j]) {
            xclus->update_value(j, Xmodel.X[i][j], newx);
            
            vector<int> xold = Xmodel.X[i];
            Xmodel.X[i][j] = newx;
            
            //can optimize here!
            yclus->update_x_value(Ymodel.Y[i], xold, Xmodel.X[i]); 
          }
        }
      }
    }
  }
  
  mat get_Mx() {
    mat val(k, kx);
    for(int r=0; r<k; ++r) {
      for(int s=0; s<kx; ++s) {
        val(r,s) = Mx[r][s];
      }
    }
    return(val);
  }
  
  mat get_My() {
    mat val(k, ky);
    for(int r=0; r<k; ++r) {
      for(int s=0; s<ky; ++s) {
        val(r,s) = My[r][s];
      }
    }
    return(val);
  }
  
  vec get_prob() {
    vec val(k);
    for(int h=0; h<k; ++h) val[h] = prob(h, false);
    return val;
  }
  
  void set_data(arma::mat Y_, arma::imat Ymask_, arma::imat X_, arma::imat Xmask_, 
                arma::ivec cx_, arma::imat lookup_table) {
    
    cx = cx_;
    
    n = Y_.n_rows;
    //Rcout <<1;
    alloc = ivec(n);
    alloc.fill(-1);
    //Rcout << 2;
    vec gam(cx.size());
    
    //Rcout << 3;
    for(int j=0; j<cx.size(); ++j) gam(j) = 1.0/cx(j);
    //Rcout << 4;
    Xmodel.init(X_, Xmask_, cx, gam, kx);
    Ymodel.init(Y_, Ymask_, X_, cx, lookup_table, ky);
    
    //Rcout << 5;
    
    Yprob.resize(k, Stick(ky, 2.0));
     //Rcout << 6;
    Xprob.resize(k, Stick(kx, 2.0));
     //Rcout << "updating top alloc" << endl;
    init_top_alloc();
     //Rcout << 8;
  }
  
  
  cube get_B() {
    cube res(Ymodel.pstar, Ymodel.q, Ymodel.k);
    for(int h=0; h<ky; ++h) res.slice(h) = Ymodel.clusters[h].B;
    return(res);
  }
  
  cube get_Phi() {
    cube res(Ymodel.q,Ymodel.q,Ymodel.k);
    for(int h=0; h<ky; ++h) res.slice(h) = Ymodel.clusters[h].Phi;
    return res;
  }
  
  cube get_Psi() {
    cube res(Xmodel.p, Xmodel.clusters[0].maxc, kx);
    for(int h=0; h<ky; ++h) res.slice(h) = Xmodel.clusters[h].flatten();
    return res;
  }
  
  ivec get_Xalloc() { return(Xmodel.alloc); }
  ivec get_Yalloc() { return(Ymodel.alloc); }
  vec  get_tau() { return(Ymodel.tau); }
  mat get_Y() {return(Ymodel.get_y()); }
  imat get_X() {
    imat X_(n, Xmodel.p);
    for(int i=0; i<n; ++i) {
      for(int j=0; j<Xmodel.p; ++j) {
        X_(i,j) = Xmodel.X[i][j];
      }
    }
    return X_;
  }
  
  List mcmc(int T, int status, int offset=0, int thin_trace=-1) {
    int s=0;
    std::vector<mat> B_trace, Sigma_trace;
    std::vector<double> beta_x_trace, beta_y_trace, alpha_trace;
    std::vector<ivec> kx_trace, ky_trace, kz_trace;
    
    while(s<T) {
      //for(int t=0; t<status; ++t) {
        s++;
        update_lower_model_par();
        update_top_alloc();
        update_low_alloc();
        
        impute_Y();
        impute_X();
        
        update_stick();
        update_conc();
        if(s % status==0) Rcout << "iteration "<< s+offset<< endl;
        if((thin_trace>-1) & (s % thin_trace==0)) {
          ivec ztab = tabulate<ivec>(alloc, k, 0);
          kz_trace.push_back(ztab);
          ivec xtab = tabulate<ivec>(Xmodel.alloc, kx, 0);
          kx_trace.push_back(xtab);
          ivec ytab = tabulate<ivec>(Ymodel.alloc, ky, 0);
          ky_trace.push_back(ytab);
        }
        
        Rcpp::checkUserInterrupt();
      }
    
    imat top_alloc(kz_trace.size(), k);
    imat Y_alloc(ky_trace.size(), ky);
    imat X_alloc(kx_trace.size(), kx);
    for(size_t i=0; i<kz_trace.size(); ++i) {
      top_alloc.row(i) = kz_trace[i].t();
      Y_alloc.row(i) = ky_trace[i].t();
      X_alloc.row(i) = kx_trace[i].t();
    }
    
    return(List::create(_["Z_alloc"] = top_alloc,
                        _["X_alloc"] = X_alloc,
                        _["Y_alloc"] = Y_alloc
                        ));
    //}
  }
  
  void set_hyperpar(Rcpp::List pars) {
    //alpha = as<double>(pars["alpha"]);
    alpa = as<double>(pars["alpha_a"]);
    alpb = as<double>(pars["alpha_b"]);
    beta_x_a = as<double>(pars["beta_x_a"]);
    beta_x_b = as<double>(pars["beta_x_b"]);
    beta_y_a = as<double>(pars["beta_y_a"]);
    beta_y_b = as<double>(pars["beta_y_b"]);
    Ymodel.tau_a = as<double>(pars["tau_a"]);
    Ymodel.tau_b = as<double>(pars["tau_b"]);
    c = as<double>(pars["v"]);
    d = as<double>(pars["w"]);
    qb = 1.0/as<double>(pars["sigma2_0beta"]);
    mat S0 = as<arma::mat>(pars["Sigma0"]);
    gamma = as<arma::vec>(pars["gamma"]);
    
    Ymodel.qb = qb;
    Ymodel.c = c; Ymodel.d = d;
    Ymodel.S0inv = inv(S0);
    Xmodel.gam = gamma;
    //S0inv = ((d-q-1)/c)*(eye(q,q));
    
    
    //double alpha, df, tau_a, tau_b, qb;
    //vec tau;
    //mat B, PhiB, Sinv, S0inv, B0;
    //double c, d;
    
  }
  
  HCMMLD(int k_, int kx_, int ky_) {

    k = k_;
    kx = kx_;
    ky = ky_;
    alpha = 1.0;
    
    alpa = .5;
    alpb = .5;
    beta_y_a = .5;
    beta_y_b = .5;
    beta_x_a = .5;
    beta_x_b = .5;
    
    
    My.resize(k);
    Mx.resize(k);
    for(int h=0; h<k; ++h) {
      for(int hh=0; hh<ky; ++hh) My[h].push_back(0);
      for(int hh=0; hh<kx; ++hh) Mx[h].push_back(0);
    }
    
    prob = Stick(k, alpha);
  }
  
};




RCPP_MODULE(HCMMLD){

    class_<HCMMLD>("hcmmld")
    
    .constructor<int , int , int>()
    .method( "set_data", &HCMMLD::set_data,  "documentation for set_data")
    .method("update_stick", &HCMMLD::update_stick)
    .method("update_lower_model_par", &HCMMLD::update_lower_model_par)
    .method("update_low_alloc", &HCMMLD::update_low_alloc)
    .method("update_top_alloc", &HCMMLD::update_top_alloc)
    .method("update_conc", &HCMMLD::update_conc)
    .method("impute_Y", &HCMMLD::impute_Y)
    .method("impute_X", &HCMMLD::impute_X)
    .method("mcmc", &HCMMLD::mcmc)
    .method("set_hyperpar", &HCMMLD::set_hyperpar)
    
    .field( "alloc", &HCMMLD::alloc)
    .field("xalpha", &HCMMLD::xalpha)
    .field("alpha", &HCMMLD::alpha)
    .field("yalpha", &HCMMLD::yalpha)
    .property( "B", &HCMMLD::get_B)
    .property( "Phi", &HCMMLD::get_Phi)
    .property( "Psi", &HCMMLD::get_Psi)
    .property( "Xalloc", &HCMMLD::get_Xalloc)
    .property( "Yalloc", &HCMMLD::get_Yalloc)
    .property( "tau", &HCMMLD::get_tau)
    .property( "X", &HCMMLD::get_X)
    .property( "Y", &HCMMLD::get_Y)    
    .property( "Mx", &HCMMLD::get_Mx)    
    .property( "My", &HCMMLD::get_My)  
    .property( "prob", &HCMMLD::get_prob)  
           /*
    .field( "alloc", &DPMixMVReg::alloc)
    .field( "alpha", &DPMixMVReg::alpha)
    .field( "Bmean", &DPMixMVReg::B)
    .field( "tau", &DPMixMVReg::tau)
    .property( "B", &DPMixMVReg::get_B)
    .property( "Phi", &DPMixMVReg::get_Phi)
    .property( "prob", &DPMixMVReg::get_prob)
    .property( "Y", &DPMixMVReg::get_y)
    // exposing member functions -- taken directly from std::vector<double>
    .method( "update_stick", &DPMixMVReg::update_stick)
    .method( "update_alloc", &DPMixMVReg::update_alloc)
    .method( "update_theta", &DPMixMVReg::update_theta)
    .method( "update_hyperpar", &DPMixMVReg::update_hyperpar)
    .method( "impute_y", &DPMixMVReg::impute_y)
    .method( "loglik_comp", &DPMixMVReg::loglik_comp)
    */

    ;
}

