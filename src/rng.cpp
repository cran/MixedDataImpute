#include "rng.h"

double rgamma(const double &a, const double &b) { 
  return R::rgamma(a, b); 
}
double rbeta(const double &a, const double &b) { 
  return R::rbeta(a, b); 
}
double runif() { 
  return R::runif(0.0, 1.0); 
}
double rchisq(const double &df) { 
  return Rf_rchisq(df); 
}
double znorm() { 
  return R::rnorm(0.0, 1.0); 
}
double rnorm(const double &mu, const double &sd) { 
  return R::rnorm(mu, sd); 
}
double rlogstick(double &alpha) {
  return log(1-pow(runif(), 1.0/alpha));
}

double rlogbeta(const double &alpha, const double &beta) {
  return log(rbeta(alpha, beta));
}

double rloggamma(const double &alpha) {
  if(alpha<0.5) {
    return log(rgamma(alpha+1.0, 1.0)) + log(runif())/alpha;
    //return(log(2) + log(runif())/alpha + log(rgamma(alpha+1, 1.0)));
  } else {
    return log(rgamma(alpha, 1.0));
    //return log(rgamma(alpha, 2.0));
  }
}

// // // [[Rcpp::export]]
mat rwish_I_root(int p, double df) {
  mat T(p, p);
  for(int i=0; i<p; i++) {
    T(i,i) = sqrt(rchisq(df-i));//no +1 because of 0 indexing!
      for(int j=0; j<i; j++) {
        T(i,j) = znorm();//Rf_rnorm(0., 1.);
      }
  }
  return(trimatl(T));
}

// // [[Rcpp::export]]
mat rwish_root(mat L, double df) {
  mat T = rwish_I_root(L.n_rows, df);
  return(trimatl(L)*trimatl(T));
}

// // [[Rcpp::export]]
mat rwish(mat S, double df) {
  mat R = chol(S);
  mat out = rwish_root(R.t(), df);
  out = out*out.t();
  return(out);
}

// // [[Rcpp::export]]
mat cpp_rmvnorm_prec(int n, vec &mu, mat &Phi) {
  RNGScope tmp;
  //NumericVector z_ = rnorm(n*mu.size());
  //mat z(z_.begin(), mu.size(), n, false, false);
  mat z(mu.size(), n);
  std::generate(z.begin(), z.end(), znorm);
  mat res = solve(trimatu(chol(Phi)), z);
  res.each_col() += mu;
  return(res);
}

//sample from N(Phi^(-1)m, Phi^(-1))
// // [[Rcpp::export]]
mat rmvnorm_post(int n, vec &m, mat &Phi) {
  RNGScope tmp;
  mat R = chol(Phi);
  vec mu = solve(trimatu(R), solve(trimatl(R.t()), m));
  //NumericVector z_ = rnorm(n*mu.size());
  //mat z(z_.begin(), mu.size(), n, false, false);
  mat z(mu.size(), n);
  std::generate(z.begin(), z.end(), znorm);
  mat res = solve(trimatu(chol(Phi)), z);
  res.each_col() += mu;
  return(res);
}



// HMC TMVN sampler (canonical frame)
void get_hit_time(double &hit_time, int &cn, mat &F, vec &g, vec &a, vec &b, double &min_t) {
  //double min_t = .00001;
  hit_time = 0;
  cn = -1;
  vec Fa = F*a;
  vec Fb = F*b;
  for (size_t i=0; i != F.n_rows; i++ ){
    //LinearConstraint lc = linearConstraints[i];
    double fa = Fa(i); //(lc.f).dot(a);
    double fb = Fb(i); //(lc.f).dot(b);
    double u = sqrt(fa*fa + fb*fb);
    if (u>g(i) && u>-g(i)){
      
      double phi =atan2(-fa,fb);      //     -pi < phi < pi
      double t1 = acos(-g(i)/u)-phi;  //     -pi < t1 < 2*pi
      
      ////Rcout << t1;
      
      if (t1<0) t1 += 2*M_PI;                //  0 < t1 < 2*pi                  
      if (fabs(t1) < min_t ) t1=0;    
      else if (fabs(t1-2*M_PI) < min_t ) t1=0;              
      
      ////Rcout << t1;
      
      double t2 = -t1-2*phi;             //  -4*pi < t2 < 3*pi
      if (t2<0) t2 += 2*M_PI;                 //-2*pi < t2 < 2*pi
      if (t2<0) t2 += 2*M_PI;                 //0 < t2 < 2*pi
      
      if (fabs(t2) < min_t ) t2=0;    
      else if (fabs(t2-2*M_PI) < min_t ) t2=0;                            
      
      //Rcout << "solutions: " << t1 << ' ' << t2 << '\n';
      double t=t1;                
      if (t1==0) t = t2;
      else if (t2==0) t = t1;
      else t=(t1<t2?t1:t2);
      //Rcout << "soln + min_t: " << t << ' '<< min_t<< ' '<<'\n';
      
      if  ((t> min_t)  && (hit_time == 0 || t < hit_time)){
        //Rcout << "SOMETHINGS HAPPENING";
        hit_time=t;
        cn =i;                    
      }            
    }       
  }
}


//overwrites x0 with the result!
  void hmc_sample(int &q, vec &x0, mat &F, vec &g, double &t_total, double &min_t) {
    if(t_total<0) t_total = R::runif(0.0, M_PI);
    double velsign = 1;
    vec s(q);
    //vec x;
    vec a(q); vec b(q); vec hit_vel(q); vec res(q); vec ck(q);
    double hit_time;
    int hit_idx;
    
    while(2) {
      std::generate(s.begin(), s.end(), znorm);
      vec x = x0;
      double tt = t_total;
      
      a = s;
      b = x;
      
      while(1) {
        R_CheckUserInterrupt();
        
        get_hit_time(hit_time, hit_idx, F, g, a, b, min_t);
        
        if((hit_time < min_t) | (tt < hit_time)) { break; }
        
        tt -= hit_time;
        
        vec newb = sin(hit_time)*a + cos(hit_time)*b;
        hit_vel  = cos(hit_time)*a - sin(hit_time)*b;
        b = newb;
        
        //double f2 = as_scalar(F.row(hit_idx)*F.row(hit_idx).t());
        double f2 = accu(pow(F.row(hit_idx), 2));
        double alpha = as_scalar(F.row(hit_idx)*hit_vel)/f2;
        a = hit_vel - 2*alpha*F.row(hit_idx).t();
        velsign = as_scalar(dot(a, F.row(hit_idx)));
        if(velsign<0) break; //numerical check
        
      }
      
      if(!(velsign<0)) {
        res = sin(tt)*a + cos(tt)*b;
        ck = F*res + g;
        if(!any_lt0(ck)) {
          break;
        }
        //Rcout << "constraints violated" << endl;
        //Rcout << ck.t();
      }
    }
    x0 = res;
    //return(res);
  }

// // [[Rcpp::export]]
mat hmc_sampler(int n, vec &x0, mat &F, vec &g, double t_total, double min_t ) {
  int q = x0.size();
  mat result = mat(q, n);
  vec current = x0;
  for(int i=0; i<n; i++) {
    hmc_sample(q, current, F, g, t_total, min_t);
    result.col(i) = current;
  }
  return(result.t());
}


