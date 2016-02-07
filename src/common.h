#ifndef common_h
#define common_h

//#define NDEBUG

#define BOOST_DISABLE_ASSERTS
#include <boost/assert.hpp>

#include <RcppArmadillo.h>
#include <algorithm> 
using namespace Rcpp;
using namespace arma;

inline double logsumexp(const double &a, const double &b){
  return a < b ? b + log(1.0 + exp(a - b)) : a + log(1.0 + exp(b - a));
}

template <class T>
double logsumexp(T &x) {
  double m = *std::max_element(x.begin(), x.end());
  double s = 0.0;
  typename T::iterator it;
  for(it=x.begin(); it!=x.end(); ++it) {
    s += exp(*it-m);
  }
  return(m+log(s));
}

// compute X^{-1}y given cholesky factor R (X = R'R)
vec chol_solve(mat &R, vec &y);

inline double myexp(double &x) { return(exp(x)); }


template <class T, class U>
T tabulate(U x, int k, int base=1) {
  T res(k);
  std::fill(res.begin(), res.end(), 0);
  for(typename U::iterator it=x.begin(); it != x.end(); ++it) {
    ++res((*it)-base);
  }
  return(res);
}

template<class T>
NumericMatrix cluster_crosstab(IntegerVector &Z, T const &X, int k, int cj, int base=1) {
  NumericMatrix result(cj, k);
  typename T::iterator itx;
  IntegerVector::iterator itz;
  for(itx=X.begin(), itz=Z.begin(); itx != X.end(); ++itx, ++itz) {
    if(*itx > 0) {
      ++result((*itx)-1, (*itz)-1);
    }
  }
  return(result);
}

template <class T>
bool any_lt0(typename T::iterator start, typename T::iterator end) {
  typename T::iterator it;
  for(it=start; it!=end; ++it) {
    if(*it<0) { break; }
  }
  return(it!=end);
}


template <class T>
bool any_lt0(T x) {
  typename T::iterator it;
  for(it=x.begin(); it!=x.end(); ++it) {
    if(*it<0) { break; }
  }
  return(it!=x.end());
}


mat arma_rbind_all(std::vector<vec> &rows);
imat arma_rbind_all(std::vector<ivec> &rows);

typedef std::vector<std::vector<int> > lookup_t;
lookup_t make_lookup(imat lookup_table, ivec cx);


#endif
