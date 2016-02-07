#include "common.h"
#include <RcppArmadillo.h>
#include <algorithm> 
using namespace Rcpp;
using namespace arma;


// compute X^{-1}y given cholesky factor R (X = R'R)
vec chol_solve(mat &R, vec &y) {
  return(solve(trimatu(R), solve(trimatl(R.t()), y)));
}


mat arma_rbind_all(std::vector<vec> &rows) {
  int n = rows.size();
  int p = rows[0].size();
  mat out(p, n);
  for(int i=0; i<n; ++i) {
    out.col(i) = rows[i];
  }
  return(out.t());
}

imat arma_rbind_all(std::vector<ivec> &rows) {
  int n = rows.size();
  int p = rows[0].size();
  imat out(p, n);
  for(int i=0; i<n; ++i) {
    out.col(i) = rows[i];
  }
  return(out.t());
}

lookup_t make_lookup(imat lookup_table, ivec cx) {
  lookup_t result;
  std::vector<int> tmp;
  int i=0;
  for(int j=0; j<cx.size(); ++j) {
    for(int c=0; c<cx[j]; ++c) {
      int tt = lookup_table(i,2);
      tmp.push_back(tt);
      ++i;
    }
    result.push_back(tmp);
    tmp.clear();
  }
  return result;
}
