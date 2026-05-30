// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>
#include <random>
using namespace Rcpp;

//' Generate N standard normal samples N(0,1) using Mersenne Twister
//'
//' @param n Number of samples
//' @param seed Random seed (default 42)
//' @return NumericVector of length n
//' @export
// [[Rcpp::export]]
NumericVector rng_normal(int n, int seed = 42) {
    std::mt19937 engine(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    NumericVector out(n);
    for (int i = 0; i < n; ++i)
        out[i] = dist(engine);
    return out;
}

//' Generate N uniform [0,1) samples using Mersenne Twister
//'
//' @param n Number of samples
//' @param seed Random seed (default 42)
//' @return NumericVector of length n
//' @export
// [[Rcpp::export]]
NumericVector rng_uniform(int n, int seed = 42) {
    std::mt19937 engine(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    NumericVector out(n);
    for (int i = 0; i < n; ++i)
        out[i] = dist(engine);
    return out;
}

//' Verify RNG: return mean and variance of N(0,1) samples
//'
//' @param n Number of samples
//' @param seed Random seed
//' @return Named NumericVector with mean and variance
//' @export
// [[Rcpp::export]]
NumericVector rng_verify(int n = 100000, int seed = 42) {
    NumericVector s = rng_normal(n, seed);
    double m  = mean(s);
    double v  = var(s);
    return NumericVector::create(Named("mean") = m, Named("variance") = v);
}
