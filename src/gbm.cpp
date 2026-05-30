// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>
#include <random>
#include <cmath>
using namespace Rcpp;

//' Simulate Geometric Brownian Motion paths
//'
//' Exact discretisation:
//'   S_{t+dt} = S_t * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
//' where Z ~ N(0,1).
//'
//' @param S0       Initial stock price
//' @param mu       Drift (annualised)
//' @param sigma    Volatility (annualised)
//' @param T        Time horizon in years
//' @param n_steps  Number of time steps
//' @param n_paths  Number of simulated paths
//' @param seed     Random seed
//' @return NumericMatrix (n_paths x (n_steps+1)), column 0 = S0
//' @export
// [[Rcpp::export]]
NumericMatrix simulate_gbm(double S0, double mu, double sigma,
                            double T, int n_steps, int n_paths,
                            int seed = 42) {
    double dt      = T / n_steps;
    double drift   = (mu - 0.5 * sigma * sigma) * dt;
    double vol     = sigma * std::sqrt(dt);

    std::mt19937 engine(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    NumericMatrix S(n_paths, n_steps + 1);

    for (int p = 0; p < n_paths; ++p) {
        S(p, 0) = S0;
        for (int t = 1; t <= n_steps; ++t)
            S(p, t) = S(p, t - 1) * std::exp(drift + vol * dist(engine));
    }
    return S;
}

//' Return only the terminal stock prices (last column of GBM)
//'
//' Useful for pricing European options without storing full paths.
//'
//' @param S0       Initial stock price
//' @param mu       Drift
//' @param sigma    Volatility
//' @param T        Time horizon
//' @param n_steps  Time steps
//' @param n_paths  Paths
//' @param seed     Seed
//' @return NumericVector of length n_paths
//' @export
// [[Rcpp::export]]
NumericVector gbm_terminal(double S0, double mu, double sigma,
                            double T, int n_steps, int n_paths,
                            int seed = 42) {
    double dt    = T / n_steps;
    double drift = (mu - 0.5 * sigma * sigma) * dt;
    double vol   = sigma * std::sqrt(dt);

    std::mt19937 engine(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    NumericVector ST(n_paths);
    for (int p = 0; p < n_paths; ++p) {
        double s = S0;
        for (int t = 0; t < n_steps; ++t)
            s *= std::exp(drift + vol * dist(engine));
        ST[p] = s;
    }
    return ST;
}
