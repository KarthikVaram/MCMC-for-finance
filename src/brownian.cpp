// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>
#include <random>
using namespace Rcpp;

//' Simulate standard Brownian motion paths
//'
//' Each path is a discretisation of W_t on [0, T] with n_steps steps.
//' Brownian increments: dW ~ N(0, dt)
//'
//' @param n_paths  Number of independent paths
//' @param n_steps  Number of time steps
//' @param T        Total time horizon
//' @param seed     Random seed
//' @return NumericMatrix (n_paths x (n_steps+1)) where column 0 = 0
//' @export
// [[Rcpp::export]]
NumericMatrix simulate_bm(int n_paths, int n_steps, double T = 1.0, int seed = 42) {
    double dt   = T / n_steps;
    double sqdt = std::sqrt(dt);

    std::mt19937 engine(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    // Rows = paths, Cols = time points (0 .. n_steps)
    NumericMatrix W(n_paths, n_steps + 1);

    for (int p = 0; p < n_paths; ++p) {
        W(p, 0) = 0.0;
        for (int t = 1; t <= n_steps; ++t)
            W(p, t) = W(p, t - 1) + sqdt * dist(engine);
    }
    return W;
}
